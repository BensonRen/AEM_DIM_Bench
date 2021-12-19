"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil

# Torch
import torch
# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import VAE
from utils.helper_functions import put_param_into_folder,write_flags_and_BVE

def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(VAE, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)


def retrain_different_dataset(index):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    from utils.helper_functions import load_flags
    # data_set_list = ["Peurifoy"]
    data_set_list = ["Chen"]
    # data_set_list = ["Yang"]
    #data_set_list = ["Peurifoy","Chen","Yang_sim"]
    for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model+"_best_model"))
        flags.model_name = "retrain" + str(index) + eval_model
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)

def hyperswipe():
    """
    This is for doing hyperswiping for the model parameters
    """
    reg_scale_list = [1e-4]
    #reg_scale_list = [1e-3,  1e-4,  5e-3]
    kl_coeff_list = [0.1]# 
    #kl_coeff_list = [5e-2, 0.1, 1, 5] 
    #kl_coeff_list = [5e-2, 0.1, 1, 5] 
    #kl_coeff_list = [5e-2, 0.1, 1, 5] 
    layer_size_list = [1000]
    dim_z_list = [10, 20]
    for kl_coeff in kl_coeff_list:
        for layer_num in range(8, 15, 2):
            for layer_size in layer_size_list:
                for dim_z in dim_z_list:
                    for reg_scale in reg_scale_list:
                        flags = flag_reader.read_flag()  	#setting the base case
                        flags.reg_scale = reg_scale
                        flags.dim_z = dim_z
                        flags.kl_coeff = kl_coeff
                        # Decoder arch
                        linear_d = [layer_size  for j in range(layer_num)]
                        linear_d[0] = flags.dim_y + flags.dim_z
                        linear_d[-1] = flags.dim_x
                        # Encoder arch
                        linear_e = [layer_size  for j in range(layer_num)]
                        linear_e[0] = flags.dim_y + flags.dim_x
                        linear_e[-1] = 2 * flags.dim_z
                        flags.linear_d = linear_d
                        flags.linear_e = linear_e
                        flags.model_name = flags.data_set + '_kl_coeff_'+str(kl_coeff) + '_layer_num_' + str(layer_num) + '_unit_' + str(layer_size) + '_dim_z_' + str(dim_z) + '_reg_scale_' + str(flags.reg_scale)
                        training_from_flag(flags)

if __name__ == '__main__':
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    # hyperswipe()
    # Call the train from flag function
    #training_from_flag(flags)

    # Do the retraining for all the data set to get the training 
    for i in range(10):
       retrain_different_dataset(i)





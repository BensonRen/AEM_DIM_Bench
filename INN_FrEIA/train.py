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
from model_maker import INN
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE

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
    ntwk = Network(INN, flags, train_loader, test_loader, ckpt_dir=flags.ckpt_dir)

    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)

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
    data_set_list = ["Peurifoy"]
    # data_set_list = ["Chen"]
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
    dim_pad_list = [10]
    lambda_mse_list = [0.0001]#, 0.0001]
    dim_z_list = [3,5]
    for dim_z in dim_z_list:
        for dim_pad in dim_pad_list:
            for couple_layer_num in [12, 14, 16]:#range(10, 20):    
                for lambda_mse in lambda_mse_list:
                    flags = flag_reader.read_flag()  	#setting the base case
                    flags.couple_layer_num = couple_layer_num
                    flags.lambda_mse = lambda_mse
                    flags.dim_z = dim_z
                    flags.dim_tot = flags.dim_y + flags.dim_z + dim_pad
                    #print("currently running flag", flags)
                    print(flags.data_set)
                    flags.model_name = flags.data_set + '_mid_layer_1024_couple_layer_num' + str(couple_layer_num) + 'labmda_mse' + str(lambda_mse) + '_lr_' + str(flags.lr) + '_dim_pad_' + str(dim_pad) + '_dim_z_' + str(flags.dim_z)
                    training_from_flag(flags)

if __name__ == '__main__':
    # Read the parameters to be set
    #flags = flag_reader.read_flag()

    # Call the train from flag function
    #training_from_flag(flags)
    # hyperswipe()
    # Do the retraining for all the data set to get the training for reproducibility
    for i in range(10):
       retrain_different_dataset(i)

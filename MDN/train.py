"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
# from model_maker import MDN
# from mdn_nips import MDN
# from mdn_tony_duan import MDN
from mdn_manu_joseph import MDN
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
    ntwk = Network(MDN, flags, train_loader, test_loader)
    
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)


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
    reg_scale_list =  [0.1, 0.01, 0.001]
    # reg_scale_list =  [1e-5, 0]
    # reg_scale_list =  [1e-4, 5e-4, 5e-5, 0]
    # layer_size_list = [100, 500, 1000]
    layer_size_list = [100, 500, 1000]
    num_gauss_list = [4]
    # num_gauss_list = [4, 8, 16, 32]
    #num_gauss_list = [5, 10, 15, 20, 25, 30]
    # dataset = 'Yang_sim' # 'Peurifoy' ## 'Chen' #  
    # dataset = 'Peurifoy' #'Yang_sim' # # 'Chen' #  
    dataset = 'Chen' # 'Yang_sim' # 'Peurifoy' ## 
    linear_start = {'Chen': 256, 'Peurifoy':201, 'Yang_sim':2000}
    linear_end = {'Chen':5, 'Peurifoy':8, 'Yang_sim':14}
    for reg_scale in reg_scale_list:
        # for layer_num in range(5,6):
        for layer_num in range(3, 10, 2):
        # for layer_num in range(10, 15, 2):
            for layer_size in layer_size_list:
                for num_gaussian in num_gauss_list:
                    flags = flag_reader.read_flag()  	#setting the base case
                    flags.data_set = dataset
                    flags.reg_scale = reg_scale
                    linear = [layer_size  for j in range(layer_num)]
                    linear[0] = linear_start[dataset]
                    linear[-1] = linear_end[dataset]
                    flags.linear = linear
                    flags.num_gaussian = num_gaussian
                    flags.model_name = flags.data_set + '_gaussian_'+str(num_gaussian) + '_layer_num_' + str(layer_num) + '_unit_' + str(layer_size) + '_lr_' + str(flags.lr) + '_reg_scale_' + str(reg_scale)
                    training_from_flag(flags)
                    # quit()
                    # try:
                    #     training_from_flag(flags)
                    # except RuntimeError as e:
                    #     print("Failing the device-side assert for MDN mdn.sample function! doing 3 retries now:")
                    #     for j in range(3):
                    #         try:
                    #             print("trying number ", j)
                    #             training_from_flag(flags)
                    #             break;
                    #         except:
                    #             print("Failing again! try again")
                                    
                                    


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

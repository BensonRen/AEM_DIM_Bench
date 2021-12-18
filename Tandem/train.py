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
from model_maker import Forward, Backward
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
    ntwk = Network(Forward, Backward, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir, forward_best_loss=ntwk.best_forward_validation_loss)
    # put_param_into_folder(ntwk.ckpt_dir)


def retrain_different_dataset(index):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     from utils.helper_functions import load_flags
     #data_set_list = ["ballistics"]
     data_set_list = ["robotic_arm", "sine_wave", "ballistics", "meta_material"]
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model))
        flags.model_name = "retrain" + str(index) + eval_model
        flags.geoboundary = [-1,1,-1,1]
        flags.batch_size = 1024
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)
def hyperswipe():
    """
    This is for doing hyperswiping for the model parameters
    """
    reg_scale_list = [1e-4]
    #reg_scale_list = [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    layer_size_list = [1500]
    for reg_scale in reg_scale_list:
        for i in range(3):
            for layer_num in range(12, 17):
                for layer_size in layer_size_list:
                        flags = flag_reader.read_flag()  	#setting the base case
                        # Decoder arch
                        linear_b = [layer_size  for j in range(layer_num)]
                        linear_b[0] = 201
                        linear_b[-1] = 8
                        #flags.conv_out_channel_b = [4, 4, 4]
                        #flags.conv_kernel_size_b = [3,3,4]
                        #flags.conv_stride_b = [1,1,2]
                        flags.linear_b = linear_b
                        flags.reg_scale = reg_scale
                        flags.model_name = flags.data_set + '_Backward_no_conv_layer_num_' + str(layer_num) + '_unit_' + str(layer_size)  + '_reg_scale_' + str(flags.reg_scale) + '_trail_' + str(i)
                        #flags.model_name = flags.data_set + '_Backward_conv_444_334_112_layer_num_' + str(layer_num) + '_unit_' + str(layer_size)  + '_reg_scale_' + str(flags.reg_scale) + '_trail_' + str(i)
                        training_from_flag(flags)



if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    
    hyperswipe()
    # Call the train from flag function
    #for i in range(20):
    #training_from_flag(flags)

    # Do the retraining for all the data set to get the training 
    #for i in range(10):
    #    retrain_different_dataset(i)

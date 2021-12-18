"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import random
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import NA
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
    ntwk = Network(NA, flags, train_loader, test_loader)

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
     #data_set_list = ["ballistics"]
     data_set_list = ["Peurifoy"]
     #data_set_list = ["Peurifoy","Chen","Yang_sim"]
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model+"_best_model"))
        flags.linear[0] = 8
        flags.model_name = "retrain" + str(index) + eval_model
        flags.geoboundary = [-1, 1, -1, 1]     # the geometry boundary of meta-material dataset is already normalized in current version
        flags.train_step = 300
        flags.test_ratio = 0.2
        training_from_flag(flags)

def hyperswipe(dataset, rep=1):
    """
    This is for doing hyperswiping for the model parameters
    """
    layer_size_list = [5000, 1000]
    #layer_size_list = [1750]
    #layer_size_list = [100, 250, 500]
    reg_scale_list = [1e-4]
    #layer_num = 7
    for reg_scale in reg_scale_list:
        for i in range(1):
            for layer_num in range(8, 25, 2):
                for layer_size in layer_size_list:
                    flags = flag_reader.read_flag()  	#setting the base case
                    linear = [layer_size for j in range(layer_num)]        #Set the linear units
                    linear[0] = 10                   # The start of linear
                    linear[-1] = 1001                # The end of linear
                    flags.lr = 1e-4
                    flags.linear = linear
                    flags.reg_scale = reg_scale
                    #flags.conv_kernel_size = [3, 3, 5]
                    #flags.conv_channel_out = [4, 4, 4]
                    #flags.conv_stride = [1, 1, 1]
                    flags.model_name = flags.data_set + 'no_conv_' + str(layer_size) + '_num_' + str(layer_num) + '_lr_' + str(flags.lr) + 'reg_scale_' + str(reg_scale) + 'trail_' + str(i)
                    #flags.model_name = flags.data_set + 'conv_444_335_111_linear_' + str(layer_size) + '_num_' + str(layer_num) + '_lr_' + str(flags.lr) + 'reg_scale_' + str(reg_scale) + 'trail_' + str(i)
                    training_from_flag(flags)
                #except:
                #    print("Probably a bad configuration")

                dirs = os.listdir(spec_dir)

    """
    # Ashwin's version

    for set in dataset:
        # Setup dataset folder
        spec_dir = os.path.join('models', set)

        if not os.path.exists(spec_dir):
            os.mkdir(spec_dir)

        dirs = os.listdir(spec_dir)

        # Clean up unfinished runs
        for run in dirs:
            d = os.path.join(spec_dir,run)
            for f in os.listdir(d):
                if f.find("training time.txt") != -1:
                    break
            else:
                shutil.rmtree(d)

        stride = []
        kernel = []

        if 'Chen' in set:

            # Faster drops helped, but did add instability. lr 0.1 was too large, reg 1e-5 reduced instability but not enough
            # reg 1e-4 combined with faster decay = 0.3 and lower starting lr = 0.001 has great, consistent results
            # ^ reg = 0 -> High instability but results similar, reg = 1e-5 -> instability is lower
            # reg 1e-5 shown to be better than 0 or 1e-4, can be improved if lr gets lower later though lr=e-4,lr_decay=.1

            reg_scale_list = [1e-5]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.0001,0.001,0.01]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.1,0.3,0.4]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (5,256)

        elif 'Peurifoy' in set:
            reg_scale_list = [1e-4]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [11,13,15,17]
            layer_size_list = [1300,1500,1700]  # [1900,1500,1100,900,700,500,300]
            lrate = [1e-4]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.6,0.7,0.8]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (8,201)

        elif 'Yang' in set:
            reg_scale_list = [1e-4]  # [1e-4, 1e-3, 1e-2, 1e-1]
            stride = 0
            kernel = 0
            layer_num = [13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.001]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.3]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (14,2000)

        else:
            return 0

        for reg_scale in reg_scale_list:
            for ln in layer_num:
                for ls in layer_size_list:
                    for lr in lrate:
                        for ld in lr_decay:
                            for i in range(rep):
                                # If this combination has been tested before, name appropriately
                                hyp_config = '_'.join(map(str, (ln, ls, lr, reg_scale, ld)))  # Name by hyperparameters

                                num_configs = 0                     # Count number of test instances
                                for configs in dirs:
                                    if hyp_config in configs:
                                        num_configs += 1

                                if num_configs >= rep:              # If # instances >= reps, make extra reps required or skip
                                    continue

                                name = '_'.join((hyp_config,str(num_configs)))

                                # Model run
                                flags = flag_reader.read_flag()
                                flags.data_set = set                               # Save info
                                flags.model_name = os.path.join(set,name)

                                flags.linear = [ls for j in range(ln)]              # Architecture
                                flags.linear[0] = ends[0]
                                flags.linear[-1] = ends[-1]
                                flags.conv_stride = stride
                                flags.conv_kernel_size = kernel

                                flags.lr = lr                                       # Other params
                                flags.lr_decay_rate = ld
                                flags.reg_scale = reg_scale
                                flags.batch_size = 1024
                                flags.train_step = 300
                                flags.normalize_input = True

                                training_from_flag(flags)

                                dirs = os.listdir(spec_dir)         # Update dirs to include latest run

def random_grid_search(dataset, rep=1):
    for set in dataset:
        # Setup dataset folder
        spec_dir = os.path.join('models', set)

        if not os.path.exists(spec_dir):
            os.mkdir(spec_dir)

        dirs = os.listdir(spec_dir)

        # Clean up unfinished runs
        for run in dirs:
            d = os.path.join(spec_dir, run)
            for f in os.listdir(d):
                if f.find("training time.txt") != -1:
                    break
            else:
                shutil.rmtree(d)

        stride_vals = None
        kernel_vals = None

        if 'Chen' in set:
            reg_scale_list = [1e-5]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.0001, 0.001]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.1, 0.3]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (5,256)

        elif 'Peurifoy' in set:
            kernel_vals = [3, 4, 5, 6, 7]
            stride_vals = [1, 1, 1]
            reg_scale_list = [1e-3, 1e-4, 1e-5, 0]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [5, 10, 15, 20]
            layer_size_list = [300, 500, 1000, 1500, 2000]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.1, 1e-2, 1e-4]
            lr_decay = [0.1, 0.3, 0.5, 0.7, 0.9]
            ends = (8, 201)

        elif 'Yang' in set:
            reg_scale_list = [1e-3, 1e-4, 1e-5, 0]  # [1e-4, 1e-3, 1e-2, 1e-1]
            stride_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            kernel_vals = [3, 4, 5, 6, 7]
            layer_num = [17, 13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.1, 0.01, 0.001, 0.0001, 1e-5]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.1, 0.3, 0.5, 0.7]
            ends = (14,2000)
        else:
            return 0

        stride = []
        kernel = []

        while(True):
            ln = random.choice(layer_num)
            ls = random.choice(layer_size_list)
            lr = random.choice(lrate)
            reg_scale = random.choice(reg_scale_list)
            ld = random.choice(lr_decay)

            conv_config = ''
            if stride_vals and kernel_vals:
                num_convs = random.randrange(4)
                stride = []
                kernel = []

                if num_convs > 0:
                    for el in range(num_convs):
                        stride.append(random.choice(stride_vals))
                        kernel.append(random.choice(kernel_vals))

                    mat = ['kernel'] + list(map(str, kernel)) + ['stride'] + list(map(str, stride))
                    print(mat)
                    conv_config = '-'.join(mat)

            for i in range(rep):
                # If this combination has been tested before, name appropriately
                hyp_config = '_'.join(map(str, (ln, ls, lr, reg_scale, ld, conv_config)))  # Name by hyperparameters

                num_configs = 0  # Count number of test instances
                for configs in dirs:
                    if hyp_config in configs:
                        num_configs += 1

                if num_configs >= rep:  # If # instances >= reps, make extra reps required or skip
                    continue

                name = '_'.join((hyp_config, str(num_configs)))

                # Model run
                flags = flag_reader.read_flag()
                flags.data_set = set  # Save info
                flags.model_name = os.path.join(set, name)

                flags.linear = [ls for j in range(ln)]  # Architecture
                flags.linear[-1] = ends[-1]
                flags.conv_stride = [1]*len(kernel)
                flags.conv_kernel_size = kernel
                flags.conv_out_channel = [4]*len(kernel)
                flags.linear[0] = ends[0]

                flags.lr = lr  # Other params
                flags.lr_decay_rate = ld
                flags.reg_scale = reg_scale
                flags.batch_size = 1024
                flags.train_step = 300
                flags.normalize_input = True

                try:
    """

if __name__ == '__main__':
    # Read the parameters to be set
    """
    for i in range(5):
        flags.model_name = 'Yang_param_pure_' + str(i+15)
        #linear = [1000  for j in range(11)]
        #linear[0] = 14
        #linear[-1] = 500
        #flags.linear = linear
        #flags.reg_scale = 1e-4
        training_from_flag(flags)
    """
    #hyperswipe()
    flags = flag_reader.read_flag()  	#setting the base case
   
    # Ashwin part
    #random_grid_search(['Peurifoy'],rep=2)
    #hyperswipe(['Peurifoy'])
    #retrain_different_dataset(0)
    
    training_from_flag(flags)
    # Do the retraining for all the data set to get the training 
    #for i in range(10):
    #    retrain_different_dataset(i)


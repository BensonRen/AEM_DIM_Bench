"""
This file serves as a modulized evaluation interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import NA
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
from utils.evaluation_helper import get_test_ratio_helper
from utils.plotsAnalysis import get_xpred_ytruth_xtruth_from_folder
from utils.plotsAnalysis import reshape_xpred_list_to_mat 
from utils.create_folder_modulized import get_folder_modulized
from utils.create_folder_modulized import check_modulized_yet
# Libs
import numpy as np
import matplotlib.pyplot as plt
from thop import profile, clever_format


def modulized_evaluate_from_model(model_dir, operate_dir, FF=False, BP=False):

    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param operate_dir: The directory to operate in (with all the Xpred,Ypred files)
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print(model_dir)
    flags = load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    if BP:
        flags.backprop_step = 300 
    else:
        flags.backprop_step = 1 
    flags.test_ratio = get_test_ratio_helper(flags)

    if flags.data_set == 'meta_material':
        save_Simulator_Ypred = False
        print("this is MM dataset, there is no simple numerical simulator therefore setting the save_Simulator_Ypred to False")
    flags.batch_size = 1                            # For backprop eval mode, batchsize is always 1
    flags.lr = 0.5
    flags.eval_batch_size = 2048
    flags.train_step = 500

    print(flags)
    
    # Make Network
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)

    # Set up the files
    Xpred_list, Xt, Yt = get_xpred_ytruth_xtruth_from_folder(operate_dir)
    X_init_mat = reshape_xpred_list_to_mat(Xpred_list)

    # Evaluation process
    print("Start eval now:")
    ntwk.modulized_bp_ff(X_init_mat=X_init_mat, Ytruth=Yt, save_dir=operate_dir, save_all=True, FF=FF)


def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            modulized_evaluate_from_model(os.path.join(models_dir, file))
    return None

def get_state_of_BP_FF(folder):
    """
    This function return 2 flag for BP and FF according to the folder name given
    """
    # Get the label of the state of BP and FF
    if 'BP_on' in folder:
        BP = True
    elif 'BP_off' in folder:
        BP = False
    else:
        print("Your folder name does not indicate state of BP: ", folder)
        exit()
    if 'FF_on' in folder:
        FF = True
    elif 'FF_off' in folder:
        FF = False
    else:
        print("Your folder name does not indicate state of FF: ", folder)
        exit()
    return BP, FF

def modulized_evaluate_different_dataset(gpu=None):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    #data_set_list = ["meta_material"]
    data_set_list = ["robotic_arm","sine_wave","ballistics",]
    folder_list = get_folder_modulized(gpu=gpu)
    for folder in folder_list:
        # Skip Random for now
        #if 'Random' not in folder:
        #    continue;
        BP, FF = get_state_of_BP_FF(folder)
        # Nothing is needed if both of them are False
        if BP is False and FF is False:
            continue;
        print("currently working on folder", folder)
        # Work on each dataset
        for dataset in data_set_list:
            if check_modulized_yet(os.path.join(folder, dataset)):
                continue;
            modulized_evaluate_from_model(model_dir="retrain0" + dataset,
                                      operate_dir=os.path.join(folder, dataset), BP=BP, FF=FF)

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    #eval_flags = flag_reader.read_flag()

    #####################
    # different dataset #
    #####################
    # This is to run the single evaluation, please run this first to make sure the current model is well-trained before going to the multiple evaluation code below
    #evaluate_different_dataset(multi_flag=False, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
    # This is for multi evaluation for generating the Fig 3, evaluating the models under various T values
    #evaluate_different_dataset(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)


    #####################
    #Modulized eval Here#
    #####################
    modulized_evaluate_different_dataset()

"""
This file serves as a evaluation interface for the network
"""
# Built in
from math import inf
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
# Libs
import numpy as np
import shutil
#import matplotlib.pyplot as plt
#from thop import profile, clever_format

def predict(model_dir, Ytruth_file ,multi_flag=False):
    """
    Predict the output from given spectra
    """
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    if model_dir.startswith('/'):                   # It is a absolute path
        flags = load_flags(model_dir)
    else:
        flags = load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    #pred_file, truth_file = ntwk.predict_inverse(Ytruth_file, multi_flag)
    #if 'Yang' not in flags.data_set:
    #    plotMSELossDistrib(pred_file, truth_file, flags)


def predict_different_dataset(multi_flag=False):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    step_func_dir = '/home/sr365/MM_Bench/Data/step_func'
    for model in os.listdir('models/'):
        if 'best' in model:# and 'Chen' in model:
            if 'Yang' in model:
                Ytruth_file = os.path.join(step_func_dir, 'Yang'+'step_function.txt')
            elif 'Chen' in model:
                Ytruth_file = os.path.join(step_func_dir, 'Chen'+'step_function.txt')
            elif 'Peurifoy' in model:
                Ytruth_file = os.path.join(step_func_dir, 'Peurifoy'+'step_function.txt')
            predict(model, Ytruth_file, multi_flag=multi_flag)

def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False, save_misc=False, 
                        MSE_Simulator=False, save_Simulator_Ypred=True, 
                        init_lr=0.01, lr_decay=0.9, BDY_strength=1, save_dir='data/',
                        noise_level=0, 
                        md_coeff=0, md_start=None, md_end=None, md_radius=None,
                        eval_batch_size=None):

    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param multi_flag: The switch to turn on if you want to generate all different inference trial results
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
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
    flags.test_ratio = get_test_ratio_helper(flags)
    flags.backprop_step = eval_flags.backprop_step
    #flags.test_ratio = 0.02

    if flags.data_set != None: #== 'Yang_sim':
        save_Simulator_Ypred = False
        print("this is Yang sim dataset, setting the save_Simulator_Ypred to False")
    flags.batch_size = 1                            # For backprop eval mode, batchsize is always 1
    flags.BDY_strength = BDY_strength
    flags.train_step = eval_flags.train_step
    flags.backprop_step = 300 

    # MD Loss: new version
    if md_coeff is not None:
        flags.md_coeff = md_coeff
    if md_start is not None:
        flags.md_start = md_start
    if md_end is not None:
        flags.md_end = md_end
    if md_radius is not None:
        flags.md_radius = md_radius

    ############################# Thing that are changing #########################
    flags.lr = init_lr
    flags.lr_decay_rate = lr_decay
    flags.eval_batch_size = 2048 if eval_batch_size is None else eval_batch_size
    flags.optim = 'Adam'
    ###############################################################################
    
    print(flags)

    # if flags.data_set == 'Peurifoy':
    #     flags.eval_batch_size = 10000
    # elif flags.data_set == 'Chen':
    #     flags.eval_batch_size = 10000
    # elif flags.data_set == 'Yang' or flags.data_set == 'Yang_sim':
    #     flags.eval_batch_size = 2000
    #
    # flags.batch_size = flags.eval_batch_size

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")
    
    # Make Network
    ntwk = Network(NA, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # For calculating parameter 
    # pred_file, truth_file = ntwk.validate_model(save_dir='data/' + flags.data_set+'_best_model', save_misc=save_misc,
    #                                       MSE_Simulator=MSE_Simulator, save_Simulator_Ypred=save_Simulator_Ypred)

    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        #dest_dir = '/home/sr365/mm_bench_multi_eval_Chen_sweep/NA_init_lr_{}_decay_{}_batch_{}_bp_{}_noise_lvl_{}/'.format(init_lr, lr_decay, flags.eval_batch_size, flags.backprop_step, noise_level)
        #dest_dir = '/home/sr365/mm_bench_compare_MDNA_loss/NA_init_lr_{}_decay_{}_MD_loss_{}'.format(flags.lr, flags.lr_decay_rate, flags.md_coeff)
        dest_dir = '/home/sr365/mm_bench_multi_eval/NA'
        #dest_dir = '/home/sr365/MM_bench_multi_eval/NA_RMSprop/'
        #dest_dir = '/data/users/ben/multi_eval/NA_lr' + str(init_lr)  + 'bdy_' + str(BDY_strength)+'/' 
        #dest_dir = os.path.join('/home/sr365/MDNA_temp/', save_dir)
        dest_dir = os.path.join(dest_dir, flags.data_set)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        #pred_file, truth_file = ntwk.evaluate(save_dir='/work/sr365/multi_eval/NA/' + flags.data_set, save_all=True,
        pred_file, truth_file = ntwk.evaluate(save_dir=dest_dir, save_all=True,
                                                save_misc=save_misc, MSE_Simulator=MSE_Simulator,
                                                save_Simulator_Ypred=save_Simulator_Ypred,
                                                noise_level=noise_level)
    else:
        # Creat the directory is not exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        pred_file, truth_file = ntwk.evaluate(save_dir=save_dir, save_misc=save_misc,
                                             MSE_Simulator=MSE_Simulator, 
                                             save_Simulator_Ypred=save_Simulator_Ypred,
                                             noise_level=noise_level)
        #pred_file, truth_file = ntwk.evaluate(save_dir='data/'+flags.data_set,save_misc=save_misc, MSE_Simulator=MSE_Simulator, save_Simulator_Ypred=save_Simulator_Ypred)
    return 
    if 'Yang' in flags.data_set:
        return
    # Plot the MSE distribution
    MSE = plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")
    return MSE


def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None

def evaluate_different_dataset(multi_flag, eval_data_all, save_Simulator_Ypred=False, MSE_Simulator=False):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    ## Evaluate all models with "reatrain" and dataset name in models/
    for model in os.listdir('models/'):
        if 'best' in model and 'Peurifoy' in model: 
        #if 'best' in model and 'Chen' in model: 
            evaluate_from_model(model, multi_flag=multi_flag, 
                        eval_data_all=eval_data_all,save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator)
    
    # Single model evaluation
    #model = 'Peurifoy_best_model'
    #evaluate_from_model(model, multi_flag=multi_flag, 
    #                 eval_data_all=eval_data_all,save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator)


def hyper_sweep_evaluation(multi_flag, eval_data_all, save_Simulator_Ypred=False, MSE_Simulator=False,
                         save_file='hypersweep_results.txt'):
    """
    hyper sweeping the evlauation parameters here
    """
    dataset = 'Chen'
    #dataset = 'Peurifoy'
    with open(save_file, 'a') as fout:
        for model in os.listdir('models/'):
            if 'best' in model and dataset in model: 
            #if 'best' in model and 'Chen' in model: 
                # Sweeping the eval batch size
                #for eval_batch_size in [5, 10, 20, 50, 100, 200, 500]:
                #for eval_batch_size in [10,  50, 2048]:
                for eval_batch_size in [100, 500, 1000]:
                #for eval_batch_size in [1000]:
                #for eval_batch_size in [16384]:
                    for md_coeff in [1e-5, 1e-4, 1e-6, 5e-6, 5e-5, 5e-4]:
                    #for md_coeff in [1e-3, 1e-2]:
                    #for md_coeff in [0]:
                    #for md_coeff in [1e-3, 1e-2]:
                    #for md_coeff in [1e-5, 1e-4, 1e-3, 1e-2]:
                        #for md_radius in [ 0.001]:
                        for md_radius in [ 0.5, 0.1, 0.01, 0.001]:
                        #for md_radius in [0]:
                            md_start = -1
                            md_end = 150
                            
                            # Make the save directory
                            save_dir = 'extreme_data_{}_bs_{}_md_coeff_{}_md_radius_{}_md_start_{}_md_end_{}'.format(model, eval_batch_size,
                                    md_coeff, md_radius, md_start, md_end)
                            if not os.path.isdir(save_dir):
                                os.makedirs(save_dir)
                            
                            # Make the evaluation of the model
                            MSE = evaluate_from_model(model, multi_flag=multi_flag, eval_data_all=eval_data_all,
                                                save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator,
                                                eval_batch_size=eval_batch_size, md_coeff=md_coeff, md_radius=md_radius,
                                                md_start=md_start, md_end=md_end, save_dir=save_dir)
                            
                            # Writing this model to the recording file
                            #fout.write('Dataset = {}, For eval_batch_size = {}, md_coeff={}, md_radius={}, md_start={}, md_end={}, MSE={}'.format(model, eval_batch_size,
                            #        md_coeff, md_radius, md_start, md_end, MSE))
                            #fout.write('\n')

                            # # Deleting that diectory
                            # shutil.rmtree(save_dir)
                            # os.mkdir('data')

def evaluate_trail_BDY_lr(multi_flag, eval_data_all, save_Simulator_Ypred=False, MSE_Simulator=False):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    #lr_list = [0.01, 0.02, 0.03, 0.008]
    lr_list = [0.01]
    #lr_list = [0.001, 0.01, 0.1, 1, 10]
    lr_decay_rate_list = [0.9]
    noise_level_list = [8e-3, 9e-3]#, 1e-3, 1e-4, 1e-5]
    #data_set_list = ["robotic_arm"]
    #data_set_list = ["robotic_arm", "ballistics"]
    #for eval_model in data_set_list:
    for lr in lr_list:
        for lr_decay_rate in lr_decay_rate_list:
            for noise_level in noise_level_list:
                for model in os.listdir('models/'):
                    if 'best' in model and 'Chen' in model: 
                        evaluate_from_model(model, multi_flag=multi_flag, 
                                    eval_data_all=eval_data_all,save_Simulator_Ypred=save_Simulator_Ypred, 
                                    MSE_Simulator=MSE_Simulator, init_lr = lr, lr_decay = lr_decay_rate,
                                    noise_level=noise_level)#, BDY_strength=BDY)
     
    #  """
    #  This function is to evaluate all different datasets in the model with one function call
    #  """
    #  ## Evaluate all models with "reatrain" and dataset name in models/
    #  for model in os.listdir('models/'):
    #      print(model)
    #      if 'Peurifoy_best' in model:
    #          evaluate_from_model(model, multi_flag=multi_flag,
    #                       eval_data_all=eval_data_all,save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator)

def evaluate_trail_BDY_lr(multi_flag, eval_data_all, save_Simulator_Ypred=False, MSE_Simulator=False):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     #lr_list = [2, 1,0.5,0.1]
     lr_list = [0.5]
     BDY_list = [0.001]
     #BDY_list = [0.05, 0.01, 0.001]
     data_set_list = ["Chen"]
     #data_set_list = ["robotic_arm", "ballistics"]
     for eval_model in data_set_list:
        for lr in lr_list:
            for BDY in BDY_list:
                useless_flags = flag_reader.read_flag()
                useless_flags.eval_model = "retrain5" + eval_model
                evaluate_from_model(useless_flags.eval_model, multi_flag=multi_flag, eval_data_all=eval_data_all, save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator, init_lr = lr, BDY_strength=BDY)

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    eval_flags = flag_reader.read_flag()
    
    #####################
    # different dataset #
    #####################
    # This is to run the single evaluation, please run this first to make sure the current model is well-trained before going to the multiple evaluation code below
    #evaluate_different_dataset(multi_flag=False, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
    # This is for multi evaluation for generating the Fig 3, evaluating the models under various T values
    evaluate_different_dataset(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
    
    # This is to test the BDY and LR effect of the NA method specially for Robo and Ballistics dataset, 2021.01.09 code trail for investigating why sometimes NA constrait the other methods
    #evaluate_trail_BDY_lr(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)

    #hyper_sweep_evaluation(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
    
    ###########
    # Predict #
    ###########
    #predict_different_dataset(multi_flag=False)

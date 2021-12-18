"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from ga import GA_manager as GA
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
from utils.evaluation_helper import get_test_ratio_helper
# Libs
import numpy as np
import matplotlib.pyplot as plt
#from thop import profile, clever_format


def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False, save_misc=False, MSE_Simulator=False,
                        save_Simulator_Ypred=False, preset_flag=None,init_lr=0.5, BDY_strength=1):
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
    flags.eval_model = model_dir  # Reset the eval mode
    flags.generations = eval_flags.generations
    flags.test_ratio = get_test_ratio_helper(flags)

    if flags.data_set == 'Yang_sim':
        save_Simulator_Ypred = False
        print("this is MM dataset, setting the save_Simulator_Ypred to False")

    flags = preset_flag if preset_flag else flags
    flags.batch_size = 1 # For backprop eval mode, batchsize is always 1
    print(flags)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")

    # Make Network
    Genetic_Algorithm = GA(flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    print("Start eval now:")
    dname = flags.save_to

    if multi_flag:
        dest_dir = './temp-dat/'+dname+'/'
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        dest_dir += flags.data_set
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        pred_file, truth_file = Genetic_Algorithm.evaluate(save_dir=dest_dir, save_all=True,
                                              save_misc=save_misc, MSE_Simulator=MSE_Simulator,
                                              save_Simulator_Ypred=save_Simulator_Ypred)
    else:
        pred_file, truth_file = Genetic_Algorithm.evaluate(save_misc=save_misc, save_dir=dname,MSE_Simulator=MSE_Simulator,
                                              save_Simulator_Ypred=save_Simulator_Ypred)

    # Plot the MSE distribution
    #plotMSELossDistrib(pred_file, truth_file, flags,save_dir=dname)
    print("Evaluation finished")


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
    data_set_list = ["Peurifoy"]
    for eval_model in data_set_list:
        for j in range(1):
            useless_flags = flag_reader.read_flag()
            useless_flags.eval_model = 'retrain' + str(j) + eval_model
            evaluate_from_model(useless_flags.eval_model, multi_flag=multi_flag, eval_data_all=eval_data_all,
                                save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator)

def test_categorical_variables():
    data_set_list = ["Peurifoy","Chen","Yang_sim"]
    SOps = ["roulette","decimation","tournament"]
    XOps = ["uniform","single-point"]

    for i in [1,2]:
        for d in data_set_list:
            for x in XOps:
                for s in SOps:
                    dir = d + '_best_model'
                    flags = load_flags(os.path.join("models", dir))
                    flags.cross_operator = x
                    flags.selection_operator = s
                    flags.data_set = d
                    flags.eval_model = dir
                    flags.crossover = 0.8
                    flags.elitism = 500
                    flags.k = 500
                    flags.mutation = 0.05
                    flags.population = flags.eval_batch_size
                    flags.ga_eval = False
                    flags.generations = 10
                    flags.xtra = i

                    evaluate_from_model(flags.eval_model,preset_flag=flags,save_Simulator_Ypred=True)

def test_gen_pop():
    data_set_list =["Peurifoy","Chen","Yang_sim"]
    c = 0
    l = os.listdir('data/sweep03')
    for d in data_set_list:
        for p in [25,50,75,100,150]:
            for g in [10,25,50,100,150]:

                if d+'_'+str(g)+'_'+str(p) in l:
                    continue

                dir = d + '_best_model'
                flags = load_flags(os.path.join("models", dir))
                flags.cross_operator = 'single-point'
                flags.selection_operator = 'roulette'
                flags.data_set = d
                flags.eval_model = dir
                flags.crossover = 0.8
                flags.elitism = int(p/5)
                flags.k = int(p/5)
                flags.mutation = 0.05
                flags.ga_eval = False
                flags.generations = g
                flags.population = p
                flags.xtra = 50

                evaluate_from_model(flags.eval_model, preset_flag=flags, save_Simulator_Ypred=True)

def test_num_samples():
    # ds = ['Yang_sim']#'Chen']#'Peurifoy']#,,]
    # ds = ['Chen']#'Peurifoy']#,,
    ds = ['Peurifoy']#,]
    # gen = [50]
    # saveto = ['GA2_50_gaussian']
    gen = [300]
    saveto = ['Test_time']
    for i,dset in enumerate(ds):
        dxy = dset + '_best_model'
        flags = load_flags(os.path.join("models", dxy))
        flags.data_set = dset

        flags.cross_operator = 'single-point'
        flags.selection_operator = 'roulette'
        flags.eval_model = dxy
        flags.crossover = 0.8
        flags.elitism = 500
        flags.k = 500
        flags.mutation = 0.05
        flags.population = 2048
        flags.ga_eval = False
        flags.generations = gen[i]
        flags.xtra = 20
        flags.test_ratio = get_test_ratio_helper(flags)
        print('flags. test_ratio', flags.test_ratio)
        flags.save_to = saveto[i]

        # evaluate_from_model(flags.eval_model, preset_flag=flags, save_Simulator_Ypred=False, multi_flag=True)
        evaluate_from_model(flags.eval_model, preset_flag=flags, save_Simulator_Ypred=False, multi_flag=True)


if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    eval_flags = flag_reader.read_flag()

    test_num_samples()
    #test_categorical_variables()

    #####################
    # different dataset #
    #####################
    # This is to run the single evaluation, please run this first to make sure the current model is well-trained before going to the multiple evaluation code below
    #evaluate_different_dataset(multi_flag=False, eval_data_all=False, save_Simulator_Ypred=False, MSE_Simulator=False)
    # This is for multi evaluation for generating the Fig 3, evaluating the models under various T values
    # evaluate_different_dataset(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)

    # This is to test the BDY and LR effect of the NA method specially for Robo and Ballistics dataset, 2021.01.09 code trail for investigating why sometimes NA constrait the other methods
    # evaluate_trail_BDY_lr(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)

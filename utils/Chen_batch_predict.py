from utils.helper_functions import simulator
from multiprocessing import Pool
from utils.evaluation_helper import plotMSELossDistrib
import numpy as np
import os
import pandas as pd

# This is the script for doing batch evaluation
num_cpu = 32
def eval_chen_for_file(filename):
    # Read the Xpred file
    Xpred = pd.read_csv(filename, sep=' ', header=None).values
    # Run the simulator
    Ypred = simulator('Chen', Xpred)
    # Save the Ypred into the same folder with name change
    with open(filename.replace('Xpred','Ypred'), 'a') as fyp:
        np.savetxt(fyp, Ypred)

def eval_whole_folder(folder):
    """
    Run the eval chen for file in a loop 
    """
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for filename in os.listdir(folder):
            if not ('Chen' in filename and 'Xpred' in filename):
                continue
            args_list.append((os.path.join(folder, filename),))
        print((args_list))
        print(len(args_list))
        pool.starmap(eval_chen_for_file, args_list)
    finally:
        pool.close()
        pool.join()

def plot_MSE(folder):
    """
    Plot the MSE plots for the simulated pairs of MSE
    """
    for file in os.listdir(folder):
        if not ('Chen' in file and 'Xpred' in file):
                continue
        Xpred_file = os.path.join(folder, file)
        Ypred_file = Xpred_file.replace('Xpred','Ypred')
        Ytruth_file = Ypred_file.replace('Ypred','Ytruth')
        save_dir = '/'.join(Xpred_file.split('/')[:-1])
        print(save_dir)
        plotMSELossDistrib(Ypred_file, Ytruth_file, save_dir=save_dir)

if __name__ == '__main__':
    # The folder to work 
    folder = '/home/sr365/mm_bench_multi_eval/VAE/Chen'
    #folder = '/home/sr365/MM_Bench/MDN_temp/data'
    #folder = '/home/sr365/MM_Bench/MDN_temp/data_sigm_0.1'
    # Run simulator for the whole folder
    eval_whole_folder(folder)
    
    #plot_MSE(folder)

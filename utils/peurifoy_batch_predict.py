from utils.helper_functions import simulator
from multiprocessing import Pool
from utils.evaluation_helper import plotMSELossDistrib
import numpy as np
import os
import pandas as pd

# This is the script for doing batch evaluation
num_cpu = 10
def eval_peurifoy_for_file(filename):
    # Read the Xpred file
    Xpred = pd.read_csv(filename, sep=' ', header=None).values
    # Run the simulator
    Ypred = simulator('Peurifoy', Xpred)
    # Save the Ypred into the same folder with name change
    with open(filename.replace('Xpred','Ypred'), 'a') as fyp:
        np.savetxt(fyp, Ypred)

def eval_whole_folder(folder):
    """
    Run the eval peurifoy for file in a loop 
    """
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for filename in os.listdir(folder):
            if not ('Peurifoy' in filename and 'Xpred' in filename):
                continue
            args_list.append((os.path.join(folder, filename),))
        print((args_list))
        print(len(args_list))
        pool.starmap(eval_peurifoy_for_file, args_list)
    finally:
        pool.close()
        pool.join()

def plot_MSE(folder):
    """
    Plot the MSE plots for the simulated pairs of MSE
    """
    for file in os.listdir(folder):
        if not ('Peurifoy' in file and 'Xpred' in file):
                continue
        Xpred_file = os.path.join(folder, file)
        Ypred_file = Xpred_file.replace('Xpred','Ypred')
        Ytruth_file = Ypred_file.replace('Ypred','Ytruth')
        save_dir = '/'.join(Xpred_file.split('/')[:-1])
        print(save_dir)
        plotMSELossDistrib(Ypred_file, Ytruth_file, save_dir=save_dir)

if __name__ == '__main__':
    # Tandem and inverse only need to do the evaluation once
    #eval_whole_folder('../inverse/data')
    #plot_MSE('../inverse/data')
    #eval_whole_folder('../Tandem/data')
    #plot_MSE('../Tandem/data')
    #quit()
    # For the multi_T evaluation
    folder_list = ['VAE','Tandem','NN','NA','GA','INN','MDN','cINN']
    #folder_list = ['GA']
    # folder_list = ['MDN','INN','VAE','cINN']
    for folders in folder_list:
        folder = '../mm_bench_multi_eval/{}/Peurifoy'.format(folders)
        # Run simulator for the whole folder
        eval_whole_folder(folder)
        
        #plot_MSE(folder)

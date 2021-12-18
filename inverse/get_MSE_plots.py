# Get the MSE plot for the evaluated results from /data* folders

import os
from utils.evaluation_helper import plotMSELossDistrib


big_folder = '/home/sr365/MM_Bench/NA/'
#big_folder = '/home/sr365/MM_Bench/NA/'

for folder in os.listdir(big_folder):
    # ignore if not a folder or no data in folder name
    if 'data' not in folder or not os.path.isdir(os.path.join(big_folder, folder)) or 'Chen' not in folder:
        continue
    
    print('going into folder ', folder)
    # Loop over the data folder 
    for file in os.listdir(os.path.join(big_folder, folder)):
        if 'test_Ytruth' not in file:
            continue
        print('going in file', file)
        truth_file = os.path.join(big_folder, folder, file)
        pred_file = os.path.join(big_folder, folder, file.replace('Ytruth', 'Ypred'))
        # Make sure Ypred is also present
        if not os.path.isfile(pred_file):
            print('no Ypred file, abort!')
            continue
        
        print('doing MSE plot for file ', file, 'in folder ', os.path.join(big_folder, folder))
        plotMSELossDistrib(pred_file=pred_file, truth_file=truth_file, 
                            save_dir=os.path.join(big_folder, folder))

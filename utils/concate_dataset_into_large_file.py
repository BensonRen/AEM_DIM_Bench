# This function concatenate a bunch of files into a large, single file for dataset reading

import os
import numpy as np
import pandas as pd

mother_folder = '/home/sr365/MM_Bench/Data/Omar_bowtie'
geometry, spectra = None, None

for file_prefix in ['inputs_Srico_MMPA_bowtie_x1000_','Abs_Srico_MMPA_bowtie_x1000_']:
    value_all = None        # Init the value
    for i in range(1, 8): 
        # Combine the values
        file = os.path.join(mother_folder, file_prefix + '{}.csv'.format(i))
        value = pd.read_csv(file, sep=',', header=None).values
        #print(np.shape(value))
        if value_all is None:
            value_all = value
        else: 
            value_all = np.concatenate([value_all, value])      # Concate the value
        print(np.shape(value_all))
    
    # Save the values
    np.savetxt(os.path.join(mother_folder, file_prefix + 'all.csv'), value_all)
        
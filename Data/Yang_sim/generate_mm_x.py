# This function generates a random geometry x for meta-material dataset. 
# This is only the data_x.csv generator, after generating this, please go to NA/predict.py and run the create_mm_dataset() function to get the data_y.csv which is the spectra of the meta-material dataset. Pls be reminded that the neural simulator only has the accuracy of 6e-5 only at the given range above.
# Running this file again would help to create a new set of meta-material dataset, which would help make sure that the models you chose from the 10 trained ones are not biased towards the validataion set instead of the real test performance.
import numpy as np
import os

def generate_meta_material(data_num):
    x_dim = 14
    # Generate random number
    data_x = np.random.uniform(size=(data_num,x_dim), low=-1, high=1)
    print('data_x now has shape:', np.shape(data_x))
    return data_x

if __name__ == '__main__':
    ndata = 10000   # Training and validation set
    # ndata = 1000    # Test set (half would be taken)
    data_x = generate_meta_material(ndata)
    os.makedirs('dataIn')
    np.savetxt('dataIn/data_x.csv', data_x)


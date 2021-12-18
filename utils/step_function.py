# This function is to generate step function for the testing of H1 condition

import numpy as np
import os
import matplotlib.pyplot as plt

dataset_list = ['Yang','Peurifoy','Chen']
x_dim_dict = {'Yang': 14, 'Peurifoy': 3, 'Chen': 5 }
y_dim_dict = {'Yang': 2000, 'Peurifoy': 201, 'Chen': 256 }


def get_step_function(dataset, orientation='s', amplitude=1, step_pos=0.5):
    """
    The main function for outputting a step function given arguments
    :param dataset: The dataset to do the step function, which controls the length of y vector
    :param orientation: The orientation of the step function, either s or z 
    :param amplitude: The high point of the step function
    :param step_pos: The position of the transition, this is a number between 0 to 1
    """
    y = np.zeros([y_dim_dict[dataset],])
    if orientation == 's':
        y[int(len(y)*step_pos):] = amplitude
    elif orientation == 'z':
        y[:int(len(y)*step_pos)] = amplitude
    else:
        print('orientation can only be z or s, aborting!')
        quit()
    return y


def generate_step_functions():
    """
    The funciton for calling the step function in loops and store them in a file
    """
    for dataset in dataset_list:
        y_tot = []
        for orientation in ['s','z']:
            for amplitude in [0.1*j for j in range(1, 11)]:
                for step_pos in [0.1*k for k in range(1, 10)]:
                    y = get_step_function(dataset=dataset, orientation=orientation, amplitude=amplitude, step_pos=step_pos)
                    y_tot.append(y)
        # print('len of y_tot = ', len(y_tot))
        y_tot = np.array(y_tot)
        print('shape of np version = ', np.shape(y_tot))
        save_dir = '/home/sr365/MM_Bench/Data/step_func/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.savetxt(os.path.join(save_dir ,dataset+'step_function.txt'), y_tot)


if __name__ == '__main__':
    generate_step_functions()
    """
    f = plt.figure()
    y = get_step_function('Yang', 's',0.7, 0.2)
    plt.plot(y)
    plt.xlabel('frequency')
    plt.ylabel('R')
    plt.title('step function')
    plt.savefig('step_funciton.png')
    """
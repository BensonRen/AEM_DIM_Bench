"""
This is the helper functions for various functions
1-4: retrieving the prediction or truth files in data/
5: Put flags.obj and parameters.txt into the folder
6-8: Functions handling flags
9-12: Simulator functions
13: Meta-simulator function
14: Normalize at eval mode (get back the original range)
"""
import os
import shutil
from copy import deepcopy
import sys
import pickle
import numpy as np
from Data.Peurifoy.generate_Peurifoy import simulate as peur_sim
from Data.Chen.generate_chen import simulate as chen_sim
# from ensemble_mm.predict_ensemble import ensemble_predict_master
# 1
def get_Xpred(path, name=None):
    """
    Get certain predicion or truth numpy array from path, with name of model specified.
    If there is no name specified, return the first found such array
    :param path: str, the path for which to search
    :param name: str, the name of the model to find
    :return: np array
    """
    out_file = None
    if name is not None:
        name = name.replace('/','_')
    for filename in os.listdir(path):
        if ("Xpred" in filename):
            if name is None:
                out_file = filename
                print("Xpred File found", filename)
                break
            else:
                if (name in filename):
                    out_file = filename
                    break
    assert out_file is not None, "Your Xpred model did not found" + name
    return np.loadtxt(os.path.join(path,out_file))


# 2
def get_Ypred(path, name=None):
    """
    Get certain predicion or truth numpy array from path, with name of model specified.
    If there is no name specified, return the first found such array
    :param path: str, the path for which to search
    :param name: str, the name of the model to find
    :return: np array
    """
    out_file = None
    name = name.replace('/','_')
    for filename in os.listdir(path):
        if ("Ypred" in filename):
            if name is None:
                out_file = filename
                print("Ypred File found", filename)
                break
            else:
                if (name in filename):
                    out_file = filename
                    break
    assert out_file is not None, "Your Xpred model did not found" + name
    return np.loadtxt(os.path.join(path,out_file))


# 3
def get_Xtruth(path, name=None):
    """
    Get certain predicion or truth numpy array from path, with name of model specified.
    If there is no name specified, return the first found such array
    :param path: str, the path for which to search
    :param name: str, the name of the model to find
    :return: np array
    """
    out_file = None
    if name is not None:
        name = name.replace('/','_')
    for filename in os.listdir(path):
        if ("Xtruth" in filename):
            if name is None:
                out_file = filename
                print("Xtruth File found", filename)
                break
            else:
                if (name in filename):
                    out_file = filename
                    break
    assert out_file is not None, "Your Xpred model did not found" + name
    return np.loadtxt(os.path.join(path,out_file))

# 4
def get_Ytruth(path, name=None):
    """
    Get certain predicion or truth numpy array from path, with name of model specified.
    If there is no name specified, return the first found such array
    :param path: str, the path for which to search
    :param name: str, the name of the model to find
    :return: np array
    """
    out_file = None
    name = name.replace('/','_')
    for filename in os.listdir(path):
        if ("Ytruth" in filename):
            if name is None:
                out_file = filename
                print("Ytruth File found", filename)
                break
            else:
                if (name in filename):
                    out_file = filename
                    break
    assert out_file is not None, "Your Xpred model did not found" + name
    return np.loadtxt(os.path.join(path,out_file))


# 5
def put_param_into_folder(ckpt_dir):
    """
    Put the parameter.txt into the folder and the flags.obj as well
    :return: None
    """
    """
    Old version of finding the latest changing file, deprecated
    # list_of_files = glob.glob('models/*')                           # Use glob to list the dirs in models/
    # latest_file = max(list_of_files, key=os.path.getctime)          # Find the latest file (just trained)
    # print("The parameter.txt is put into folder " + latest_file)    # Print to confirm the filename
    """
    # Move the parameters.txt
    destination = os.path.join(ckpt_dir, "parameters.txt");
    shutil.move("parameters.txt", destination)
    # Move the flags.obj
    destination = os.path.join(ckpt_dir, "flags.obj");
    shutil.move("flags.obj", destination)


# 6
def save_flags(flags, save_dir, save_file="flags.obj"):
    """
    This function serialize the flag object and save it for further retrieval during inference time
    :param flags: The flags object to save
    :param save_file: The place to save the file
    :return: None
    """
    with open(os.path.join(save_dir, save_file),'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object


# 7
def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags


# 8
def write_flags_and_BVE(flags, best_validation_loss, save_dir, forward_best_loss=None):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :param forard_best_loss: The forward best loss only applicable for Tandem model
    :return: None
    """
    flags.best_validation_loss = best_validation_loss  # Change the y range to be acceptable long string
    if forward_best_loss is not None:
        flags.best_forward_validation_loss = forward_best_loss
    # To avoid terrible looking shape of y_range
    yrange = flags.y_range
    # yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    yrange_str = [yrange[0], yrange[-1]]
    copy_flags = deepcopy(flags)
    copy_flags.y_range = yrange_str  # in order to not corrupt the original data strucutre
    flags_dict = vars(copy_flags)
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        print(flags_dict, file=f)
    # Pickle the obj
    save_flags(flags, save_dir=save_dir)



    
# 15
def simulator(data_set, Xpred):
    """
    This is the simulator which takes Xpred from inference models and feed them into real data
    simulator to get Ypred
    :param data_set: str, the name of the data set
    :param Xpred: (N, dim_x), the numpy array of the Xpred got from the inference model
    :return: Ypred from the simulator
    """

    if data_set == 'Peurifoy':
        # The geometric boundary of peurifoy dataset is [30, 70], normalizing manually
        Xpred = Xpred * 20. + 50
        Ypred = []
        for i in range(len(Xpred)):
            spec = peur_sim(Xpred[i, :])
            Ypred.append(spec)
        return np.array(Ypred)
    elif data_set == 'Chen':
        # The geometric boundary of Chen dataset is [5, 50], unnormalizing manually
        Xpred = Xpred*22.5+27.5
        return chen_sim(Xpred)
    elif data_set == 'Yang':
        sys.exit("You are using Yang dataset, there is no simulator built-in for Yang! Please use neural simulator")
    else:
        sys.exit("In Simulator: Your data_set entry is not correct, check again!")


# 16
def normalize_eval(x, x_max, x_min):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
    return x


# 17
def unnormalize_eval(x, x_max, x_min):
    """
    UnNormalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = x[:, i] * x_range + x_avg
    return x

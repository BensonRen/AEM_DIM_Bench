"""
This file serves to hold helper functions that is related to the "Flag" object which contains
all the parameters during training and inference
"""
# Built-in
import argparse
# Libs

# Own module
from parameters import *

# Torch

def read_flag():
    """
    This function is to write the read the flags from a parameter file and put them in formats
    :return: flags: a struct where all the input params are stored
    """
    parser = argparse.ArgumentParser()
    # Data_Set parameter
    parser.add_argument('--data-set', default=DATA_SET, type=str, help='which data set you are chosing')
    parser.add_argument('--test-ratio', default=TEST_RATIO, type=float, help='the ratio of the test set')

    # Model Architectural Params
    parser.add_argument('--num-gaussian', type=int, default=NUM_GAUSSIAN, help='The number of gaussians')
    parser.add_argument('--linear', type=list, default=LINEAR, help='The fc layers units')

    # Optimizer Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-batch-size', default=EVAL_BATCH_SIZE, type=int,
                        help='The Batch size for back propagation')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    #    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
    #                        help='decay learning rate at this number of steps')

    # Data Specific Params
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool,
                        help='whether we should normalize the input or not')

    # Running specific
    parser.add_argument('--eval-model', default=EVAL_MODEL, type=str, help='the folder name of the model that you want to evaluate')
    parser.add_argument('--use-cpu-only', type=bool, default=USE_CPU_ONLY, help='The boolean flag that indicate use CPU only')
    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags

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

    # GA SPECIFIC ARGUMENTS
    # Eseential Algorithmic Params
    parser.add_argument('--population', type=int, default=POP_SIZE, help='The size of populations in each generation')
    parser.add_argument('--elitism', type=int, default=ELITISM, help='# of highest fitness models passed to next generation')
    parser.add_argument('--mutation', type=float, default=MUTATION,
                        help='Probability of a point mutation occuring in an individual')
    parser.add_argument('--crossover', type=float, default=CROSSOVER,
                        help='Probability of crossover occuring b/w parents to make children')
    parser.add_argument('--k', type=int, default=K, help='If selection operator is decimation - identifies fitness threshold '
                                                         'for individuals to become parents. If selection operator is '
                                                         'tournament - identifies number of individuals in each tournament ')

    # Categorical Algorithmic Params
    parser.add_argument('--selection-operator', type=str, default=SELECT_OPS,
                        help="Selection operator: 'roulette', 'decimation', 'tournament'")
    parser.add_argument('--cross-operator', type=str, default=CROSS_OPS, help="Crossover Operator: 'single-point', 'uniform'")
    parser.add_argument('--ga-eval', type=bool, default=GA_EVAL, help='Evaluation of GA')

    # Optimization Params
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--generations', default=GENERATIONS, type=int, help='# steps for back propagation')

    # Data Specific Params
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool,
                        help='whether we should normalize the input or not')

    # Network Specific Parameters
    # Running specific
    parser.add_argument('--eval-model', default=EVAL_MODEL, type=str, help='the folder name of the model that you want to evaluate')
    parser.add_argument('--use-cpu-only', type=bool, default=USE_CPU_ONLY, help='The boolean flag that indicate use CPU only')

    # Model Architectural Params
    parser.add_argument('--use-lorentz', type=bool, default=USE_LORENTZ, help='The boolean flag that indicate whether we use lorentz oscillator')
    parser.add_argument('--linear', type=list, default=LINEAR, help='The fc layers units')
    parser.add_argument('--conv-out-channel', type=list, default=CONV_OUT_CHANNEL, help='The output channel of your 1d conv')
    parser.add_argument('--conv-kernel-size', type=list, default=CONV_KERNEL_SIZE, help='The kernel size of your 1d conv')
    parser.add_argument('--conv-stride', type=list, default=CONV_STRIDE, help='The strides of your 1d conv')

    # Optimizer Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--lr',type=float, default=LEARN_RATE, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')

    flags = parser.parse_args()
    return flags


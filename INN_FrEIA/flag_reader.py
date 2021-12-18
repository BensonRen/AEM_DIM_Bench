"""
This file serves to hold helper functions that is related to the "Flag" object which contains
all the parameters during training and inference
"""
# Built-in
import argparse
import pickle
import os
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
    # VAE model architecture hyper parameters
    parser.add_argument('--dim-z', default=DIM_Z, type=int, help='dimension of the latent variable z')
    parser.add_argument('--dim-x', default=DIM_X, type=int, help='dimension of the latent variable x')
    parser.add_argument('--dim-y', default=DIM_Y, type=int, help='dimension of the latent variable y')
    parser.add_argument('--dim-tot', default=DIM_TOT, type=int, help='dimension of the total dimension of both sides')
    parser.add_argument('--dim-spec', default=DIM_SPEC, type=int, help='dimension of the spectra encoded conponent')
    parser.add_argument('--couple-layer-num', default=COUPLE_LAYER_NUM, type=int, help='The number of coupling blocks to use')
    parser.add_argument('--subnet-linear', type=list, default=SUBNET_LINEAR, help='The fc layers units for subnetwork')
    parser.add_argument('--linear-se', type=list, default=LINEAR_SE, help='The fc layers units for spectra encoder model')
    parser.add_argument('--conv-out-channel-se', type=list, default=CONV_OUT_CHANNEL_SE, help='The output channel of your 1d conv for spectra encoder model')
    parser.add_argument('--conv-kernel-size-se', type=list, default=CONV_KERNEL_SIZE_SE, help='The kernel size of your 1d conv for spectra encoder model')
    parser.add_argument('--conv-stride-se', type=list, default=CONV_STRIDE_SE, help='The strides of your 1d conv fro spectra encoder model')
    # Loss ratio
    parser.add_argument('--lambda-mse', type=float, default=LAMBDA_MSE, help='the coefficient for mse loss lambda')
    parser.add_argument('--lambda-z', type=float, default=LAMBDA_Z, help='the coefficient for latent variable MMD loss lambda')
    parser.add_argument('--lambda-rev', type=float, default=LAMBDA_REV, help='the coefficient for reverse MMD loss lambda')
    parser.add_argument('--zeros-noise-scale', type=float, default=ZEROS_NOISE_SCALE, help='the noise scale of random variables for zeros')
    parser.add_argument('--y-noise-scale', type=float, default=Y_NOISE_SCALE, help='the noise scale on y')
    # Optimization Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--grad-clamp', default=GRAD_CLAMP, type=float, help='gradient is clamped within [-15, 15]')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-batch-size', default=EVAL_BATCH_SIZE, type=int, help='The Batch size for back propagation')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--verb-step', default=VERB_STEP, type=int, help='# steps to print and check best performance')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    # Data Specific params
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--ckpt-dir', default=CKPT_DIR, type=str, help='ckpt_dir')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool,
                        help='whether we should normalize the input or not')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    # Running specific params
    parser.add_argument('--eval-model', default=EVAL_MODEL, type=str, help='the folder name of the model that you want to evaluate')
    parser.add_argument('--use-cpu-only', type=bool, default=USE_CPU_ONLY, help='The boolean flag that indicate use CPU only')
    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags


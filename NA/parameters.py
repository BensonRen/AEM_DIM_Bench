"""
Params for Back propagation model
"""
# Define which data set you are using
#DATA_SET = 'Yang_sim'
from math import inf


DATA_SET = 'Chen'
#DATA_SET = 'Peurifoy'
#DATA_SET = 'Omar'
TEST_RATIO = 0.2

# Model Architectural Params for meta_material data Set
USE_LORENTZ = False
LINEAR = [5, 500, 500, 256]
#LINEAR = [10, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 1001]
CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []


# MD loss related params
MD_COEFF = 5e-5
MD_RADIUS = 0.2
MD_START = -inf
MD_END = inf

# Model Architectural Params for Yang dataset
#LINEAR = [4, 500, 500, 500, 500, 1]                 # Dimension of data set cross check with data generator
#CONV_OUT_CHANNEL = [4, 4, 4]
#CONV_KERNEL_SIZE = [3, 3, 5]
#CONV_STRIDE = [2, 1, 1]


# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 2048
EVAL_STEP = 20
TRAIN_STEP = 300
BACKPROP_STEP = 300
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.8
STOP_THRESHOLD = 1e-9

# Data specific Params
X_RANGE = [i for i in range(2, 16 )]
#Y_RANGE = [i for i in range(10 , 2011 )]                       # Real Meta-material dataset range
Y_RANGE = [i for i in range(16 , 2017 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '../Data'                                               # All simulated simple dataset
#DATA_DIR = '/work/sr365/'                                      # real Meta-material dataset
#DATA_DIR = '/work/sr365/NN_based_MM_data/'                      # Artificial Meta-material dataset
#DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

# Running specific
USE_CPU_ONLY = False
#EVAL_MODEL = "sine_wavereg2e-05trail_0_forward_swipe9"
EVAL_MODEL = "mm"
#EVAL_MODEL = "robotic_armreg0.0005trail_0_backward_complexity_swipe_layer500_num6"
#EVAL_MODEL = "ballisticsreg0.0005trail_0_complexity_swipe_layer500_num5"
#EVAL_MODEL = "meta_materialreg2e-05trail_0_forward_swipe6"
#EVAL_MODEL = "20200506_104444"

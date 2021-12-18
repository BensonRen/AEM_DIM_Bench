"""
The parameter file storing the parameters for INN Model
"""

# Define which data set you are using
#DATA_SET = 'Yang_sim'
#DATA_SET = 'Chen'
DATA_SET = 'Peurifoy'

TEST_RATIO = 0.2

# Architectural Params
"""
# Chen
DIM_Z = 5
DIM_X = 5
DIM_Y = 256
"""
# Peurifoy
DIM_Z = 8
DIM_X = 8
DIM_Y = 201
"""
# Yang
DIM_Z = 14
DIM_X = 14
DIM_Y = 2000
"""

COUPLE_LAYER_NUM = 13
DIM_SPEC = None
# The below definitions are useless now since we are using the package
LINEAR_SE = []                      # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []

# This set of parameters are used for dataset meta-material
#LINEAR_SE = [150, 500, 500, 500, 500, DIM_Y]     # Linear units for spectra encoder
#CONV_OUT_CHANNEL_SE = [4, 4, 4]
#CONV_KERNEL_SIZE_SE = [5, 5, 8]
#CONV_STRIDE_SE = [1, 1, 2]

# Optimization params
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 2048
EVAL_STEP = 50
GRAD_CLAMP = 15
TRAIN_STEP = 500
VERB_STEP = 50
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.8
STOP_THRESHOLD = -float('inf')

X_RANGE = [i for i in range(2, 16 )]
Y_RANGE = [i for i in range(16 , 2017 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '/home/sr365/MM_Bench/Data/'                                               # All simulated simple dataset
#DATA_DIR = '../Data/Yang_data/'                                               # All simulated simple dataset
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
EVAL_MODEL = 'fake_2k'

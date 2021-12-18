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
DIM_X = 5
DIM_Y = 256
DIM_Z = 1 
DIM_TOT = 300
"""
# Peurifoy
DIM_X = 8
DIM_Y = 201
DIM_Z = 1 
DIM_TOT = 9
"""
# Yang
DIM_X = 14
DIM_Y = 2000
DIM_Z = 1 
DIM_TOT = 4
"""

# Architectural Params
COUPLE_LAYER_NUM = 6
DIM_SPEC = None
SUBNET_LINEAR = []                                          # Linear units for Subnet FC layer
#LINEAR_SE = [150, 150, 150, 150, DIM_Y]                                              # Linear units for spectra encoder
LINEAR_SE = []                                              # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []
#CONV_OUT_CHANNEL_SE = [4, 4, 4]
#CONV_KERNEL_SIZE_SE = [5, 5, 8]
#CONV_STRIDE_SE = [1, 1, 2]

# Loss ratio
LAMBDA_MSE = 0.1             # The Loss factor of the MSE loss (reconstruction loss)
LAMBDA_Z = 300.             # The Loss factor of the latent dimension (converging to normal distribution)
LAMBDA_REV = 400.           # The Loss factor of the reverse transformation (let x converge to input distribution)
ZEROS_NOISE_SCALE = 5e-2          # The noise scale to add to
Y_NOISE_SCALE = 1e-2


# Optimization params
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
GRAD_CLAMP = 15
TRAIN_STEP = 500
VERB_STEP = 50
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.9
STOP_THRESHOLD = -float('inf')
CKPT_DIR = 'models/'

# Data specific params
X_RANGE = [i for i in range(2, 10 )]
#Y_RANGE = [i for i in range(10 , 2011 )]                       # Real Meta-material dataset range
Y_RANGE = [i for i in range(10 , 310 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '/home/sr365/MM_Bench/Data/'                                               # All simulated simple dataset
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
EVAL_MODEL = None

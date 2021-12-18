"""
The parameter file storing the parameters for VAE Model
"""

# Define which data set you are using
DATA_SET = 'Yang_sim'
#DATA_SET = 'Chen'
#DATA_SET = 'Peurifoy'

TEST_RATIO = 0.2

# Architectural Params
"""
# Chen
DIM_Z = 5
DIM_X = 5
DIM_Y = 256
# Peurifoy
DIM_Z = 8
DIM_X = 8
DIM_Y = 201
"""
# Yang
DIM_Z = 14
DIM_X = 14
DIM_Y = 2000
DIM_SPEC = None
LINEAR_D = [DIM_Y + DIM_Z, 500, 500, 500, 500, 500, 500, 500,    DIM_X]           # Linear units for Decoder
LINEAR_E = [DIM_Y + DIM_X, 500, 500, 500, 500, 500, 500, 500, 2*DIM_Z]                   # Linear units for Encoder
LINEAR_SE = []                      # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []
#LINEAR_SE = [150, 500, 500, 500, 500, DIM_Y]                      # Linear units for spectra encoder
#CONV_OUT_CHANNEL_SE = [4, 4, 4]
#CONV_KERNEL_SIZE_SE = [5, 5, 8]
#CONV_STRIDE_SE = [1, 1, 2]

# Optimization params
KL_COEFF = 0.005
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
TRAIN_STEP = 300
VERB_STEP = 30
LEARN_RATE = 1e-4
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
EVAL_MODEL = None

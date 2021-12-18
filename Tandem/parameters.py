"""
Hyper-parameters of the Tandem model
"""
# Define which data set you are using
#DATA_SET = 'Yang_sim'
#DATA_SET = 'Chen'
DATA_SET = 'Peurifoy'

TEST_RATIO = 0.2

# Model Architecture parameters
LOAD_FORWARD_CKPT_DIR = None
"""
#LINEAR_F = [4, 500, 500, 500, 1]
LINEAR_F = [8, 1000, 1000, 1000, 1000, 150]
CONV_OUT_CHANNEL_F = [4, 4, 4]
CONV_KERNEL_SIZE_F = [8, 5, 5]
CONV_STRIDE_F = [2, 1, 1]

LINEAR_B = [150, 1000, 1000, 1000, 1000, 1000, 8]
CONV_OUT_CHANNEL_B = [4, 4, 4]
CONV_KERNEL_SIZE_B = [5, 5, 8]
CONV_STRIDE_B = [1, 1, 2]

"""
# Model Architectural Params for gaussian mixture dataset
LINEAR_F = [8, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 201]
#LINEAR_F = [3, 500, 500, 500, 500, 500,  256]
CONV_OUT_CHANNEL_F = []
CONV_KERNEL_SIZE_F = []
CONV_STRIDE_F = []
#CONV_OUT_CHANNEL_F = [4,4,4]
#CONV_KERNEL_SIZE_F = [4,3,3]
#CONV_STRIDE_F = [2,1,1]

#LINEAR_B = [2, 500, 500, 500, 500, 500, 3]
LINEAR_B = [201, 500, 500, 500, 500, 500, 3]
CONV_OUT_CHANNEL_B = []
CONV_KERNEL_SIZE_B = []
CONV_STRIDE_B = []


# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 5e-4 
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024
EVAL_STEP = 50
TRAIN_STEP = 300
VERB_STEP = 20
LEARN_RATE = 1e-4
LR_DECAY_RATE = 0.8
STOP_THRESHOLD = -1 # -1 means dont stop

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True

# Data-specific parameters# Data specific Params
X_RANGE = [i for i in range(2, 16 )]
Y_RANGE = [i for i in range(16 , 2017 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '/home/sr365/MM_Bench/Data/'                                               # All simulated simple dataset
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

EVAL_MODEL = None

"""
Params for Back propagation model
"""
# Define which data set you are using
DATA_SET = 'Yang_sim'
#DATA_SET = 'Chen'
#DATA_SET = 'Peurifoy'
TEST_RATIO = 0.2

# Model Architectural Params for meta_material data Set
NUM_GAUSSIAN = 10
LINEAR = [2,  1000, 1000, 1000, 1000, 1000, 1000, 1000, 4]

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 5
TRAIN_STEP = 500
LEARN_RATE = 5e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.8
STOP_THRESHOLD = -float('inf')

# Data specific Params
X_RANGE = [i for i in range(2, 16 )]
Y_RANGE = [i for i in range(16 , 2017 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME = None 
DATA_DIR = '/home/sr365/MM_Bench/Data/'                                               # All simulated simple dataset
#DATA_DIR = '../Data/Yang_data/'                                               # All simulated simple dataset
GEOBOUNDARY =[0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.786, 0.786]
NORMALIZE_INPUT = True

# Running specific
USE_CPU_ONLY = False
EVAL_MODEL = None 

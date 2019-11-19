import numpy as np
from math import sin, cos, pi
#import tensorflow as tf
####################################################################################
##################################### CONFIG #######################################
class Config(object):
    ###============ player params =========
    CAP_RANGE = 2.
    VD = 1.
    VI_FAST = 1.5
    VI_SLOW = .8
    TAG_RANGE = 5.
    
    TIME_STEP = 0.1
    
    ##========= target =========

    XI0 = [np.array([0, 5.])]
    XD0 = [np.array([-3., 3.]), np.array([3., 3.])]

    ###============ learning params =========
    LEARNING_RATE = 0.01
    LAYER_SIZES = [30, 6, 30]
    # ACT_FUNCS = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh]
    TAU = 0.01
    MAX_BUFFER_SIZE = 10000
    BATCH_SIZE = 1000
    TRAIN_STEPS = 100
    TARGET_UPDATE_INTERVAL = 1

    ###============ saving params =========
    # DATA_FILE = 'BarrierData.csv'
    # MODEL_DIR = 'models/'
    BARRIER_DIR = 'BarrierFn'
    POLICY_DIR = 'PolicyFn'

    SAVE_FREQUENCY = 100
    PRINTING_FREQUENCY = 50

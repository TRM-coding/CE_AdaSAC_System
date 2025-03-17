from argparse import Namespace
import torch


CONFIG = Namespace(
    GPU_AVAILABLE                 = (0,5),
    UNAVAILABLE                   = [0],
    DEFAULT_DEVICE                = "cuda:7",
    CUT_STEP                      =0.2,
    MODEL_LAYER_NUMBER            =50,
    ALPHA                         =0.5,
    QUANTISIZED_TYPE              =torch.qint8,

    TEST_DATA_TOTAL_NUMBER        = 100,
    TEST_DATA_BATCH_SIZE          = 100,
    TEST_DATA_LEARNING_RATE       = 1,
    TEST_DATA_WARM_LR             = 1e-3,
    TEST_DATA_CHANNEL             = 3,
    TEST_DATA_DIM1                = 224,
    TEST_DATA_DIM2                = 224,
    TEST_DATA_OUTPUT_SIZE         = 1000,
    TEST_DATA_RANDN_MAGNIFICATION = 100,
    TEST_DATA_CONFIDENCE          = 1000000,
    TEST_DATA_TARGET_ACC          = 0.8,

    LOCAL_SPEED                   =2.72e10,
    CLOUD_SPEED                   =1.7e13,
    NETWORK_SPEED                 =1e7,
    ACC_CUT_POINT                 =0.7,

)
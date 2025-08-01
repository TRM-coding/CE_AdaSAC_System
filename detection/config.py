from argparse import Namespace
import torch


CONFIG = Namespace(
    GPU_AVAILABLE                 = (0,7),
    UNAVAILABLE                   = [0,1],
    DEFAULT_DEVICE                = "cuda:3",
    CUT_STEP                      =0.1,
    MODEL_LAYER_NUMBER            =50,
    ALPHA                         =0.9,# alpha ->time
    QUANTISIZED_TYPE              =torch.qint8,

    TEST_DATA_TOTAL_NUMBER        = 500,#how many batchs
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

    LOCAL_SPEED                   =5e11,
    CLOUD_SPEED                   =1.7e13,
    NETWORK_SPEED                 =1e7,#B/s
    ACC_CUT_POINT                 =0.7,

    GENERATE_EPOCH_WARM=15,
    INIT_SIZE_WARM=300,
    WORKERNUMBER=6,
    ASTOEPOCH=30,
    ALPHASTEP=0.1,

    SAVE_PATH_SCHEME="./asto_res50.txt",
    LOAD_NUMBER=64,
    CLOUD_PORT=5000,
    CLOUDIP='10.126.59.25',
    TIMEOUT=60,
    EVAL_REDUCE_NUMBER=[10,20,30,40,],
    PRESSURE_REDUCE=False,

    IMAGENET_PATH="/SSD/val"
)

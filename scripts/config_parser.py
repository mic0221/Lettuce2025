#!/usr/bin/env python3
"""
Parses "_config_dict.json" to extract all settings as variables.

Extraction is done in such a way that variables are assigned
the settings as their correct object types, using eval().
E.g. the string "torch.optim.Adam" will be the PyTorch function torch.optim.Adam.

Adjust _config_dict = utils.parse_json("___") based on
fine-tune_resize.py: config_resize.json
fine-tune_patch.py: config_patch.json

Adapted from Jacky To
"""
# Import statements
import albumentations as A
import torch
import segmentation_models_pytorch as smp
import cv2
from albumentations.pytorch import ToTensorV2


# Import supporting modules
import utils


# Parse config.json as dict
_config_dict = utils.parse_json("config_patch.json")

# Assign settings to variables
# Seed
SEED = _config_dict["SEED"]

# Setup device settings
DEVICE = _config_dict["DEVICE"]
MULTI_GPU = eval(_config_dict["MULTI_GPU"])
NUM_WORKERS = _config_dict["NUM_WORKERS"]
PIN_MEMORY = eval(_config_dict["PIN_MEMORY"])

# Setup hyperparameters and other training specifics
LEARNING_RATE = _config_dict["LEARNING_RATE"]
NUM_EPOCHS = _config_dict["NUM_EPOCHS"]
OPTIMIZER = eval(_config_dict["OPTIMIZER"])
SCALER = eval(_config_dict["SCALER"])
LOSS_FN = eval(_config_dict["LOSS_FN"])
PERFORMANCE_FN = [eval(_fn) for _fn in _config_dict["PERFORMANCE_FN"]]

# Setup data loading settings
DATASET = eval(_config_dict["DATA_CLASS"])
IMG_DIR = _config_dict["IMG_DIR"]
LABEL_DIR = _config_dict["LABEL_DIR"]
if LABEL_DIR == "None":
    LABEL_DIR = eval(LABEL_DIR)  # In case of no label for e.g. classifier

TRAIN_FRAC = _config_dict["TRAIN_FRAC"]
KFOLD = _config_dict["KFOLD"]
if KFOLD == "None":
    KFOLD = eval(KFOLD)  # In case of no K-fold cross validation
TRAIN_TRANSFORMS = A.Compose(
    [eval(_tf) for _tf in _config_dict["TRAIN_TRANSFORMS"]], is_check_shapes=False
)
TEST_TRANSFORMS = A.Compose(
    [eval(_tf) for _tf in _config_dict["TEST_TRANSFORMS"]], is_check_shapes=False
)
BATCH_SIZE = _config_dict["BATCH_SIZE"]

# Setup model settings
MODEL_TYPE = eval(_config_dict["MODEL_TYPE"])
MODEL_NAME = _config_dict["MODEL_NAME"]
if MODEL_NAME == "None":
    MODEL_NAME = eval(MODEL_NAME)  # In case of no model name (classifier)
ENCODER_NAME = _config_dict["ENCODER_NAME"]
ENCODER_WEIGHTS = _config_dict["ENCODER_WEIGHTS"]
if ENCODER_WEIGHTS == "None":  # Allow for untrained encoder
    ENCODER_WEIGHTS = eval(ENCODER_WEIGHTS)
N_CHANNELS = _config_dict["N_CHANNELS"]
N_CLASSES = _config_dict["N_CLASSES"]
DECODER_ATTENTION = _config_dict["DECODER_ATTENTION"]
if DECODER_ATTENTION == "None":  # Allow for no decoder attention
    DECODER_ATTENTION = eval(DECODER_ATTENTION)
ENCODER_FREEZE = eval(_config_dict["ENCODER_FREEZE"])

# Setup checkpointing, save and load
CHECKPOINT_FREQ = _config_dict["CHECKPOINT_FREQ"]
if CHECKPOINT_FREQ == "None":  # Allow for no checkpointing
    CHECKPOINT_FREQ = eval(CHECKPOINT_FREQ)
SAVE_MODEL_DIR = _config_dict["SAVE_MODEL_DIR"]
SAVE_MODEL_NAME = _config_dict["SAVE_MODEL_NAME"]
LOAD_MODEL = eval(_config_dict["LOAD_MODEL"])
LOAD_MODEL_PATH = _config_dict["LOAD_MODEL_PATH"]

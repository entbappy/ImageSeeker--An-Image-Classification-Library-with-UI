import os
import sys
import warnings


from utils import models_config as mc
import pandas as pd

"""Public ImageSeeker utilities.
This module is used as a shortcut to access all the symbols. Those symbols was
exposed under train engine and predict engine.
"""
df = pd.read_csv('input_data.csv')

#Data config
TRAIN_DATA_DIR = df['TRAIN_DATA_DIR'][0]
print(TRAIN_DATA_DIR)
VALID_DATA_DIR = df['VALID_DATA_DIR'][0]
CLASSES = df['CLASSES'][0]
SIZE = df['IMAGE_SIZE'].values[0].split(',')
h = int(SIZE[0])
w = int(SIZE[1])
c = int(SIZE[2])
IMAGE_SIZE = h,w,c
AUGMENTATION = df['AUGMENTATION'][0]
BATCH_SIZE = df['BATCH_SIZE'][0]

# Model config
MODEL_OBJ = df['MODEL_OBJ'][0]
print("I am Model obj", MODEL_OBJ)
MODEL_OBJ = mc.return_model(MODEL_OBJ)
MODEL_NAME = df['MODEL_NAME'][0]
EPOCHS = df['EPOCHS'][0]
OPTIMIZER = df['OPTIMIZER'][0]
LOSS_FUNC = df['LOSS_FUNC'][0]
FREEZE_ALL = df['FREEZE_ALL'][0]



def configureData(TRAIN_DATA_DIR = TRAIN_DATA_DIR, VALID_DATA_DIR = VALID_DATA_DIR, AUGMENTATION = AUGMENTATION, CLASSES = CLASSES, IMAGE_SIZE = IMAGE_SIZE, BATCH_SIZE = BATCH_SIZE):
    CONFIG = {
        'TRAIN_DATA_DIR' : TRAIN_DATA_DIR,
        'VALID_DATA_DIR' : VALID_DATA_DIR,
        'AUGMENTATION': AUGMENTATION,
        'CLASSES' : CLASSES,
        'IMAGE_SIZE' : IMAGE_SIZE,
        'BATCH_SIZE' : BATCH_SIZE,
    }

    return CONFIG





def configureModel(MODEL_OBJ = MODEL_OBJ, MODEL_NAME=MODEL_NAME, EPOCHS = EPOCHS, FREEZE_ALL= FREEZE_ALL , OPTIMIZER=OPTIMIZER, LOSS_FUNC=LOSS_FUNC):
    CONFIG = {
        'MODEL_OBJ' : MODEL_OBJ,
        'MODEL_NAME' : MODEL_NAME,
        'EPOCHS' : EPOCHS,
        'FREEZE_ALL' : FREEZE_ALL,
        'OPTIMIZER': OPTIMIZER,
        'LOSS_FUNC' : LOSS_FUNC,
    }

    return CONFIG
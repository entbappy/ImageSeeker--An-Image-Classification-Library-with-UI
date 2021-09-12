'''
@author: Bappy Ahmed
Email: entbappy73@gmail.com
Date: 06-sep-2021
'''

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16

# Configure your data

TRAIN_DATA_DIR = "H:\\Parsonal\\Coding Practice\\iNeuron\\Moduler Coding\\ImageSeeker\\data\\train"        # Your training data path
VALID_DATA_DIR = "H:\\Parsonal\\Coding Practice\\iNeuron\\Moduler Coding\\ImageSeeker\\data\\valid"       # Your validation data path
CLASSES = 2                                                                                               # Number of classes in your data
IMAGE_SIZE = (224,224,3)                                                                                  #Image resulution/dimention with respect to your classification models
AUGMENTATION = True                                                                                       # If you want to apply Augmentation in your data (Default is True)
BATCH_SIZE = 32                                                                                           # Number of batch  (Default is 32)
PREDICTION_DATA_DIR = 'H:\\Parsonal\\\Coding Practice\\iNeuron\\Moduler Coding\\ImageSeeker\\prediction'  # Your prediction/test data path

##################################################################################################
#----------------------------------- Configure Your Data & Model ---------------------------------
##################################################################################################

# Configure your model

MODEL_OBJ = ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3))                         # Your pretrain model object
# PRETRAIN_MODEL_DIR = "H:\Parsonal\Coding Practice\iNeuron\Moduler Coding\ImageSeeker\Models\VGG16.h5"    #If you have any pretrain model exist path (Default is None)
PRETRAIN_MODEL_DIR = None
MODEL_NAME ='ResNet50'                                                                                     # Your model name
EPOCHS = 2                                                                                               # Number of Epochs
OPTIMIZER = 'adam'                                                                                         # Optimizers name/object
LOSS_FUNC = 'categorical_crossentropy'                                                                     # Your loss function name/object
FREEZE_ALL= True                                                                                           # Model layers freezing (Default is True)
FREEZE_TILL=None                                                                                          # You can define number of freezing layers (Defualt is None)

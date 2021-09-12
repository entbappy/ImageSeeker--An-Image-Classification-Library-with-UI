'''
@author: Bappy Ahmed
Email: entbappy73@gmail.com
Date: 06-sep-2021
'''

from utils import data_manager as dm
from utils.config import configureData
from utils.config import configureModel
import tensorflow as tf
import os
import numpy as np

config_data = configureData()
config_model = configureModel()

#Manage Image
image_list = os.listdir(config_data['PREDICTION_DATA_DIR'])


def predict():

    """The logic for prediction step.
  
    This method should contain the mathematical logic prediction.
    This typically includes the forward pass with respect to updated weights.

     Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `nd array` containing values.Typically, the
      values of the `Model`'s metrics are returned. Example:
      `[[0,1]]`.

    """

    # load model
    model_path = f"New_trained_model/{'new' + config_model['MODEL_NAME'] + '.h5'}"
    model = tf.keras.models.load_model(model_path)
    for image in image_list:
        predict = dm.manage_input_data(os.path.join(config_data['PREDICTION_DATA_DIR'],image))
        result = model.predict(predict)
        results = np.argmax(result, axis=-1)
        print(f"Original image : {image}. Predicted as {results}")




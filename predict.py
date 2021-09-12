
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        from utils import data_manager as dm
        from utils.config import configureData
        from utils.config import configureModel

        config_data = configureData()
        config_model = configureModel()

        # load model
        # model = load_model('model.h5')
        model_path = f"New_trained_model/{'new' + config_model['MODEL_NAME'] + '.h5'}"
        print('Loading...', model_path)
        model = load_model(model_path)

        # summarize model
        #model.summary()
        imagename = self.filename
        predict = dm.manage_input_data(imagename)
        result = model.predict(predict)
        results = np.argmax(result, axis=-1)
        print(dm.class_name())
        print(results)

        if results[0] == 1:
            prediction = 'dog'
            return [{ "image" : prediction}]
        else:
            prediction = 'cat'
            return [{ "image" : prediction}]



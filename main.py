'''
@Author: Bappy Ahmed
Date: 03 sep 2021
Email: entbappy73@gmail.com
'''

from flask import Flask, render_template, request,jsonify
import pandas as pd
from com_in_ineuron_ai_utils.utils import decodeImage
from flask_cors import CORS, cross_origin
from predict import dogcat

user_data = pd.read_csv('input_data.csv')


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = dogcat(self.filename)


app = Flask(__name__) # initializing a flask app
CORS(app)

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/input',methods=['GET'])  # route to display the home page
def input_form():
    return render_template("input_form.html")

@app.route('/train',methods=['POST','GET']) # route to show the predictions in a web UI
def train_func():
    if request.method == 'POST':
        try:
            # Data config
            TRAIN_DATA_DIR = request.form['TRAIN_DATA_DIR']
            VALID_DATA_DIR = request.form['VALID_DATA_DIR']
            CLASSES =int(request.form['CLASSES'])
            IMAGE_SIZE = request.form['IMAGE_SIZE']
            AUGMENTATION = request.form['AUGMENTATION']
            BATCH_SIZE = int(request.form['BATCH_SIZE'])


            # Model config
            MODEL_OBJ = request.form['MODEL_OBJ']
            MODEL_NAME = request.form['MODEL_NAME']
            FREEZE_ALL = request.form['FREEZE_ALL']
            OPTIMIZER = request.form['OPTIMIZER']
            EPOCHS = int(request.form['EPOCHS'])
            LOSS_FUNC = request.form['LOSS_FUNC']


            user_data['TRAIN_DATA_DIR'] = TRAIN_DATA_DIR
            user_data['VALID_DATA_DIR'] = VALID_DATA_DIR
            user_data['AUGMENTATION'] = AUGMENTATION
            user_data['CLASSES'] = CLASSES
            user_data['IMAGE_SIZE'] = IMAGE_SIZE
            user_data['BATCH_SIZE'] = BATCH_SIZE

            user_data['MODEL_OBJ'] = MODEL_OBJ
            user_data['MODEL_NAME'] = MODEL_NAME
            user_data['EPOCHS'] = EPOCHS
            user_data['FREEZE_ALL'] = FREEZE_ALL
            user_data['OPTIMIZER'] = OPTIMIZER
            user_data['LOSS_FUNC'] = LOSS_FUNC

            user_data.to_csv('input_data.csv',index=False)

            import train_engine
            # return render_template('loading.html')
            hist = train_engine.train()

            return render_template('input_form.html', output = hist)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


@app.route('/test',methods=['GET','POST'])  # route to display the home page
def predcit():
    return render_template("predict.html")


@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictiondogcat()
    return jsonify(result)




if __name__ == "__main__":
    clApp = ClientApp()
	# app.run(debug=True)
    app.run(host='127.0.0.1', port=8000, debug=True)

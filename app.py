from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import efficientnet.keras as efn
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import keras.utils as image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'processoreffB2.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    preds = np.squeeze(model.predict(img)[0]) 
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("basepath:-----",basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        clss_index = np.argmax(preds)    # Simple argmax
        #class names:
        clss_name = ["Bend_Pin","Good","Missing_Pins","Short_Pins"]
        return clss_name[clss_index]

    return None

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from LOAD import *

app = Flask(__name__, template_folder='templates')

def init():
    global model#,graph
    # load the pre-trained Keras model
    with tf.device('/cpu:0'):
        model = load_model('model1/model.h5')
    # graph = tf.get_default_graph()

@app.route('/')
def upload_file():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['POST'])
def upload_image_file():
   if request.method == 'POST':
        img = Image.open(request.files['file'].stream).convert("RGB")
        img = img.resize((150,150))
        # print(img)
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,150,150,3)
        # with graph.as_default():
        y_pred = model.predict_classes(im2arr)
        clases = ['blenheim_spaniel',
                  'chihuahua',
                  'japanses spaniel',
                  'maltese',
                  'papillon',
                  'pekinese',
                  'shih-tzu',
                  'toy_terrier']
        return 'Predicted Number: ' + clases[y_pred[0]]
		
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    init()
    app.run(debug = True)
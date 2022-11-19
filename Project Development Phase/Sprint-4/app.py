#Importing the required libraries

from __future__ import division, print_function
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import model_from_json




global graph
graph = tf.compat.v1.get_default_graph()

json_file=open("final_model.json","r")
loaded_model_json = json_file.read()
json_file.close()

app = Flask(__name__)
found = [
        "https://en.wikipedia.org/wiki/Amorphophallus_titanum",
        "https://en.wikipedia.org/wiki/Great_Indian_bustard",
        "https://en.wikipedia.org/wiki/Cypripedioideae",
        "https://en.wikipedia.org/wiki/Pangolin",
        "https://en.wikipedia.org/wiki/Spoon-billed_sandpiper",
        "https://en.wikipedia.org/wiki/Seneca_white_deer",
        ]
predictions = ["Corpse Flower", 
           "Great Indian Bustard", 
           "Lady's slipper orchid", 
           "Pangolin", 
           "Spoon Billed Sandpiper", 
           "Seneca White Deer"
          ]

@app.route('/', methods = ['GET'])
def index():
	return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
	text = 'get'
	if request.method == 'POST':
		f = request.files['image']
		
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)
		img = tf.keras.utils.load_img(file_path, target_size = (224,224))

		x = tf.keras.utils.img_to_array(img)
		x = np.expand_dims(x, axis = 0)
		
		with graph.as_default():
			loaded_model = model_from_json(loaded_model_json)
			loaded_model.load_weights("final_model.h5")
			print("Model loaded.")
			predict_x=loaded_model.predict(x) 
			preds=np.argmax(predict_x,axis=1)
			print("Predicted Species " + str(predictions[preds[0]]))
		data = {"name": predictions[preds[0]], "showMore": found[preds[0]]}
	return render_template("prediction.html", predicted_info = data)

if __name__ == '__main__':
	app.run(threaded = False, debug = True)

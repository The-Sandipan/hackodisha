import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from model1 import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import load_img
import tensorflow as tf

from tensorflow.keras.utils import load_img
import io
import PIL.JpegImagePlugin



app = Flask(__name__)
model = load_model()

class_names = ['PITUITARY','NOTUMOR','MENINGIOMA','GLIOMA']  



@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('home.html', message='No selected file')

        if file:
            image = PIL.JpegImagePlugin.JpegImageFile(file)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img = Image.open(buffer)
            img = img.resize((224, 224)) 
            img_array1 = np.array(img) / 255.0  

            img_array1 = np.expand_dims(img_array1, axis=-1)
            img_array1 = np.repeat(img_array1, 3, axis=-1)
            img_array1 = np.expand_dims(img_array1, axis=0)

            img_array1 = preprocess_input(img_array1)

            predictions = model.predict(img_array1)
            top_class_index = np.argmax(predictions)
            predicted_class = class_names[top_class_index]

            return render_template('home.html', message=f'Predicted: {predicted_class}')
    
    return render_template('home.html', message='Upload an image')


if __name__ == '__main__':
    app.run(debug=True)






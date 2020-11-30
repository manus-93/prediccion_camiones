#from flask import request, jsonify, Flask,render_template

import numpy as np
from keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import base64
from io import BytesIO


def get_model():
    global model
    model = load_model('camiones_model_full.h5')
    print(" * Model loaded!")

print(" * Loading Keras model...")
get_model()                                                         # cargo el modelo una unica vez 

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255
    return image

x=open("texto1.txt","r")                                               # aca ingreso l ruta de los textos en base 64
encoded=x.read()
print(encoded[0:10])                                                # imprimo primeras 10 lineas del encode

decoded = base64.b64decode(encoded)
image = Image.open(BytesIO(decoded))
processed_image = preprocess_image(image, target_size=(240, 352))   # preprocesmiento
prediction = model.predict(processed_image).tolist()                # etapa de prediccion
print("porcentaje de cada categoria: ")
print("Camion con material","Nada","Otro","Tolva vacia")
print(np.array(prediction[0])*100.0/sum(np.array(prediction[0])))






'''
@app.route('/', methods=['GET'])
def main():
    #path is filename string
    image_file = url_for('static', filename=path)

    return render_template('main.html', image_file=image_file)
    '''
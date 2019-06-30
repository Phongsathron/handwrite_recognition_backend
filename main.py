from flask import Flask, request, jsonify

import numpy as np
from keras.models import Sequential
from keras import layers
import cv2

model = None

def model(name):
    model = Sequential()

    if name == 'lenet5':
        model.add(layers.Conv2D(filters=6, kernel_size=(5,5), strides=1, activation='relu', input_shape=(32,32,1)))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Conv2D(filters=16, kernel_size=(5,5), strides=1, activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.load_weights('lenet5_digit_1.h5')

        return model

app = Flask(__name__)

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {name}!'

@app.route('/digit/lenet5', methods=['POST'])
def lenet5_predict():
    if request.method == 'POST':
        f = request.files['image']
        f.save('temp.png')

        image = cv2.imread('temp.png')
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prediction = model.predict(np.expand_dims(image, axis=0))
        print(prediction.argmax())

    return jsonify({
        'success': True,
        'result': prediction.argmax()
    })

if __name__ == '__main__':
    model = model('lenet5')
    app.run(host='0.0.0.0', debug=True)

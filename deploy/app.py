from pickle import load 
import tensorflow as tf 
import tensorflow_text
# from tensorflow import Module 
# from keras_preprocessing.sequence import pad_sequences 
# from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose 

# Serve model as a flask application
# import pickle
# import numpy as np
from flask import Flask, request




translator = None
app = Flask(__name__)




def load_model():
    global translator
    # model variable refers to the global variable
    # with open('iris_trained_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    translator = tf.saved_model.load('./metadata/translator')
    print('load_model')


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        print('received = ', data)
        print(type(data))
        prediction = translator(data)  # runs globally loaded model on the data
        print("\n")
        print('Translated result is = ', prediction)
        print(type(prediction))
        print('Translated result is = ', prediction[0])
        print(type(prediction[0]))
        print('Translated result is = ', prediction[0].numpy().decode("utf-8"))

    return prediction[0].numpy().decode("utf-8")


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)


    # translation, _, _ = translator('este é o primeiro livro que eu fiz.')
    # print(translation.numpy().decode("utf-8"))
import flask
import pickle

import joblib
import pandas as pd
from flask import Flask, request, jsonify


from model_building import utils

app = Flask(__name__)


def predict_spam_or_ham(input_string):
    # Load the trained model and CountVectorizer
    naive_bayes = joblib.load('./naive_bayes_model.pkl')
    count_vectorizer = joblib.load('./count_vectorizer.pkl')
    print(input_string)
    # Preprocess the input string
    preprocessed_input = count_vectorizer.transform([input_string['comment']])
    # Make the prediction
    prediction = naive_bayes.predict(preprocessed_input)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        prediction = predict_spam_or_ham(data)
        if prediction == 0:
            label = "ham"
        else:
            label = "spam"
        print(label)
        response = {'prediction': label}
        return jsonify(response)


if __name__ == '__main__':
    app.run()
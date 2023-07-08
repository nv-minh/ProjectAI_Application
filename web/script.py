import os
import flask
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


def predict_spam_or_ham(input_string):
    # Load the trained model and CountVectorizer
    naive_bayes = joblib.load('../naive_bayes_model.pkl')
    count_vectorizer = joblib.load('../count_vectorizer.pkl')
    # Preprocess the input string
    preprocessed_input = count_vectorizer.transform([input_string])
    # Make the prediction
    prediction = naive_bayes.predict(preprocessed_input)
    return prediction[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input_ = request.form.to_dict()
    prediction = predict_spam_or_ham(input_["cmt"])
    if prediction == 0:
        label = "ham"
    else:
        label = "spam"
    return render_template("result.html", prediction=label, c=input_["cmt"])


if __name__ == '__main__':
    app.run()

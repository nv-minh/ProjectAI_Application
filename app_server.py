from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)


def predict_spam_or_ham(input_string):
    # Load the trained model and CountVectorizer
    print(input_string)
    naive_bayes = joblib.load('naive_bayes_model.pkl')
    count_vectorizer = joblib.load('count_vectorizer.pkl')
    # Preprocess the input string
    preprocessed_input = count_vectorizer.transform([input_string])
    # Make the prediction
    prediction = naive_bayes.predict(preprocessed_input)
    return prediction[0]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_ = request.form['cmt']
    prediction = predict_spam_or_ham(input_)
    if prediction == 0:
        label = "ham"
    else:
        label = "spam"
    return label


if __name__ == '__main__':
    app.run()

import flask
import pickle
import pandas as pd
from flask import Flask, request, jsonify


from model_building import utils

app = Flask(__name__)


def predict_spam_or_ham(data):
    messages_raw = pd.read_excel('./message_modified_v1.2.xlsx', dtype={'msg': str})
    list_content = messages_raw['msg'].to_list()
    list_content.append(data['comment'])
    list_content = utils.entity_tagging(list_content)
    list_content_vec, list_len_sms, dictionary = utils.vectorize(list_content, 'bow')
    load_clf = pickle.load(open('bow_svm_clf.pkl', 'rb'))
    label_predict = load_clf.predict(list_content_vec[:])
    return label_predict[-1]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        prediction = predict_spam_or_ham(data)
        if prediction == 0:
            label = "ham"
        else:
            label = "spam"
        response = {'prediction': label}
        return jsonify(response)


if __name__ == '__main__':
    app.run()
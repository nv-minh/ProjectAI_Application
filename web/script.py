import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import gensim

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


import gensim.models.keyedvectors as word2vec

# model_embedding = word2vec.KeyedVectors.load('./word.model')

word_labels = []
max_seq = 70
embedding_size = 128

word_labels = model_embedding.key_to_index.keys()


def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)

    so_lan_du_1_cau = 0

    for i in range(max_seq):
        indexword = i % lencmt

        so_lan_du_1_cau = i // lencmt
        if max_seq - so_lan_du_1_cau * lencmt < lencmt:
            break
        if words[indexword] in word_labels:
            matrix[i] = model_embedding[words[indexword]]

    matrix = np.array(matrix)
    return matrix


def ValuePredictor(comment):
    comment = gensim.utils.simple_preprocess(comment)
    comment = ' '.join(comment)
    comment = ViTokenizer.tokenize(comment)
    # print(comment)
    # model_sentiment = tf.keras.models.load_model("models.h5")
    maxtrix_embedding = np.expand_dims(comment_embedding(comment), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    result = model_sentiment.predict(maxtrix_embedding)
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input_ = request.form.to_dict()

        result = ValuePredictor(input_["cmt"])

        pos = round(result[0][1], 10)
        neg = round(result[0][0], 10)

        max_index = np.argmax(result)
        if max_index == 1:
            prediction = "Tích cực"
            p = round(result[0][1] * 100, 2)

        if max_index == 0:
            prediction = "Tiêu cực"
            p = round(result[0][0] * 100, 2)

        return render_template("result.html", prediction=prediction, r=result, c=input_["cmt"], pos=pos, neg=neg, p=p)


if __name__ == '__main__':
    app.run()

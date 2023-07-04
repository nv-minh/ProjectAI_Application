from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model
from sklearn.model_selection import KFold
import codecs
import numpy as np
from gensim import corpora, models
import pickle


def load_corpus(filename):
    list_label = []
    list_content = []
    f = codecs.open(filename, 'r', 'utf-8')
    for line in f:
        line = line.strip().split('\t')
        list_label.append(line[0])
        list_content.append(line[1])
    return list_label, list_content


def doc_2_vec(content, label, vector, df_below=3, df_above=1.0, length=1):
    len_sms = []
    list_corpus = []
    list_corpus_new = []
    for line in content:
        line = line.split()
        list_corpus.append(line)
        len_sms.append(len(line))
    bigram = models.phrases.Phrases(list_corpus, min_count=10)
    list_corpus = list(bigram[list_corpus])
    for line in list_corpus:
        temp = []
        for item in line:
            item = item.split('_')
            if (len(item) > 1) and (len(set(item).intersection([u'date', u'phone', u'link', u'currency', u'emoticon'])) > 0):
                for word in item:
                    temp.append(word)
            elif len(set(item).intersection([u'date', u'phone', u'link', u'currency', u'emoticon'])) == 0:
                word = '_'.join(item)
                temp.append(word)
            else:
                temp.append(item[0])
        list_corpus_new.append(temp)
    dictionary = corpora.Dictionary(list_corpus_new)
    dictionary.filter_extremes(no_below=df_below, no_above=df_above, keep_n=100000)
    temp_corpus_bow = [dictionary.doc2bow(line) for line in list_corpus_new]
    content_bow = np.zeros((len(list_corpus), len(dictionary.keys())))
    for i in range(len(temp_corpus_bow)):
        for item in temp_corpus_bow[i]:
            content_bow[i][item[0]] = item[1]
    if vector == 'bow':
        content_vec = content_bow
    len_sms = np.asarray(len_sms)
    len_sms = np.reshape(len_sms, (len(len_sms), 1))
    if length == 1:
        content_vec = np.concatenate((content_vec, len_sms), axis=1)
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(label)
    return content_vec, label, len_sms, dictionary


def vectorize(content, vector, df_below=3, df_above=1.0, length=1):
    len_sms = []
    list_corpus = []
    list_corpus_new = []
    for line in content:
        line = line.split()
        list_corpus.append(line)
        len_sms.append(len(line))
    bigram = models.phrases.Phrases(list_corpus, min_count=10)
    list_corpus = list(bigram[list_corpus])
    for line in list_corpus:
        temp = []
        for item in line:
            item = item.split('_')
            if (len(item) > 1) and (len(set(item).intersection([u'date', u'phone', u'link', u'currency', u'emoticon'])) > 0):
                for word in item:
                    temp.append(word)
            elif len(set(item).intersection([u'date', u'phone', u'link', u'currency', u'emoticon'])) == 0:
                word = '_'.join(item)
                temp.append(word)
            else:
                temp.append(item[0])
        list_corpus_new.append(temp)
    dictionary = corpora.Dictionary(list_corpus_new)
    dictionary.filter_extremes(no_below=df_below, no_above=df_above, keep_n=100000)
    temp_corpus_bow = [dictionary.doc2bow(line) for line in list_corpus_new]
    content_bow = np.zeros((len(list_corpus), len(dictionary.keys())))
    for i in range(len(temp_corpus_bow)):
        for item in temp_corpus_bow[i]:
            content_bow[i][item[0]] = item[1]
    tfidf = models.TfidfModel(temp_corpus_bow)
    temp_corpus_tfidf = tfidf[temp_corpus_bow]
    content_tfidf = np.zeros((len(list_corpus), len(dictionary.keys())))
    for i in range(len(temp_corpus_tfidf)):
        for item in temp_corpus_tfidf[i]:
            content_tfidf[i][item[0]] = item[1]
    if vector == 'bow':
        content_vec = content_bow
    elif vector == 'tfidf':
        content_vec = content_tfidf
    len_sms = np.asarray(len_sms)
    len_sms = np.reshape(len_sms, (len(len_sms), 1))
    if length == 1:
        content_vec = np.concatenate((content_vec, len_sms), axis=1)
    return content_vec, len_sms, dictionary



def build_classifier_svm(content, label):
    clf = svm.LinearSVC(C=0.1, max_iter=10000, dual=False).fit(content, label)
    return clf

def classifying_svm(content, content_vec, clf, label):
    label_predict = clf.predict(content_vec)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def predicting_svm(content, content_vec, clf):
    label_predict = clf.predict(content_vec)
    for i in range(len(content)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    return label_predict


def evaluation(list_false_positive, list_false_negative, list_true_positive, list_true_negative):
    print('False Positive Rate: ' + str(sum(list_false_positive)*20) + '%')
    print('False Negative Rate: ' + str(sum(list_false_negative)*20) + '%')
    print('True Positive Rate: ' + str(sum(list_true_positive)*20) + '%')
    print('True Negative Rate: ' + str(sum(list_true_negative)*20) + '%')


def count_vocab(list_content):
    list_vocab = []
    list_corpus = []
    count = 1
    for line in list_content:
        line = line.split()
        list_vocab += line
        list_corpus.append(line)
        count += 1
    print(len(list_vocab))
    temp = list(set(list_vocab))
    print(len(temp))
    dictionary = corpora.Dictionary(list_corpus)
    print(dictionary)
    return temp, list_corpus


def classification(content_test, content_vec_train, content_vec_test, label_train, label_test, classifier, vectorize_method):
    if classifier == 'svm':
        clf = build_classifier_svm(content_vec_train, label_train)
        # if vectorize_method == 'bow':
        #     pickle.dump(clf, open(r'C:\KhoiNXM\Spam message Vietnamese\Dev\vietnamese-spam-sms-filtering\models\bow_svm_clf.pkl', 'wb'))
        matrix = classifying_svm(content_test, content_vec_test, clf, label_test)
    false_positive = matrix[1, 0] / float(sum(matrix[1]))
    false_negative = matrix[0, 1] / float(sum(matrix[0]))
    true_positive = matrix[0, 0] / float(sum(matrix[0]))
    true_negative = matrix[1, 1] / float(sum(matrix[1]))
    return false_positive, false_negative, true_positive, true_negative


def kfold_classification(content, content_vec, label, classifier, fold, vectorize_method):
    content = np.asarray(content)
    list_false_positive = []
    list_false_negative = []
    list_true_positive = []
    list_true_negative = []
    kf = KFold(n_splits=fold, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(label):
        content_train, content_test = content[train_index], content[test_index]
        content_vec_train, content_vec_test = content_vec[train_index], content_vec[test_index]
        label_train, label_test = label[train_index], label[test_index]
        false_positive, false_negative, true_positive, true_negative = \
            classification(content_test, content_vec_train, content_vec_test, label_train, label_test, classifier, vectorize_method)
        list_false_positive.append(false_positive)
        list_false_negative.append(false_negative)
        list_true_positive.append(true_positive)
        list_true_negative.append(true_negative)
    return list_false_positive, list_false_negative, list_true_positive, list_true_negative


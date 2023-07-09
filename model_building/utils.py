from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import KFold
import numpy as np
from gensim import corpora, models
import re
import codecs

from sklearn.naive_bayes import MultinomialNB

date_pattern = []
phone_pattern = []
link_pattern = []
currency_pattern = []
emoticon_pattern = []

date_string = ' date '
phone_string = ' phone '
link_string = ' link '
currency_string = ' currency '
emoticon_string = ' emoticon '
# Regex for date
date_pattern.append(r'\d{1,2}/\d{1,2}/\d{2,4}')
date_pattern.append(r'\d{1,2}-\d{1,2}-\d{2,4}')
date_pattern.append(r'\+\d{1,2}:\d{1,4}')
date_pattern.append(r'\d{1,2}/\d{1,4}')
date_pattern.append(r'\d{1,2}-\d{1,4}')
date_pattern.append(r'\d{1,2}h\d{0,2}')
# Regex for phone
phone_pattern.append(r'\+\d{10,12}')
phone_pattern.append(r'\d{3,5}\.\d{3,4}\.\d{3,5}')
phone_pattern.append(r'\d{8,12}')
phone_pattern.append(r'1800\d{4}')
phone_pattern.append(r'195')
phone_pattern.append(r'900')
phone_pattern.append(r'1342')
phone_pattern.append(r'191')
phone_pattern.append(r'888')
phone_pattern.append(r'333')
phone_pattern.append(r'1414')
phone_pattern.append(r'1576')
phone_pattern.append(r'8170')
phone_pattern.append(r'9123')
phone_pattern.append(r'9118')
phone_pattern.append(r'266')
phone_pattern.append(r'153')
phone_pattern.append(r'199')
phone_pattern.append(r'9029')
phone_pattern.append(r'8049')
phone_pattern.append(r'1560')
phone_pattern.append(r'9191')
# Regex for link
link_pattern.append(r'www\..*')
link_pattern.append(r'http://.*')
# Regex for currency
currency_pattern.append(r'[0-9|\,\.]{3,}VND')
currency_pattern.append(r'[0-9|\.]{3,}VND')
currency_pattern.append(r'[0-9|\.]{3,}d')
currency_pattern.append(r'[0-9|\.]{3,}đ')
currency_pattern.append(r'[0-9|\.]{3,}tr')
currency_pattern.append(r'[0-9|\.]{3,}Tr')
currency_pattern.append(r'[0-9|\.]{3,}TR')
# Regex for emoticon
emoticon_pattern.append(r'o.O')
emoticon_pattern.append(r'O.o')
emoticon_pattern.append(r'\(y\)')
emoticon_pattern.append(r'\(Y\)')
emoticon_pattern.append(r':v')
emoticon_pattern.append(r':V')
emoticon_pattern.append(r':3')
emoticon_pattern.append(r'-_-')
emoticon_pattern.append(r'\^_\^')
emoticon_pattern.append(r'<3')
emoticon_pattern.append(r':-\*')
emoticon_pattern.append(r':\*')
emoticon_pattern.append(r":'\(")
emoticon_pattern.append(r':p ')
emoticon_pattern.append(r':P')
emoticon_pattern.append(r':d')
emoticon_pattern.append(r':D')
emoticon_pattern.append(r':-\?')
emoticon_pattern.append(r'>\.<')
emoticon_pattern.append(r'><')
emoticon_pattern.append(r':-\w ')
emoticon_pattern.append(r':\)\)')
emoticon_pattern.append(r';\)\)')
emoticon_pattern.append(r'=\)\)')
emoticon_pattern.append(r':-\)')
emoticon_pattern.append(r':\)')
emoticon_pattern.append(r':\]')
emoticon_pattern.append(r'=\)')
emoticon_pattern.append(r':-\(')
emoticon_pattern.append(r':\(')
emoticon_pattern.append(r':\[')
emoticon_pattern.append(r'=\(')
emoticon_pattern.append(r':-\|')
emoticon_pattern.append(r':\|')
emoticon_pattern.append(r':-\/')
emoticon_pattern.append(r':\/')
emoticon_pattern.append(r':-\\')
emoticon_pattern.append(r':\\')
emoticon_pattern.append(r':-\$')
emoticon_pattern.append(r':\$')
emoticon_pattern.append(r':-\!')
emoticon_pattern.append(r':\!')
emoticon_pattern.append(r':-\+')
emoticon_pattern.append(r':\+')
emoticon_pattern.append(r':-@')
emoticon_pattern.append(r':@')
emoticon_pattern.append(r':-#')
emoticon_pattern.append(r':#')
emoticon_pattern.append(r':-&')
emoticon_pattern.append(r':&')
stop_list = ['.', ',', '/', '?', ';', ':', '&', '@', '!', '`', "'", '"', '>', '<', '*', '%', '#', '(', ')', '[', ']',
             '-', '_', '=', '+', '{', '}', '~', '$', '^', '*', '|', '\\']



def entity_tagging(corpus):
    corpus_new = []
    for line in corpus:
        sent = []
        for word in line.split():
            for date_pat in date_pattern:
                word = re.sub(date_pat, date_string, word)
            for currency_pat in currency_pattern:
                word = re.sub(currency_pat, currency_string, word)
            for phone_pat in phone_pattern:
                word = re.sub(phone_pat, phone_string, word)
            for link_pat in link_pattern:
                word = re.sub(link_pat, link_string, word)
            for emoticon_pat in emoticon_pattern:
                word = re.sub(emoticon_pat, emoticon_string, word)
            sent.append(word)
        sent = ' '.join(sent)
        for item in stop_list:
            sent = sent.replace(item, ' ')
        sent = sent.lower()
        sent = sent.split()
        for i in range(len(sent)):
            if sent[i].isdigit():
                sent[i] = 'number'
        corpus_new.append(' '.join(sent))
    return corpus_new

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
    for line in content[:-1]:
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
    return content_vec, len_sms, dictionary


def build_classifier_nb(content, label):
    clf = MultinomialNB().fit(content, label)
    return clf

def evaluation(list_false_positive, list_false_negative, list_true_positive, list_true_negative):
    print('False Positive Rate: ' + str(sum(list_false_positive)*20) + '%')
    print('False Negative Rate: ' + str(sum(list_false_negative)*20) + '%')
    print('True Positive Rate: ' + str(sum(list_true_positive)*20) + '%')
    print('True Negative Rate: ' + str(sum(list_true_negative)*20) + '%')




def classification(content_test, content_vec_train, content_vec_test, label_train, label_test, classifier, vectorize_method):
    if classifier == 'nb':
        clf = build_classifier_nb(content_vec_train, label_train)
        matrix = classifying_nb(content_test, content_vec_test, clf, label_test)
    false_positive = matrix[1, 0] / float(sum(matrix[1]))
    false_negative = matrix[0, 1] / float(sum(matrix[0]))
    true_positive = matrix[0, 0] / float(sum(matrix[0]))
    true_negative = matrix[1, 1] / float(sum(matrix[1]))
    return false_positive, false_negative, true_positive, true_negative

def build_classifier_nb(content, label):
    clf = MultinomialNB().fit(content, label)
    return clf



def classifying_nb(content, content_vec, clf, label):
    label_predict = []
    probability_list = clf.predict_proba(content_vec)
    for i in range(len(probability_list)):
        if probability_list[i][1] > 0.6:
            label_predict.append(1)
        else:
            label_predict.append(0)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def predicting_nb(content, content_vec, clf):
    label_predict = []
    probability_list = clf.predict_proba(content_vec)
    for i in range(len(probability_list)):
        if probability_list[i][1] > 0.6:
            label_predict.append(1)
        else:
            label_predict.append(0)
    for i in range(len(content)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    return label_predict




def evaluation(list_false_positive, list_false_negative, list_true_positive, list_true_negative):
    print('False Positive Rate: ' + str(sum(list_false_positive)*20) + '%')
    print('False Negative Rate: ' + str(sum(list_false_negative)*20) + '%')
    print('True Positive Rate: ' + str(sum(list_true_positive)*20) + '%')
    print('True Negative Rate: ' + str(sum(list_true_negative)*20) + '%')




def classification(content_test, content_vec_train, content_vec_test, label_train, label_test, classifier, vectorize_method):
    if classifier == 'nb':
        clf = build_classifier_nb(content_vec_train, label_train)
        matrix = classifying_nb(content_test, content_vec_test, clf, label_test)
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
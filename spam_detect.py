
import pandas as pd
from model_building import utils
import pickle



# Reads in saved classification model


input_message = "Tang ngay phieu giam gia 1 trieu dong cho cac cap doi dang ky tour du lich Viet Nam – Singapore trong ngay 14-2. Hay cung nhau co mot chuyen di day thu vi voi CRMVIET. Dang ky ngay tai : crmviet.vn"
messages_raw = pd.read_excel('./message_modified_v1.2.xlsx', dtype={'msg': str})
list_content = messages_raw['msg'].to_list()
list_content.append(input_message)
list_content = utils.entity_tagging(list_content)
list_content_vec, list_len_sms, dictionary = utils.vectorize(list_content, 'bow')


load_clf = pickle.load(open('bow_svm_clf.pkl', 'rb'))
label_predict = load_clf.predict(list_content_vec[:])
if label_predict[-1] == 0:
    label = 'ham'
else:
    label = 'spam'
print('This is %s message' % label)
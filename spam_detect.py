
import pandas as pd
from model_building import utils
import pickle



# Reads in saved classification model


input_message = "Mình đăng kí đổi modem loại mới dc ko ad"
messages_raw = pd.read_excel('./message_modified_v1.2.xlsx', dtype={'msg': str})
list_content = messages_raw['msg'].to_list()
list_content.append(input_message)
list_content = utils.entity_tagging(list_content)
list_content_vec, list_len_sms, dictionary = utils.vectorize(list_content, 'bow')


load_clf = pickle.load(open('bow_nb_clf.pkl', 'rb'))
label_predict = load_clf.predict(list_content_vec[:])
if label_predict[-1] == 0:
    label = 'ham'
else:
    label = 'spam'
print('This is %s message' % label)
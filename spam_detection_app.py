
import pandas as pd
from model_building import entity_tagging, utils
import pickle



# Reads in saved classification model


input_message = "That Tuyet Voi de So Huu CanHo Sieu CAOCAP ParkParoma Dep NHAT Vinhomes, Vi tri VANG gia Tot NHAT. Goi ngay e Long 090 282 5353 tham quan nha mau va giu cho"
messages_raw = pd.read_excel('./message_modified_v1.2.xlsx', dtype={'msg': str})
list_content = messages_raw['msg'].to_list()
list_content.insert(0, input_message)
list_content = entity_tagging.entity_tagging(list_content)
list_content_vec, list_len_sms, dictionary = utils.vectorize(list_content, 'bow')


load_clf = pickle.load(open('bow_svm_clf.pkl', 'rb'))
label_predict = load_clf.predict(list_content_vec[:1])
if label_predict[0] == 0:
    label = 'ham'
else:
    label = 'spam'
print('This is %s message' % label)


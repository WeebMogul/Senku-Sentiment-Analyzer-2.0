from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import make_scorer,precision_score,recall_score,f1_score,accuracy_score
import pandas as pd

import time

train = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv")

vect = TfidfVectorizer(min_df=5,max_df=0.8,sublinear_tf=True)
train_vectors = vect.fit_transform(train['Content'])
test_vectors = vect.transform(test['Content'])
print(vect.get_feature_names())

class_linear = svm.SVC(kernel='linear')
t0 = time.time()
class_linear.fit(train_vectors,train['Label'])
t1 = time.time()
pred_linear = class_linear.predict(test_vectors)
t2 = time.time()
time_train = t1-t0
time_test = t2-t1

print("Training time: %fs; Prediction time: %fs" % (time_train, time_test))
report = classification_report(test['Label'], pred_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])
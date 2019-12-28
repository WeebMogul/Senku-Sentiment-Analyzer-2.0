#Sklearn

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.utils import resample,shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.metrics import make_scorer, precision_score, recall_score,f1_score, accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import make_scorer,precision_score,recall_score,accuracy_score,f1_score

#NLTK
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#Other
import pandas as pd
import re
import string
import matplotlib.pyplot as pypl
import numpy as np
import seaborn as sd
#import imblearn
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

contract = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
stopword = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def penntag(pen):
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[pen[:2]]
    except:
        return 'n'


def comment_cleaner(comm, comment_array):
    temp_comm = []
    megos = ' '
    uncontracted = ' '.join([contract[word] if word in contract else word for word in comm.lower().split()])
    stopwords_removed = [word for word in uncontracted.lower().split() if word not in stopword]
    POS_words = nltk.pos_tag(stopwords_removed)
    for i in range(0, len(POS_words)):
        lemmas = lemma.lemmatize(POS_words[i][0], pos=penntag(POS_words[i][1]))
        temp_comm.append(lemmas)
    megos = ' '.join(word for word in temp_comm)
    return megos


for ep in range(1, 2):
    # kaguya_file = 'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' +str(1)+' .csv'

    # df1 = pd.read_csv('D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(1) + ' .csv',index_col=0, encoding='utf-8-sig')
    # df2 = pd.read_csv('D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(2) + ' .csv',index_col=0, encoding='utf-8-sig')
    # df3 = pd.read_csv('D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(23) + ' .csv',index_col=0, encoding='utf-8-sig')
    df1 = pd.read_csv(
        'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 1 .csv',
        index_col=0, encoding='utf-8-sig')
    df2 = pd.read_csv(
        'D:Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 2 .csv',
        index_col=0, encoding='utf-8-sig')
    df3 = pd.read_csv(
        'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 3 .csv',
        index_col=0, encoding='utf-8-sig')
    df4 = pd.read_csv(
        'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 4 .csv',
        index_col=0, encoding='utf-8-sig')

    df12 = pd.concat([df1, df2, df3
                      ])
    train_array = []
    test_array = []
    train_target = []
    comtest_array = []
    # df = df.sample(frac=1)
    # Convert dataframe values into string
    df12 = df12[['Comments', 'Actual Polarity']]
    df12['Comments'] = df12['Comments'].astype(str)
    df12['Length'] = df12['Comments'].apply(len)
    df12 = df12[df12['Length'] > 5]
    df12['Comments'] = df12['Comments'].apply(lambda s: comment_cleaner(s, train_array))

    # Remove punctuation marks and tokenize each and every word
    df12['Comments'] = df12['Comments'].str.replace('[^\w\s]', ' ')
    df12['Comments'] = df12['Comments'].str.replace('[\d+]', ' ')
    df12['Comments'] = df12['Comments'].str.replace('(^| ).(( ).)*( |$)', ' ')

    # Split into positive and negative datasets
    pos_df = df12[df12['Actual Polarity'] == 1]
    neg_df = df12[df12['Actual Polarity'] == 0]
    neu_df = df12[df12['Actual Polarity'] == 2]

    df_len = len(pos_df)

    #train_df = pd.concat([pos_df, neg_df,neu_df])
    train_df = pd.concat([pos_df, neg_df,neu_df])
    train_df = train_df.reset_index(drop=True)

    y = train_df['Actual Polarity']

    x_train, x_test, y_train, y_test = train_test_split(train_df['Comments'], train_df['Actual Polarity'], test_size=0.2,random_state=22)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    x_tr = vec.fit_transform(x_train)
    x_ts = vec.transform(x_test)

    sm = RandomOverSampler(random_state=77)

    X_train_res, y_train_res = sm.fit_sample(x_tr, y_train)

    scores = { 'Accuracy' : make_scorer(accuracy_score),
           'Precision': make_scorer(precision_score),
           'Recall': make_scorer(recall_score),
          'F1-Score': make_scorer(f1_score)
          }


# Train SVM model
    C = [0.00001,0.0001,0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0]
    gammas = [0.00001,0.0001,0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0]
    #iter = [1,10,50,100,500,1000]
    params = {'C': C,'gamma':gammas}

 #clas_linear = svm.SVC(kernel='linear', C=1.0, gamma=0.001)
 #clas_linear.fit(x,train_target)
    '''

 pred_linear = clas_linear.predict(x_test)
 print(accuracy_score(y_test, pred_linear))
 print(precision_score(y_test, pred_linear))
 print(recall_score(y_test, pred_linear))
    '''
 #cv_svm = cross_validate(clas_linear,x,train_target,cv=10,return_train_score=True)
 #print('Training accuracy ', np.mean(cv_svm['train_score']))
 # #print('Testing accuracy ', np.mean(cv_svm['test_score']))
    '''
 cv_pr = cross_validate(clas_linear,x,train_target,cv=10,scoring=scores,return_train_score=True)
 # print(cv_pr)
 print('Training accuracy :',np.mean(cv_pr['train_Accuracy']))
 print('Training precision :',np.mean(cv_pr['train_Precision']))
 print('Training recall :',np.mean(cv_pr['train_Recall']))
 print('Training F1-Score :',np.mean(cv_pr['train_F1-Score']))
 print()
 print('\n')
 print('Testing accuracy :',np.mean(cv_pr['test_Accuracy']))
 print('Testing precision :',np.mean(cv_pr['test_Precision']))
 print('Testing recall :',np.mean(cv_pr['test_Recall']))
 print('Testing F1-Score :', np.mean(cv_pr['test_F1-Score']))
# print(cv_pr['F1-Score'])
    '''

    #grids = GridSearchCV(svm.SVC(kernel='linear'), params, cv=10)
    grids = GridSearchCV(svm.SVC(kernel='rbf'), params, cv=10)
    grids.fit(X_train_res, y_train_res)
    print(grids.best_params_)

    gras = GridSearchCV(svm.SVC(kernel='linear'), params, cv=10)
    gras.fit(X_train_res, y_train_res)
    print(gras.best_params_)
'''
 pred_linear = clas_linear.predict(x_test)
 print(accuracy_score(y_test, pred_linear))
 print(precision_score(y_test, pred_linear))
 print(recall_score(y_test, pred_linear))
'''
'''
 ars = []
 com = comment_cleaner('uncanny anime',ars)
 xea = vec.transform(com)
 x_lin = clas_linear.predict(xea)
 print(x_lin)


# print("Training time: %fs; Prediction time: %fs" % (time_train, time_test))


# es = cross_validate(clas_linear,x,train_target,scoring=scores,cv=10)

# print(es)

#test_file = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' +str(2)+' .csv',index_col=0,encoding='cp1252')
#test_file = test_file
 test_file = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(2) + ' .csv',index_col=0,encoding='utf-8-sig')
#test_file = test_file.sample(frac=1).reset_index(drop=True)
 #test_file = test_file[0:00]
 test_file['Comments'] = test_file['Comments'].astype(str)

 # Remove punctuation marks and tokenize each and every word
 test_file['Comments'] = test_file['Comments'].str.replace('[^\w\s]', '')

 pos_test_file = test_file[test_file['Actual Polarity'] == 1]
 neg_test_file = test_file[test_file['Actual Polarity'] == 0]

 train_test = pd.concat([pos_test_file,neg_test_file])
 train_test = train_test.reset_index(drop=True)
# train_test = train_test.sample(frac = 1)



 comtest_array = []

 for i in range(0,int(train_test.shape[0])):
     sen = train_test['Comments'][i]
     comtest_array = comment_cleaner(sen,comtest_array)

 #ela_2 = comtest_array[0:300]
 xe = vec.transform(comtest_array)
# print(xe)
 ye = train_test['Actual Polarity'][0:train_test.shape[0]]
 #print(ye)

 xe_train,xe_test,ye_train,ye_test = train_test_split(xe,ye,test_size=0.3,random_state=1)
# print(xe_test)
# print(ye_test)
 test_svm = svm.SVC(kernel='linear',C=10.0,gamma = 0.001)


# grids = GridSearchCV(svm.SVC(kernel='linear'), params, cv=10)
# grids.fit(x, train_target)
 #print(grids.best_params_)

 test_svm.fit(x,train_target)

#ars = vec.transform(['It having shitty fanservice scenes unfortunately'])
 x_pred = test_svm.predict(xe)
#print(x_pred)
 print(accuracy_score(ye,x_pred))
 print(precision_score(ye,x_pred))
 print(recall_score(ye,x_pred))

 #print(precision_score())


# for i in range(0,train_test.shape[0]):
   #  if (x_pred[i] == 1):
    #   print(train_test['Comment'][i],' : ',x_pred[i])
'''
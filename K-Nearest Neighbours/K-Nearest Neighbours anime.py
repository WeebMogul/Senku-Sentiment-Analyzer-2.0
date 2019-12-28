from sklearn.neighbors import KNeighborsClassifier
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.metrics import make_scorer,precision_score,recall_score,f1_score,accuracy_score
from sklearn.utils import resample,shuffle
import pandas as pd
import string
import matplotlib.pyplot as pypl
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
    # kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' +str(1)+' .csv'

    # df1 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(1) + ' .csv',index_col=0, encoding='utf-8-sig')
    # df2 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(2) + ' .csv',index_col=0, encoding='utf-8-sig')
    # df3 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(23) + ' .csv',index_col=0, encoding='utf-8-sig')
    df1 = pd.read_csv(
        'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 1 .csv',
        index_col=0, encoding='utf-8-sig')
    df2 = pd.read_csv(
        'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 2 .csv',
        index_col=0, encoding='utf-8-sig')
    df3 = pd.read_csv(
        'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 3 .csv',
        index_col=0, encoding='utf-8-sig')
    df4 = pd.read_csv(
        'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 4 .csv',
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


    #neu_df['Comment'] = neu_df['Comment'].
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

    acc_train = []
    acc_test = []
    cas_rep = []
    conu = []

    scores = { 'accuracy' : make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_Score': make_scorer(f1_score)
          }
''''
grid_params = {'n_neighbors':np.arange(1,30),'weights':['distance'],'metric':['manhattan']}
er = GridSearchCV(KNeighborsClassifier(),grid_params,cv=10,verbose=1)
gu = er.fit(x_tr,y_train)

print(gu.best_params_)
print(gu.best_estimator_)
print(gu.best_score_)
'''
for i in range(1,101):
 '''
 cv_knn = cross_validate(KNeighborsClassifier(i),X_train_res, y_train_res, cv=10, return_train_score=True,scoring=scores)
 print('Nearest Neighbour ',str(i))
 print('Training accuracy :', np.mean(cv_knn['train_accuracy']))
 print('Training precision :', np.mean(cv_knn['train_precision']))
 print('Training recall :', np.mean(cv_knn['train_recall']))
 print('Testing accuracy :', np.mean(cv_knn['test_accuracy']))
 print('Testing precision :', np.mean(cv_knn['test_precision']))
 print('Testing recall :', np.mean(cv_knn['test_recall']))
 print('\n')
 '''
 modelKnn = KNeighborsClassifier(n_neighbors=i)
 modelKnn.fit(X_train_res, y_train_res)
 yp = modelKnn.predict(X_train_res)

 pred_linear = modelKnn.predict(x_ts)
 print(i)
 print(accuracy_score(y_train_res, yp))
 print(accuracy_score(y_test, pred_linear))
 print(precision_score(y_test, pred_linear,average=None))
 print(recall_score(y_test, pred_linear,average=None))

'''
 print('Train accuracy neighbours ' + str(i) + ': ' + str(accuracy_score(y_train, yp)))
 print('Train F1 neighbours ' + str(i) + ': ' + str(f1_score(y_train, yp,average=None)))
 print('Test accuracy neighbours ' + str(i)+ ': ' + str(accuracy_score(y_test,y_predict)))
 print('Test F1 neighbours ' + str(i) + ': ' + str(f1_score(y_test, y_predict,average=None)))

# print('For neighbours ' + str(i) + ': ' + str(precision_score(y_test,y_predict)))
# print('For neighbours ' + str(i) + ': ' + str(recall_score(y_test,y_predict)))
 print('\n')

ur = []
ur = comment_cleaner('The isekai genre is very good',ur)
er = vec.transform(ur)
ae = modelKnn.predict(er)
print(ae)

test_file = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(4) + ' .csv',index_col=0,encoding='utf-8-sig')
# test_file = test_file.sample(frac=1).reset_index(drop=True)
test_file['Comments'] = test_file['Comments'].astype(str)

# Remove punctuation marks and tokenize each and every word
test_file['Comments'] = test_file['Comments'].str.replace('[^\w\s]', '')

pos_test_file = test_file[test_file['Actual Polarity'] == 1]
neg_test_file = test_file[test_file['Actual Polarity'] == 0]

# pos_test_file = test_file[test_file['Polarity'] == 1]
# neg_test_file = test_file[test_file['Polarity'] == 0]
neu_test_file = df[(df['Actual Polarity'] == 2) & ((df['Polarity'] == 1) | (df['Polarity'] == 0))]

#print(neg_test_file['Comments'])
train_test = pd.concat([pos_test_file,neg_test_file,neu_test_file])
train_test = train_test.reset_index(drop=True)
# train_test = train_test.sample(frac = 1)

# print(train_test['Polarity'])
# print(train_test.shape[0])

ela = []
for i in range(0,int(train_test.shape[0])):
     sen = train_test['Comments'][i]
     elao = comment_cleaner(sen,ela)
#print(ela)

xe = vec.transform(elao)
# print(xe)
ye = train_test['Actual Polarity'][0:train_test.shape[0]]
 #print(ye)

aac = []
ars = []
res = []
#e_train,xe_test,ye_train,ye_test = train_test_split(xe,ye,test_size=0.1,random_state=0)
# print(xe_test)
# print(ye_test)
n = np.arange(1,50)
for i in range(1,2):
 mk = KNeighborsClassifier(n_neighbors=1)
 mk.fit(x,train_target)
 x_pred = mk.predict(xe)
# print(x_precallisiond)
 aac.append(accuracy_score(ye,x_pred))
 #ars.append(precision_score(ye,x_pred))
# res.append(recall_score(ye,x_pred))
print(aac)
'''
'''
pypl.figure()
pypl.plot(n,aac)
pypl.title('accuracyuracy of each Knn neighbour for Kaguya-Sama episode '+str(ep))
pypl.xlabel('K-Nearest Neighbour')
pypl.ylabel('accuracyuracy rate')
pypl.show()
'''
'''

 for i in range(0,train_test.shape[0]):
     if (x_pred[i] == 0):
      print(train_test['Comments'][i],' : ',x_pred[i])
 #test_df = train_df.sample(frac=1).reset_index(drop=True)[0:5]
 
     comms = ['best girl','The OP is banger','Fuck this']
     labels = [1,1,0]
     data = {'Comment':comms,'Actual Polarity':labels}
     test_df = pd.DataFrame(data)

     elp = []
     for i in range(0, int(test_df.shape[0])):
         sente = test_df['Comment'][i]
         #print(sente)
         train_wordsa = comment_cleaner(sente, elp)

     #print(train_wordsa)
     xe = vec.transform(train_wordsa)
     ye = test_df['Actual Polarity'][0:int(test_df.shape[0])]
     x_precallisiond = modelKnn.precallisiondict(xe)
     print(x_precallisiond)
     print(accuracyuracy_score(x_precallisiond,ye))
     '''
  #cv_knn = cross_validate(modelKnn,x,y,cv=10,scoring=scores)

 #accuracy.append(np.mean(cv_knn['test_accuracyuracy ']))
 #print('accuracyuracy : ',np.mean(cv_knn['test_accuracyuracy ']))

     #precallision.append(np.mean(cv_knn['test_precallisioncision ']))
     #print('precallisioncision : ',np.mean(cv_knn['test_precallisioncision ']))

    #recall.append(np.mean(cv_knn['test_recallall ']))
     #print('recallall : ',np.mean(cv_knn['test_recallall ']))

    #  f1_score.append(np.mean(cv_knn['test_f1_score-Score ']))
     #print('f1_score Measure: ',np.mean(cv_knn['test_f1_score-Score ']))

    #  error.append(1 - accuracy[i])
    # print('erroror rate',1- accuracy[i])
    # print("\n")










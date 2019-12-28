from sklearn.neighbors import KNeighborsClassifier
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.utils import resample,shuffle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score,f1_score, accuracy_score,classification_report
import pandas as pd
import string
import numpy as np
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as pypl

nltk.download("popular")
stopword = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# boogiepop_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Boogiepop cleaned\Cleaned_Boogiepop_Wa_Waranai_Episode_' + str(ep) + '_Comment_list.csv'
# kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Kaguya-sama cleaned\Cleaned_Kaguya_sama_Episode_' + str(1) + '_Comment_list.csv'
# slime_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Tensei Slime cleaned\Cleaned_Tensei_Slime_Episode_' + str(ep) + '_Comment_list.csv'


def penntag(pen):
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[pen[:2]]
    except:
        return 'n'



def comment_cleaner(comm, comment_array):
    temp_comm = []
    stopwords_removed = [word for word in comm.lower().split() if word not in stopword]
    POS_words = nltk.pos_tag(stopwords_removed)
    for i in range(0, len(POS_words)):
        lemmas = lemma.lemmatize(POS_words[i][0], pos=penntag(POS_words[i][1]))
        temp_comm.append(lemmas)
    #print(temp_comm)
    megos = ' '.join(word for word in temp_comm)
   # print(megos)
    comment_array.append(temp_comm)
    return comment_array
    temp_comm.clear()

for ep in range(1, 2):

    # kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' +str(1)+' .csv'
    kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(
        1) + ' .csv'

    df = pd.read_csv(kaguya_file, index_col=0, encoding='utf-8-sig')

    #df = df.sample(frac=1)
    # Convert dataframe values into string
    df['Comments'] = df['Comments'].astype(str)

    # Remove punctuation marks and tokenize each and every word
    df['Comments'] = df['Comments'].str.replace('[^\w\s]', ' ')

    # Split into positive and negative datasets
    pos_df = df[df['Actual Polarity'] == 1]
    neg_df = df[df['Actual Polarity'] == 0]
    neu_df = df[df['Actual Polarity'] == 2]

    neg_upsample = resample(neg_df, replace=True, n_samples=len(pos_df),random_state=0)
    neu_upsample = resample(neu_df,replace=True,n_samples=len(pos_df))

    # Concatenate them into one
    train_df = pd.concat([pos_df, neg_df])
    train_df = train_df.reset_index(drop=True)

    comment_array = []
    train_target = []
    comtest_array = []

    for i in range(0, int(train_df.shape[0])):
        sentences = train_df['Comments'][i]
        train_words = comment_cleaner(sentences, comment_array)

    def return_back_df(doc):
        return doc


    vec = TfidfVectorizer(analyzer='word', preprocessor=return_back_df, tokenizer=return_back_df,ngram_range=(1,2),use_idf=True,norm='l2')
    x = vec.fit_transform(train_words)
    train_target = train_df['Actual Polarity'][0:int(train_df.shape[0])]
    '''
    res = x.todense()
    ges = res.tolist()
    vo = vec.get_feature_names()

    tval = pd.DataFrame(ges,columns=vo)
    sum = tval.sum(numeric_only =True)

    sum.sort_values(ascending=False)
    normalized = sum / sum.max()
    print(normalized.sort_values(ascending=False))
    '''
    x,train_target = shuffle(x,train_target)
    x_train, x_test, y_train, y_test = train_test_split(x, train_target, test_size=0.3,random_state=1,stratify=train_target)
    #Naive Bayes test
    scores = {'Accuracy': make_scorer(accuracy_score),
              'Precision': make_scorer(precision_score),
              'Recall': make_scorer(recall_score),
              'F1-Score': make_scorer(f1_score)
              }
    NB = MultinomialNB()
    #NB = BernoulliNB()

    #NB = GaussianNB()
    NB.fit(x_train,y_train)

    pred_linear = NB.predict(x_test)
    print(accuracy_score(y_test, pred_linear))
    print(precision_score(y_test, pred_linear))
    print(recall_score(y_test, pred_linear))


    '''
    xu = cross_validate(NB, x, train_target, cv=10, scoring=scores,return_train_score=True)
    print('Training accuracy :', np.mean(xu['train_Accuracy']))
    print('Training precision :', np.mean(xu['train_Precision']))
    print('Training recall :', np.mean(xu['train_Recall']))
    print('Training F1-Score :', np.mean(xu['train_F1-Score']))
    print('\n')
    print('Testing accuracy :', np.mean(xu['test_Accuracy']))
    print('Testing precision :', np.mean(xu['test_Precision']))
    print('Testing recall :', np.mean(xu['test_Recall']))
    print('Testing F1-Score :', np.mean(xu['test_F1-Score']))
    #NB.fit(x,train_target)
    y_pred = NB.predict(x_test)

    #print(accuracy_score(y_test,y_pred))
   # print(precision_score(y_test,y_pred))
    #print(recall_score(y_test,y_pred))
    '''

    test_file = pd.read_csv(
        'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(
            2) + ' .csv', index_col=0, encoding='utf-8-sig')
    # test_file = test_file.sample(frac=1).reset_index(drop=True)
    # test_file = test_file[0:00]
    test_file['Comments'] = test_file['Comments'].astype(str)

    # Remove punctuation marks and tokenize each and every word
    test_file['Comments'] = test_file['Comments'].str.replace('[^\w\s]', '')

    pos_test_file = test_file[test_file['Actual Polarity'] == 1]
    neg_test_file = test_file[test_file['Actual Polarity'] == 0]
    neu_test_file = test_file[test_file['Actual Polarity'] == 2]

    train_test = pd.concat([pos_test_file, neg_test_file])
    train_test = train_test.reset_index(drop=True)
    # train_test = train_test.sample(frac = 1)

    comtest_array = []

    for i in range(0, int(train_test.shape[0])):
        sen = train_test['Comments'][i]
        comtest_array = comment_cleaner(sen, comtest_array)

    # ela_2 = comtest_array[0:300]
    veca = TfidfVectorizer(analyzer='word', preprocessor=return_back_df, tokenizer=return_back_df, ngram_range=(1, 2),
                          use_idf=True, norm='l2')
    xe = vec.transform(comtest_array)
    # print(xe)
    ye = train_test['Actual Polarity'][0:train_test.shape[0]]
    # print(ye)
    '''
    res = x.todense()
    ges = res.tolist()
    vo = vec.get_feature_names()

    tval = pd.DataFrame(ges, columns=vo)
    sum = tval.sum(numeric_only=True)

    sum.sort_values(ascending=False)
    normalized = sum / sum.max()
    print(normalized.sort_values(ascending=False))
    '''
    # xe_train, xe_test, ye_train, ye_test = train_test_split(xe, ye, test_size=0.2, random_state=0, stratify=ye)
    # print(xe_test)
    # print(ye_test)

    nb = MultinomialNB()
    nb.fit(x,train_target)
    x_pred = nb.predict(xe)
    #xes = cross_validate(NB,xe,ye,cv=10,return_train_score=True)
    #print(xes)
    print(accuracy_score(ye, x_pred))
    print(precision_score(ye, x_pred))
    print(recall_score(ye, x_pred))
    '''
    da = {'Comments': train_test['Comments'],'Actual Polarity':x_pred}
    xool = pd.DataFrame(da)
    print(xool.head())
    #print(classification_report(ye,x_pred,target_names=['Negative','Positive','Neutral']))
    '''
    '''
    er = []
    ur = comment_cleaner('uncanny isekai',er)
    ar = vec.transform(ur)
    s = NB.predict(ar)
    print(s)
    '''

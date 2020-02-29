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
#from IPython.display import display, HTML
#Other
import pandas as pd
import re
import string
import matplotlib.pyplot as pypl
import numpy as np
import seaborn as sd
import imblearn
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

stopword = set(stopwords.words('english'))
stopword.update(('know','really','say','way','thing','need','look','want','actually','use', 'think', 'would',
                 'use','muda','dr','make','go','get','it','even','also','already','much','could','that','one','though','still'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

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
    df1 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 1 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df2 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 2 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df3 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 3 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df4 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 4 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df5 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 5 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df6 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 6 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df7 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 7 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df8 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 8 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df9 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 9 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df10 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 10 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df11 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 11 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df12 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 12 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df13 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 13 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df14 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 14 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df15 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 15 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df16 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 16 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df17 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 17 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df18 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 18 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df19 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 19 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df20 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 20 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df21 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 21 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df22 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 22 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df23 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 23 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')
    df24 = pd.read_csv(
        'D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode 24 Comment list with Sentiment rating.csv',
        index_col=0, encoding='utf-8-sig')

    df123 = pd.concat(
        [df1, df2, df3, df4, df5, df6,
         df7, df8, df9, df10, df11, df12,
         df13,df14,df15,df16,df17,df18,
         df19,df20,df21,df22,df23,df24
         ])
    train_array = []
    test_array = []
    train_target = []
    comtest_array = []
    # df = df.sample(frac=1)
    # Convert dataframe values into string
    df123 = df123[['Comment', 'Sentiment Rating']]
    df123['Comment'] = df123['Comment'].astype(str)
    df123['Length'] = df123['Comment'].apply(len)
    df123 = df123[df123['Length'] > 5]
    df123['Comment'] = df123['Comment'].apply(lambda s: comment_cleaner(s, train_array))

    # Remove punctuation marks and tokenize each and every word
    df123['Comment'] = df123['Comment'].str.replace('[^\w\s]', ' ')
    df123['Comment'] = df123['Comment'].str.replace('[\d+]', ' ')
    df123['Comment'] = df123['Comment'].str.replace('(^| ).(( ).)*( |$)', ' ')

    # Split into positive and negative datasets
    pos_df = df123[df123['Sentiment Rating'] == 1]
    neg_df = df123[df123['Sentiment Rating'] == 0]
    neu_df = df123[df123['Sentiment Rating'] == 2]

    # neu_df['Comment'] = neu_df['Comment'].
    df_len = len(pos_df)

    train_df = pd.concat([pos_df, neg_df,neu_df])
    # train_df = pd.concat([pos_df, neg_df,neu_df])
    train_df = train_df.reset_index(drop=True)

    x = train_df['Comment'].values
    y = train_df['Sentiment Rating'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2, random_state=22)

    vec = TfidfVectorizer(ngram_range=(1, 2),min_df=0.01,max_df=0.8,analyzer='word',sublinear_tf=True,max_features=60000)
    # vec = CountVectorizer(ngram_range=(1, 2))
    x_tr = vec.fit_transform(x_train)
    x_ts = vec.transform(x_test)

    sm = RandomOverSampler(random_state=22)

    X_train_res, y_train_res = sm.fit_sample(x_tr, y_train)

    #knn_pipeline = Pipeline([('Knn',KNeighborsClassifier(n_neighbors=3)))

    NB = MultinomialNB()
    LSVM = svm.SVC(C=1.0,gamma=0.01,kernel='linear')
    modelKnn = KNeighborsClassifier(n_neighbors=25)

    classi = [modelKnn,LSVM,NB]
    steps = [('classifiers', classi)]
    classifier_names = ['K Nearest Neighbours', 'Linear SVM', 'Multinomial Naive Bayes']

    acc_train = []
    acc_test = []
    cas_rep = []
    conu = []

    for classif in classi:
        pipi = Pipeline([('classifiers', classif)])
        pipi.fit(X_train_res, y_train_res)
        pri = pipi.predict(X_train_res)
        pred = pipi.predict(x_ts)
        acc_train.append(accuracy_score(y_train_res, pri))
        acc_test.append(accuracy_score(y_test, pred))
        # print(accuracy_score(y_test,pred))
        cas_rep.append(classification_report(y_test, pred))
        # print(classification_report(y_test, pred))

        conu.append(confusion_matrix(y_test, pred))
    # print(conf)

    for i in range(0, 3):
        #  print('Train Accuracy of ',classifier_names[i],' : ',acc_train[i])
        print('Train Accuracy of ', classifier_names[i], ' : ', acc_train[i])
        print('Test Accuracy of ', classifier_names[i], ' : ', acc_test[i])
        # de = pd.DataFrame(confu[i],index = ['Negative','Positive','Neutral'],columns = ['Negative','Positive','Neutral'])
        de = pd.DataFrame(conu[i], index=['Negative', 'Positive','Neutral'], columns=['Negative', 'Positive','Neutral'])
        pypl.figure()
        title = 'Confusion Matrix of ' + classifier_names[i]
        print('\nClassification report of ', classifier_names[i], ' : \n')
        print(conu[i])
        print(cas_rep[i])
        pypl.title(title)
        sd.heatmap(de, annot=True, cmap='Blues', fmt='g')
        pypl.show()
        print('\n\n\n')
        de.iloc[0:0]

        # test_file = pd.read_csv(
        #     'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(
        #         4) + ' .csv', index_col=0, encoding='utf-8-sig')
        # test_file = test_file.sample(frac=1).reset_index(drop=True)
        # test_file = test_file[0:00]

        # test_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(2) + ' .csv'
'''
    test_file = pd.read_csv(
        'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 5 .csv',
        index_col=0, encoding='utf-8-sig')
    test_file['Comment'] = test_file['Comment'].astype(str)

    # Remove punctuation marks and tokenize each and every word
    test_file['Comment'] = test_file['Comment'].str.replace('[^\w\s]', '')
    df12 = df12[['Comment', 'Actual Polarity']]

    pos_test_file = test_file[test_file['Actual Polarity'] == 1]
    neg_test_file = test_file[test_file['Actual Polarity'] == 0]
    neu_test_file = test_file[test_file['Actual Polarity'] == 2]

    # train_test = pd.concat([pos_test_file, neg_test_file,neu_test_file])
    train_test = pd.concat([pos_test_file, neg_test_file])
    train_test = train_test.reset_index(drop=True)
    train_test = train_test[['Comment', 'Actual Polarity']]
    # train_test = train_test.sample(frac = 1)




   
    for i in range(0, int(train_test.shape[0])):
        sen = train_test['Comment'][i]
        comtest_array = comment_cleaner(sen, comtest_array)
 
    # ela_2 = comtest_array[0:300]
    # veca = TfidfVectorizer(analyzer='word', preprocessor=return_back_df, tokenizer=return_back_df, ngram_range=(1, 2),sublinear_tf=True)
    # x = vec.transform(sentences)
    com=[]
    te = train_test['Comment'].apply(lambda s: comment_cleaner(s, com))
    xe = vec.transform(te)
    # print(xe)

    ye = train_test['Actual Polarity']

   # classi = [modelKnn, SVM, NB]
    steps = [('classifiers', classi)]
    acca_test = []
    clas_rep = []
    confu = []
    acc_train = []

    for classif in classi:
        pipi = Pipeline([('classifiers', classif)])
        pipi.fit(X_train_res, y_train_res)
        pred = pipi.predict(xe)
        arcs = (accuracy_score(ye, pred))
        acca_test.append(arcs)
        ue = classification_report(ye, pred)
        clas_rep.append(ue)
        conf = confusion_matrix(ye, pred)
        confu.append(conf)

    classifier_names = ['K Nearest Neighbours', 'Linear SVM','Multinomial Naive Bayes']

    for i in range(0, 4):
        #  print('Train Accuracy of ',classifier_names[i],' : ',acc_train[i])
        print('Test Accuracy of ', classifier_names[i], ' : ', acc_test[i])
        # de = pd.DataFrame(confu[i],index = ['Negative','Positive','Neutral'],columns = ['Negative','Positive','Neutral'])
        de = pd.DataFrame(confu[i], index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
        pypl.figure()
        title = 'Confusion Matrix of ' + classifier_names[i]
        print('\nClassification report of ', classifier_names[i], ' : \n')
        print(clas_rep[i])
        pypl.title(title)
        sd.heatmap(de, annot=True, cmap='Blues', fmt='g')
        pypl.show()
        print('\n\n\n')
        de.iloc[0:0]

'''

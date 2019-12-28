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
    stopwords_removed = [word for word in comm.lower().split() if word not in stopword]
    POS_words = nltk.pos_tag(stopwords_removed)
    for i in range(0, len(POS_words)):
        lemmas = lemma.lemmatize(POS_words[i][0], pos=penntag(POS_words[i][1]))
        temp_comm.append(lemmas)
    # print(temp_comm)
    megos = ' '.join(word for word in temp_comm)
    comment_array.append(temp_comm)
    return temp_comm
    # comment_array.clear()


for ep in range(1, 2):
    # kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' +str(1)+' .csv'

    # df1 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(1) + ' .csv',index_col=0, encoding='utf-8-sig')
    # df2 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(2) + ' .csv',index_col=0, encoding='utf-8-sig')
    # df3 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(23) + ' .csv',index_col=0, encoding='utf-8-sig')
    df1 = pd.read_csv(
        r'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis-master\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 1 .csv',
        index_col=0, encoding='utf-8-sig')
    df2 = pd.read_csv(
        r'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis-master\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 2 .csv',
        index_col=0, encoding='utf-8-sig')
    df3 = pd.read_csv(
        r'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis-master\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 3 .csv',
        index_col=0, encoding='utf-8-sig')
    df4 = pd.read_csv(
        r'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis-master\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 4 .csv',
        index_col=0, encoding='utf-8-sig')
    df5 = pd.read_csv(
        r'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis-master\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 5 .csv',
        index_col=0, encoding='utf-8-sig')
    df6 = pd.read_csv(
        r'D:\Heriot-Watt-Msc-Project-Sentiment-Analysis-master\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 6 .csv',
        index_col=0, encoding='utf-8-sig')
    # df4 = pd.read_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode 4 .csv',index_col=0, encoding='utf-8-sig')


    df12 = pd.concat([df1,df2,df3
                      ])

    df12['Total words'] = df12['Comments'].str.split().str.len()

    print(df12[df12['Total words'] < 6].count())

    # df = df.sample(frac=1)
    # Convert dataframe values into string
    df12 = df12[['Comments', 'Actual Polarity']]
    df12['Comments'] = df12['Comments'].astype(str)
    print(df12.shape)
    print(df12.shape)
    # Remove punctuation marks and tokenize each and every word
    df12['Comments'] = df12['Comments'].str.replace('[^\w\s]', '')
    df12['Length'] = df12['Comments'].str.lower().str.split().apply(len)
   # df12 = df12[df12['Length'] < 100]
    # Split into positive and negative datasets
    pos_df = df12[df12['Actual Polarity'] == 1]
    neg_df = df12[df12['Actual Polarity'] == 0]
    df_len = len(pos_df)

    # neg_upsample = resample(neg_df, replace=False, n_samples=df_len)

    # Concatenate them into one
    # train_df = pd.concat([pos_df, neg_upsample,neu_upsample])
    train_df = pd.concat([pos_df, neg_df])
    train_df = train_df.reset_index(drop=True)

   # train_df['Comment'], train_df['Actual Polarity'] = shuffle(train_df['Comment'], train_df['Actual Polarity'])
    train_array = []
    test_array = []
    train_target = []
    comtest_array = []
    y = train_df['Actual Polarity']

    x_train, x_test, y_train, y_test = train_test_split(train_df['Comments'], train_df['Actual Polarity'], test_size=0.2,random_state=22)

    def return_back_df(doc):
        return doc


    x_train = x_train.apply(lambda s: comment_cleaner(s, train_array))
    print(x_train)
    x_test = x_test.apply(lambda s: comment_cleaner(s, test_array))
    # wer = pd.DataFrame({'Comment':X})

    # display(train_words)

    '''
    for i in range(0, int(X.shape[0])):
        train_words = x_train[i]
        sen = comment_cleaner(train_words, train_array)
   # wer = pd.DataFrame({'Comment':sentences})

    '''
    vec = TfidfVectorizer(analyzer='word', preprocessor=return_back_df, tokenizer=return_back_df, ngram_range=(1, 2),
                          max_features=2000,sublinear_tf=True,max_df=0.7)
    x_tr = vec.fit_transform(x_train)
    x_ts = vec.transform(x_test)

    sm = RandomOverSampler(random_state=77)

    X_train_res, y_train_res = sm.fit_sample(x_tr, y_train)

    #knn_pipeline = Pipeline([('Knn',KNeighborsClassifier(n_neighbors=3)))

    NB = MultinomialNB()
    LSVM = svm.SVC(C=1.0,gamma=0.0001,kernel='linear')
    RSVM = svm.SVC(C=100.0,gamma=0.0001,kernel='rbf')
    modelKnn = KNeighborsClassifier(n_neighbors=3)

   # steps = [('classifiers', classi)]
   # classifier_names = ['K Nearest Neighbours', 'Linear SVM', 'Multinomial Naive Bayes']
   # nb = MultinomialNB()
   # for i in range(1,100):
    #nb = KNeighborsClassifier(n_neighbors=3)
    #nb = svm.SVC(C=100.0,gamma=0.001)

   # nb.fit(X_train_res, y_train_res)

    #print(nb.score(X_train_res, y_train_res))

   # y_pred = nb.predict(x_te)
    #print(accuracy_score(y_test, y_pred))
  #  print(confusion_matrix(y_test, y_pred))

    #modelKnn = KNeighborsClassifier(n_neighbors=3)
    #SVM = svm.SVC(kernel='linear', C=10.0, gamma=0.0001)
   # NB = MultinomialNB()

    classi = [modelKnn,LSVM,RSVM,NB]
    steps = [('classifiers', classi)]
    classifier_names = ['K Nearest Neighbours', 'Linear SVM','Radial Basis Function SVM', 'Multinomial Naive Bayes']

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

    for i in range(0, 4):
        #  print('Train Accuracy of ',classifier_names[i],' : ',acc_train[i])
        print('Train Accuracy of ', classifier_names[i], ' : ', acc_train[i])
        print('Test Accuracy of ', classifier_names[i], ' : ', acc_test[i])
        # de = pd.DataFrame(confu[i],index = ['Negative','Positive','Neutral'],columns = ['Negative','Positive','Neutral'])
        de = pd.DataFrame(conu[i], index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
        pypl.figure()
        title = 'Confusion Matrix of ' + classifier_names[i]
        print('\nClassification report of ', classifier_names[i], ' : \n')
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



    '''
    '''
    for i in range(0, int(train_test.shape[0])):
        sen = train_test['Comment'][i]
        comtest_array = comment_cleaner(sen, comtest_array)
    '''
    '''
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

    classifier_names = ['K Nearest Neighbours', 'Linear SVM', 'Radial Basis Function','Multinomial Naive Bayes']

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



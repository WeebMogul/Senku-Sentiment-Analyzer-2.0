#Sklearn

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
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



    df12 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12
                      ])
    train_array = []
    test_array = []
    train_target = []
    comtest_array = []
    # df = df.sample(frac=1)
    # Convert dataframe values into string
    df12 = df12[['Comment', 'Sentiment Rating']]
    df12['Comment'] = df12['Comment'].astype(str)
    df12['Length'] = df12['Comment'].apply(len)
    df12 = df12[df12['Length'] > 5]
    df12['Comment'] = df12['Comment'].apply(lambda s: comment_cleaner(s, train_array))

    # Remove punctuation marks and tokenize each and every word
    df12['Comment'] = df12['Comment'].str.replace('[^\w\s]', ' ')
    df12['Comment'] = df12['Comment'].str.replace('[\d+]', ' ')
    df12['Comment'] = df12['Comment'].str.replace('(^| ).(( ).)*( |$)', ' ')

    # Split into positive and negative datasets
    pos_df = df12[df12['Sentiment Rating'] == 1]
    neg_df = df12[df12['Sentiment Rating'] == 0]
    neu_df = df12[df12['Sentiment Rating'] == 2]

    # neu_df['Comment'] = neu_df['Comment'].
    df_len = len(pos_df)

    train_df = pd.concat([pos_df, neg_df])
    # train_df = pd.concat([pos_df, neg_df,neu_df])
    train_df = train_df.reset_index(drop=True)

    y = train_df['Sentiment Rating']

    x_train, x_test, y_train, y_test = train_test_split(train_df['Comment'], train_df['Sentiment Rating'],
                                                        test_size=0.2, random_state=22)

    vec = TfidfVectorizer(ngram_range=(1, 3),sublinear_tf=True)
    # vec = CountVectorizer(ngram_range=(1, 2))
    x_tr = vec.fit_transform(x_train)
    x_ts = vec.transform(x_test)

    sm = RandomOverSampler(random_state=22)

    X_train_res, y_train_res = sm.fit_sample(x_tr, y_train)

    scores = { 'Accuracy' : make_scorer(accuracy_score),
           'Precision': make_scorer(precision_score),
           'Recall': make_scorer(recall_score),
          'F1-Score': make_scorer(f1_score)
          }


# Train SVM model
    C = [0.01, 0.1, 1.0, 10.0,100.0]
    gammas = [0.01, 0.1, 1.0, 10.0,100.0]
    #iter = [1,10,50,100,500,1000]
    params = {'C': C,'gamma':gammas}

    clas_linear = svm.SVC(kernel='linear',C=10.0,gamma=0.01)
    clas_linear.fit(X_train_res,y_train_res)
    train_linear = clas_linear.predict(X_train_res)
    pred_linear = clas_linear.predict(x_ts)
    print(accuracy_score(y_train_res,train_linear))
    print(accuracy_score(y_test, pred_linear))
    print(precision_score(y_test, pred_linear))
    print(recall_score(y_test, pred_linear))
    print(confusion_matrix(y_test,pred_linear))
    '''
    grid = GridSearchCV(svm.SVC(kernel='linear'),cv=10,param_grid=params, refit = True)
    grid.fit(X_train_res, y_train_res)
    print(grid.best_params_)
    #print(grid.best_estimator_)
    '''




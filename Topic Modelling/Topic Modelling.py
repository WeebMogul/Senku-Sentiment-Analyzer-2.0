from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample,shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,cross_val_score
from sklearn.metrics import make_scorer,precision_score,recall_score,accuracy_score,f1_score
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import string
import matplotlib.pyplot as pypl

stopword = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def comment_cleaner(comm, comment_array):
    temp_comm = []
    stopwords_removed = [word for word in comm.lower().split() if word not in stopword]
    '''
    POS_words = nltk.pos_tag(stopwords_removed)
    for i in range(0, len(POS_words)):
        lemmas = lemma.lemmatize(POS_words[i][0], pos=penntag(POS_words[i][1]))
        temp_comm.append(lemmas)
    '''
    megos = ' '.join(word for word in stopwords_removed)
    comment_array.append(megos)
    return comment_array
    stopwords_removed.clear()
    comment_array.clear()


def penntag(pen):
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[pen[:2]]
    except:
        return 'n'

for ep in range(1,2):
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' +str(1)+' .csv'
 kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Tensei Slime cleaned\Tensei Slime Episode ' + str(
     19) + ' .csv'
 df1 = pd.read_csv(kaguya_file, index_col=0,encoding='utf-8-sig')

 #kaguya_file2 = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Kaguya-sama cleaned\Kaguya-sama Episode ' + str(2) + ' .csv'
 #df2 = pd.read_csv(kaguya_file2,index_col=0,encoding='cp1252')

 #df1 = df1[0:df1.shape[0]]
# Convert dataframe values into string
 df1['Comments'] = df1['Comments'].astype(str)
 #df2['Comment'] = df2['Comment'].astype(str)

# Remove punctuation marks and tokenize each and every word
 df1['Comments'] = df1['Comments'].str.replace('[^\w\s]', '')
 #df2['Comment'] = df2['Comment'].str.replace('[^\w\s]', ' ')

 #df12 = pd.concat([df1,df2]).reset_index(drop=True)
# Create datasets with positive and negative polarity

 pos_df = df1[df1['Actual Polarity'] == 1]
 print(pos_df.shape[0])
 neg_df = df1[df1['Actual Polarity'] == 0]
 print(neg_df.shape[0])
# Concatenate positive and negative datasets


 neg_resample = resample(neg_df,replace=True,n_samples=len(pos_df),random_state=0)

 train_df = pd.concat([neg_df])
 train_df = train_df.reset_index(drop=True)


 print(train_df.shape[0])

 comment_array = []
 train_target = []


# Clean comments

 for i in range(0, int(train_df.shape[0])):
     sentences = train_df['Comments'][i]
     train_words = comment_cleaner(sentences, comment_array)

 def return_back_df(doc):
     return doc

# perfom TF-IDF on the tokens
 tomol = CountVectorizer(stop_words='english')
 tf = tomol.fit_transform(train_words)
 tf_feature = tomol.get_feature_names()

 lda = LatentDirichletAllocation(n_components=20).fit(tf)

 topi = []
# def topics()
 def print_topics(model, vectorizer, top_n=10):
     for idx, topic in enumerate(model.components_):
         print("Topic : " ,str(idx))
         for i in topic.argsort()[:-top_n- 1:-1]:
           topi.append((vectorizer.get_feature_names()[i]))
         print(topi)
         topi.clear()
          # print([(vectorizer.get_feature_names()[i])])


 print("LDA Model:")
 print_topics(lda, tomol)
 print("=" * 20)


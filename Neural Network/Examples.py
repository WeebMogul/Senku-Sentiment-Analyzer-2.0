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
from yellowbrick.text import FreqDistVisualizer

comments = []
col = []

def penntag(pen):
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[pen[:2]]
    except:
        return 'n'


stopword = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def comment_cleaner(comm, comment_array):
    temp_comm = []
    megos = ' '
    stopwords_removed = [word for word in comm.lower().split() if word not in stopword]
    POS_words = nltk.pos_tag(stopwords_removed)
    for i in range(0, len(POS_words)):
        lemmas = lemma.lemmatize(POS_words[i][0], pos=penntag(POS_words[i][1]))
        temp_comm.append(lemmas)
    megos = ' '.join(word for word in temp_comm)
    return megos

df = pd.read_csv('D:/Python/Machine Learning/Python Movie Dataset/movie_data.csv')

train_array = []
test_array = []
train_target = []
comtest_array = []


def further_comment_clean(df):
    df123 = df
    #df123['review'] = df123['review'].astype(str)
    df123 = df123[df123['sentiment'] == 1]
    df123['Review'] = df123['Review'].apply(lambda s: comment_cleaner(s, train_array))
    df123['Review'] = df123['Review'].str.replace('[^\w\s]', ' ')
    df123['Review'] = df123['Review'].str.replace('[\d+]', ' ')
    df123['Review'] = df123['Review'].str.replace('(^| ).(( ).)*( |$)', ' ')
    return df123

dfx = further_comment_clean(df)
print(dfx['Review'])

tomol = CountVectorizer(stop_words='english')
tf = tomol.fit_transform(dfx['Review'])
tf_feature = tomol.get_feature_names()

lda = LatentDirichletAllocation(n_components=20).fit(tf)

topi = []
# def topics()

def print_topics(model, vectorizer, top_n=10):
     for idx, topic in enumerate(model.components_):
         print("\nTopic : " ,str(idx),"\n")
         for i in topic.argsort()[:-top_n- 1:-1]:
           topi.append((vectorizer.get_feature_names()[i]))
         for i in range(0,10):
             print(topi[i])
         topi.clear()
          # print([(vectorizer.get_feature_names()[i])])


print("LDA Model:")
print_topics(lda, tomol)
print("=" * 20)

'''
totals = df123['Comment']

cou = CountVectorizer(stop_words='english',ngram_range=(2,2)).fit(totals)
#cou = TfidfVectorizer(stop_words='english',ngram_range=(1,2)).fit(totals)
bags = cou.transform(totals)
sum_words = bags.sum(axis=0)

word_freq = [(word,sum_words[0,idx]) for word,idx in cou.vocabulary_.items()]
word_freq = sorted(word_freq,key=lambda x: x[1],reverse=True)

words = []
count = []

for i in range(1,len(word_freq)):
    words.insert(i,word_freq[i][0])
    count.insert(i,word_freq[i][1])

print(word_freq)

y = np.arange(1,40)
pypl.title('Most frequent positive words in Dr. Stone')
pypl.barh(y,count[:39])
pypl.yticks(y,words[:39])
pypl.xlabel('Word Count')
pypl.ylabel('Words')
pypl.gca().invert_yaxis()
pypl.show()
'''











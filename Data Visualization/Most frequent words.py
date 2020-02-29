import matplotlib.pyplot as pypl
import pandas as pd
import numpy as np 
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
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
stopword.update(('know','really','say','way','thing','need','look','want','like','actually','use', 'think', 'would',
                 'use','muda','dr','make','go','get','it','even','also','already','much','could','that','one','though',
                 'still','thousand'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

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
    [df1, df2, df3, df4,
     #df5, df6, df7, df8, df9, df10, df11, df12, df13, df14,
     #df15, df16, df17,
     #df18, df19,df20, df21, df22, df23, df24
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
df123 = df123[df123['Sentiment Rating'] == 0]
   # df123 = df123[df123['Length'] > 5]
df123['Comment'] = df123['Comment'].apply(lambda s: comment_cleaner(s, train_array))

    # Remove punctuation marks and tokenize each and every word
df123['Comment'] = df123['Comment'].str.replace('[^\w\s]', ' ')
df123['Comment'] = df123['Comment'].str.replace('[\d+]', ' ')
df123['Comment'] = df123['Comment'].str.replace('(^| ).(( ).)*( |$)', ' ')


totals = df123['Comment']

cou = CountVectorizer(stop_words=['know','really','say','way','thing','need','look','want','like','actually','use', 'think', 'would',
                 'use','muda','dr','make','go','get','it','even','also','already','much','great','make','sense','could','that','one','though','still','you','re'],ngram_range=(2,2)).fit(totals)
bags = cou.transform(totals)
sum_words = bags.sum(axis=0)

word_freq = [(word,sum_words[0,idx]) for word,idx in cou.vocabulary_.items()]
word_freq = sorted(word_freq,key=lambda x: x[1],reverse=True)


words = []
count = []

number_of_words = 20

word_freq = [w for w in word_freq if w[0] != 'dr stone']
print(word_freq[:20])

for i in range(0,len(word_freq)):
    words.insert(i,word_freq[i][0])
    count.insert(i,word_freq[i][1])

y = np.arange(1,number_of_words)
pypl.title('Most frequent negative words in Dr. Stone episode 1-4')
#pypl.title('Most frequent positive words in Dr. Stone episode 18-24')
#pypl.title('Most frequent neutral words in Dr. Stone episode 18-24')
pypl.barh(y,count[:number_of_words-1])
pypl.yticks(y,words[:number_of_words-1])
pypl.xlabel('Word Count')
pypl.ylabel('Words')
pypl.gca().invert_yaxis()
pypl.show()












from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as pypl
import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score,f1_score, accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV,train_test_split
comments = []
rating = []

pos_value = 0
neg_value = 0
neu_value = 0
pos_count = 0

anime_dict = \
{
    'kawaii': 0.8,
    'cringe': -0.2,
    'god tier': 0.8,
    'favourite': 0.8,
    'cracked me up': 0.7,
    'hype': 0.7
}
DrStone_max_ep = 6
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
df123 = df123[df123['Sentiment Rating'] < 3]
df123['Comment'] = df123['Comment'].astype(str)
x_train, x_test, y_train, y_test = train_test_split(df123['Comment'].values, df123['Sentiment Rating'].values,random_state=22)
mean = []
DrStone_max_ep = 2
for ia in range(1, DrStone_max_ep):
    #stone_file = 'D:\Python\Senku Sentiment Analyzer 2.0\Data Cleaner\Dr.Stone Episode ' + str(ia) + ' Comment list (cleaned).csv'
    str_comments = x_test
  #  str_comments = dfa['Comment'].astype(str)
    str_length = len(x_test)

    analyzer = SentimentIntensityAnalyzer()
   # analyzer.lexicon.update(anime_dict)

    for i in range(0, int(str_length)):

       vs = analyzer.polarity_scores(str_comments[i])

       if vs['compound'] > 0.05:
        pos_value += 1
        comments.insert(i-1, str_comments[i])
        rating.insert(i-1, 1)

       if vs['compound'] < -0.05:
        neg_value += 1
        comments.insert(i-1, str_comments[i])
        rating.insert(i-1, 0)

       if (vs['compound'] < 0.05) & (vs['compound'] > - 0.05):
        neu_value += 1
        comments.insert(i-1, str_comments[i])
        rating.insert(i-1, 2)

       pos_count += 1

    #actu = verf_file['Sentiment Rating']
    data = {'Comment': x_test, 'Sentiment Rating': rating,'Actual rating':y_test}
    df = pd.DataFrame(data)
    df.reset_index(drop=True)

for i in range(1, DrStone_max_ep):

    total = len(comments)
    dfx = df[df['Sentiment Rating'] == df['Actual rating']].count() ['Sentiment Rating']
    dfneg = df[(df['Sentiment Rating'] == 0) & (df['Actual rating'] == 0)].count()['Sentiment Rating']
    dfpos = df[(df['Sentiment Rating'] == 1) & (df['Actual rating'] == 1)].count()['Sentiment Rating']
    dfneu = df[(df['Sentiment Rating'] == 2) & (df['Actual rating'] == 2)].count()['Sentiment Rating']
    print('Correct negative : ',dfneg)
    print('Correct positive :',dfpos)
    print('Correct neutral :',dfneu)
    print(df[df['Actual rating'] == 0].count()['Actual rating'])
    print(df[df['Actual rating'] == 1].count()['Actual rating'])
    print(df[df['Actual rating'] == 2].count()['Actual rating'])
    print((dfx/total)*100)
    #mean.append((dfx/total)*100)
   # comments.clear()
   # rating.clear()

   # comments.clear()
   # rating.clear()

'''
    data = {'Comment': comments, 'Sentiment Rating': rating}
    df = pd.DataFrame(data)
    df.reset_index(drop=True)

    df.to_csv('D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode ' + str(ia) + ' Comment list with Sentiment rating.csv', encoding='utf-8-sig')

    comments.clear()
    rating.clear()
'''
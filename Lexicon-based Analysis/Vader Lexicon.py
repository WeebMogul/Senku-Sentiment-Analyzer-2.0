from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as pypl
import numpy as np

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
DrStone_max_ep = 25

for ia in range(1, DrStone_max_ep):

    stone_file = 'D:\Python\Senku Sentiment Analyzer 2.0\Data Cleaner\Dr.Stone Episode ' + str(ia) + ' Comment list (cleaned).csv'

    df = pd.read_csv(stone_file)

    str_comments = df['Comment'].astype(str)
    str_length = len(str_comments.index)

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

    data = {'Comment': comments, 'Sentiment Rating': rating}
    df = pd.DataFrame(data)
    df.reset_index(drop=True)

    df.to_csv('D:\Python\Senku Sentiment Analyzer 2.0\Manually determined sentences\Dr. Stone\Dr.Stone Episode ' + str(ia) + ' Comment list with Sentiment rating.csv', encoding='utf-8-sig')

    comments.clear()
    rating.clear()
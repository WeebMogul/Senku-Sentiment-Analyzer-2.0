from textblob import TextBlob
from textblob.sentiments import BaseSentimentAnalyzer
import pandas as pd
import matplotlib.pyplot as pypl
import numpy as np

pos_count = 0
neg_count = 0
pos_value = 0
neutra = 0
me = 0
ar = 0

Kaguya_sama_max_ep = 13
Boogiepop_max_ep = 19
Slime_anime_max_ep = 25

positive = []
negative = []
neutral = []

pos_tags = []
neg_tags = []
neu_tags = []

for ia in range(1,Kaguya_sama_max_ep):
 kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Kaguya-sama cleaned\Cleaned_Kaguya_sama_Episode_'+str(ia)+'_Comment_list.csv'
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Boogiepop cleaned\Cleaned_Boogiepop_Wa_Waranai_Episode_' + str(11) + '_Comment_list.csv'
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Tensei Slime cleaned\Cleaned_Tensei_Slime_Episode_' + str(i) + '_Comment_list.csv'

 df = pd.read_csv(kaguya_file)
 sam = df['Comment'].astype(str)
 sala = len(sam.index)

 for i in range(0,int(sala)):

  analysis = TextBlob(str.lower(sam[i]))

  if(analysis.sentiment.polarity >= 0.0001):
          pos_count += 1
          positive.insert(i - 1, sam[i])
          pos_tags.append(1)
  if(analysis.sentiment.polarity <= -0.0001):
            neg_count +=1
            negative.insert(i-1,sam[i])
            neg_tags.append(0)
            #print(sam[i])
  if(analysis.sentiment.polarity == 0.0):
     neutra += 1
     neutral.insert(i-1,sam[i])
     neu_tags.append(2)

  pos_value += 1


 pos_count = 0
 neg_count = 0
 pos_value = 0
 neutra = 0

 pos_data = {'Comment':positive,'Polarity':pos_tags}
 neg_data = {'Comment':negative,'Polarity':neg_tags}
 neu_data = {'Comment':neutral,'Polarity':neu_tags}
 pos_df = pd.DataFrame(pos_data)
 neg_df = pd.DataFrame(neg_data)
 neu_df = pd.DataFrame(neu_data)

 comment_df = pd.concat([pos_df,neg_df,neu_df])
 comment_df = comment_df.reset_index(drop=True)
 comment_df.to_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\K-Nearest Neighbours\Kaguya-sama Episode ' +str(ia)+' .csv',encoding='utf-8-sig')

 positive.clear()
 negative.clear()
 neutral.clear()
 pos_tags.clear()
 neu_tags.clear()
 neg_tags.clear()
 pos_data.clear()
 neg_data.clear()
 neu_data.clear()
 comment_df = comment_df.drop(['Comment','Polarity'],axis=1)
 pos_df = pos_df.drop(['Comment', 'Polarity'], axis=1)
 neg_df = neg_df.drop(['Comment', 'Polarity'], axis=1)
 neu_df = neu_df.drop(['Comment', 'Polarity'], axis=1)

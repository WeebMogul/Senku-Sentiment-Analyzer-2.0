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



for i in range(1,Slime_anime_max_ep):
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Kaguya-sama cleaned\Cleaned_Kaguya_sama_Episode_'+str(i)+'_Comment_list.csv'
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Boogiepop cleaned\Cleaned_Boogiepop_Wa_Waranai_Episode_' + str(11) + '_Comment_list.csv'
 kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Tensei Slime cleaned\Cleaned_Tensei_Slime_Episode_' + str(i) + '_Comment_list.csv'

 df = pd.read_csv(kaguya_file)
 sam = df['Comment'].astype(str)
 sala = len(sam.index)

 for i in range(0,int(sala)):

  analysis = TextBlob(str.lower(sam[i]))

  if(analysis.sentiment.polarity >= 0.0001):
          pos_count += 1

  if(analysis.sentiment.polarity <= -0.0001):
          neg_count +=1

            #print(sam[i])
  if((analysis.sentiment.polarity == 0.0)):
     neutra += 1


  pos_value += 1

 print(pos_count)
 #print(neg_count)
 #print(neutra)
 pos_perc = (pos_count/pos_value)*100
 neg_perc = (neg_count/pos_value)*100
 neu_perc = (neutra/pos_value)*100

 positive.insert(i-1,(pos_perc))
 negative.insert(i-1,(neg_perc))
 neutral.insert(i-1,(neu_perc))
 #print(positive)
 #print(negative)
 #print(neutral)
 pos_count = 0
 neg_count = 0
 pos_perc = 0
 neg_perc = 0
 pos_value = 0
 neutra = 0

 start = 0
 end = 3
print ("Percentage of positive comments : "  + str(pos_count) + '%')
print ("Percentage of negative comments : " + str(neg_count) + '%')
#print ("Percentage of neutral comments : " + str((neutra/pos_value)*100) + '%\n')

#print(len(pos_tags))

nas = np.arange(Slime_anime_max_ep-1)
pypl.figure()
pypl.xticks((np.arange(Slime_anime_max_ep-1))+0.1,np.arange(1,Slime_anime_max_ep))
pypl.bar(nas+0.00,positive,width=0.25)
pypl.bar(nas+0.25,negative,width=0.25)
pypl.bar(nas+0.50,neutral,width=0.25)
pypl.xlabel('Episode number')
pypl.ylabel('Percentage of sentiments')
pypl.legend(['Positive','Negative','Neutral'],bbox_to_anchor =(1,1),loc='upper left',ncol=1)

#pypl.title("Sentiment Analysis on every episode discussion post of the anime show Kaguya-sama:Love is War using Textblob lexicon")
#pypl.title("Sentiment Analysis on every episode discussion post of the anime show Boogiepop Wa Waranai using Textblob lexicon")
pypl.title("Sentiment Analysis on every episode discussion post of the anime show Tensei Slime using Textblob lexicon")
pypl.show()

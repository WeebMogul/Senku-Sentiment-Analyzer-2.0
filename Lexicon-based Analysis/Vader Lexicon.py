from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as pypl
import numpy as np

positive = []
negative = []
neutral = []

pos_value = 0
neg_value = 0
neu_value = 0
pos_count = 0

anime_dict = \
{
    'kawaii':0.8,
    'cringe':-0.2,
    'god tier':0.8,
    'favourite':0.8,
    'cracked me up':0.7,
    'hype': 0.7
}
Kaguya_sama_max_ep = 13
Boogiepop_max_ep = 13
Slime_anime_max_ep = 13

for ia in range (1,Boogiepop_max_ep):
 kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Kaguya-sama cleaned\Cleaned_Kaguya_sama_Episode_' + str(ia) + '_Comment_list.csv'
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Boogiepop cleaned\Cleaned_Boogiepop_Wa_Waranai_Episode_' + str(ia) + '_Comment_list.csv'
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Tensei Slime cleaned\Cleaned_Tensei_Slime_Episode_' + str(ia) + '_Comment_list.csv'

 df = pd.read_csv(kaguya_file)
 sams = df['Comment'].astype(str)
 sals = len(sams.index)
 analyzer = SentimentIntensityAnalyzer()
 analyzer.lexicon.update(anime_dict)
 for i in range(0,int(sals)):
  vs = analyzer.polarity_scores(sams[i])
  if(vs['compound']> 0.05):
     pos_value +=1
    # print(sams[i])
  if(vs['compound'] < -0.05):
      neg_value +=1
      print(sams[i])
  if ((vs['compound'] < 0.05) & (vs['compound'] > - 0.05)):
      neu_value +=1
      #print(sams[i])
  pos_count+=1

 print('No. of positive comments in episode ' + str(ia)+ ' : '+ str(pos_value) + ' out of ' + str(pos_count))
 positive.append(((pos_value/pos_count)*100))
 negative.append(((neg_value/pos_count)*100))
 #print(positive)
 neutral.append(((neu_value/pos_count)*100))

 pos_value = 0
 neg_value = 0
 neu_value = 0
 pos_count = 0
#print(positive)
nas = np.arange(24)

nas = np.arange(Slime_anime_max_ep-1)
pypl.figure()
pypl.xticks((np.arange(Slime_anime_max_ep-1))+0.1,np.arange(1,Slime_anime_max_ep))
pypl.bar(nas+0.00,positive,width=0.30)
pypl.bar(nas+0.31,negative,width=0.25)
#pypl.bar(nas+0.50,neutral,width=0.25)
pypl.xlabel('Episode number')
pypl.ylabel('Percentage of sentiments')
pypl.legend(['Positive','Negative','Neutral'],bbox_to_anchor =(1,1),loc='upper left',ncol=1)

for a,b in zip(nas,positive):
  #pypl.text(a-0.2,b+0.3,str(round(b,2)) + "%")
  pypl.text(a - 0.2, b + 0.3, str(round(b, 2)))
for a,b in zip(nas,negative):
  #pypl.text(a+0.15,b+0.3,str(round(b,2)) + "%")
  pypl.text(a + 0.15, b + 0.3, str(round(b, 2)))
#for a,b in zip(nas,neutral):
  #pypl.text(a-0.2,b+0.3,str(round(b,2)))

pypl.title("Sentiment Analysis on every episode discussion post of the anime show Kaguya-sama:Love is War using VADER lexicon")
#pypl.title("Sentiment Analysis on every episode discussion post of the anime show Boogiepop Wa Waranai using VADER lexicon")
#pypl.title("Sentiment Analysis on every episode discussion post of the anime show Tensei Slime using VADER lexicon")
pypl.show()

'''
pypl.figure()
pypl.xticks(np.arange(12),np.arange(1,13))
pypl.bar(nas+0.00,positive,width=0.25)
pypl.bar(nas+0.25,negative,width=0.25)
pypl.legend(['Positive','Negative'])
pypl.title("Kaguya-sama")
pypl.show()
'''
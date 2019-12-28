
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
Slime_anime_max_ep = 19

positive = []
negative = []
neutral = []

pos_comments = []
neg_comments = []
neu_comments = []

for ia in range(1,Slime_anime_max_ep):
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Kaguya-sama cleaned\Cleaned_Kaguya_sama_Episode_'+str(i)+'_Comment_list.csv'
 kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Boogiepop cleaned\Cleaned_Boogiepop_Wa_Waranai_Episode_' + str(ia) + '_Comment_list.csv'
 #kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Tensei Slime cleaned\Cleaned_Tensei_Slime_Episode_' + str(11) + '_Comment_list.csv'

 df = pd.read_csv(kaguya_file)
 sam = df['Comment'].astype(str)
 sala = len(sam.index)

 for i in range(0,int(sala)):

  analysis = TextBlob(str.lower(sam[i]))

  if(analysis.sentiment.polarity >= 0.0001):
              pos_count += 1
              pos_comments.insert(i-1,sam[i])
  if(analysis.sentiment.polarity <= -0.0001):
              neg_count +=1
              neg_comments.insert(i - 1, sam[i])
  if(analysis.sentiment.polarity == 0.0):
            neutra +=1
            print(sam[i])
            neu_comments.insert(i-1,sam[i])

  pos_value += 1

 print(pos_count)
 print(neg_count)
 print(neutra)
 pos_perc = (pos_count/pos_value)*100
 neg_perc = (neg_count/pos_value)*100
 neu_perc = (neutra/pos_value)*100

 positive.insert(i-1,(pos_perc))
 negative.insert(i-1,(neg_perc))
 neutral.insert(i-1,(neu_perc))
 print(positive)
 print(negative)
 print(neutral)
 pos_count = 0
 neg_count = 0
 pos_perc = 0
 neg_perc = 0
 pos_value = 0
 neutra = 0


 print ("Percentage of positive comments : "  + str(pos_count) + '%')
 print ("Percentage of negative comments : " + str(neg_count) + '%')
#print ("Percentage of neutral comments : " + str((neutra/pos_value)*100) + '%\n')

 true_pos = 0
 false_pos = 0
 tot = 0

 pos_data = {'Comments' : pos_comments ,'Polarity':1,'Actual Polarity':''}
 neg_data = {'Comments' : neg_comments ,'Polarity':0,'Actual Polarity':''}
 neu_data = {'Comments' : neu_comments ,'Polarity':2,'Actual Polarity':''}
 pos_comms = pd.DataFrame(pos_data)
 neg_comms = pd.DataFrame(neg_data)
 neu_comms = pd.DataFrame(neu_data)

 #print(pos_comms)

 conact = pd.concat([pos_comms,neg_comms,neu_comms],sort=False)
 print(conact['Polarity'])
 conact = conact.reset_index(drop=True)

 conact.to_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Manually determines\Boogiepop Wa Waranai Episode ' +str(ia)+' .csv',encoding='utf-8-sig')
 pos_comments.clear()
 neg_comments.clear()
 neu_comments.clear()
'''
for i in range(0, int(len(neuo_comms.index))):
    ana = TextBlob(str.lower(neuo_comms[i]))

    if (ana.sentiment.polarity >= 0.0001):
        true_pos += 1
    else:
        false_pos += 1
       # pos_comments.insert(i - 1, sam[i])
    tot += 1

print('True Positives : '+str(true_pos)+'/'+str(tot))
'''

'''
nas = np.arange(Slime_anime_max_ep-1)
pypl.figure()
pypl.xticks((np.arange(Slime_anime_max_ep-1))+0.1,np.arange(1,Slime_anime_max_ep))
pypl.bar(nas+0.00,positive,width=0.30)
pypl.bar(nas+0.32,negative,width=0.25)
pypl.xlabel('Episode number')
pypl.ylabel('Percentage of sentiments')
pypl.legend(['Positive','Negative','Neutral'])
for a,b in zip(nas,positive):
  #pypl.text(a-0.2,b+0.3,str(round(b,2)) + "%")
  pypl.text(a - 0.2, b + 0.3, str(round(b, 2)))
for a,b in zip(nas,negative):
  #pypl.text(a+0.15,b+0.3,str(round(b,2)) + "%")
  pypl.text(a + 0.15, b + 0.3, str(round(b, 2)))
#for a,b in zip(nas,neutral):
  #pypl.text(a-0.2,b+0.3,str(round(b,2)))

#pypl.title("Sentiment Analysis on every episode discussion post of the anime show Kaguya-sama:Love is War using Textblob")
#pypl.title("Sentiment Analysis on every episode discussion post of the anime show Boogiepop Wa Waranai using Textblob")
pypl.title("Sentiment Analysis on every episode discussion post of the anime show Tensei Slime using Textblob")
pypl.show()
'''
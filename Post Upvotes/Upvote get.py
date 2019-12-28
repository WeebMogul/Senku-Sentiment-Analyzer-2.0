import praw
import pandas as pd
import numpy as np
import enum
'Finally, note that the value of submission.num_comments may not match up 100% with the number of comments extracted via PRAW. ' \
'This discrepancy is normal as that count includes deleted, removed, and spam comments.'

'Source : https://praw.readthedocs.io/en/latest/tutorials/comments.html'

reddit = praw.Reddit(client_id ='jL_MW1s2qeizKg',
                   client_secret='YiCB2KsvL4azFKPsxNA0KaIkpEo',
                   user_agent='qweqwdsafsa')

Kaguya_sama = {1: 'af8o9s',
               2: 'ahnb1o',
               3: 'ak23r1',
               4: 'amfczc',
               5: 'aotkr4',
               6: 'ara3ix',
               7: 'atww05',
               8: 'awj7tr',
               9: 'az4yez',
               10: 'b1u0bi',
               11: 'b4kt6e',
               12: 'b7cuou'}

Boogiepop = {1: 'acihui',
             2: 'aciknk',
             3: 'aevoai',
             4: 'ahapse',
             5: 'ajpaar',
             6: 'am2b3o',
             7: 'aogs1z',
             8: 'aqwv5c',
             9: 'ati03i',
             10: 'atvtmj',
             11: 'atvtn8',
             12: 'atvxqj',
             13: 'atw2zj',
             14: 'aw5bjg',
             15: 'ayql3m',
             16: 'b1epxw',
             17: 'b6x060',
             18: 'b44ucq'}

Slime_anime = {1: '9khgsd',
             2: '9mfx67',
             3: '9odqrm',
             4: '9qevr6',
             5: '9se4oz',
             6: '9uf14f',
             7: '9wficn',
             8: '9yibuf',
             9: 'a0kvdt',
             10: 'a2qaz9',
             11: 'a4x5xq',
             12: 'a71bh5',
             13: 'a96ds7',
             14: 'adjdoy',
             15: 'afxi6d',
             16: 'aibrpq',
             17: 'akp5lj',
             18: 'an3exy',
             19: 'aphyum',
             20: 'aryzxs',
             21: 'aun5nt',
             22: 'ax91ye',
             23: 'azut50',
             24: 'b2k3yu'}

Kaguya_sama_max_ep = 13
Boogiepop_max_ep = 19
Slime_anime_max_ep = 25

Kaguya_votes = []
Slime_votes = []
Boogiepop_votes = []
for i in range(1,Kaguya_sama_max_ep):
    posts = reddit.submission(id=Kaguya_sama[i])
    k_votes = posts.score
    Kaguya_votes.append(k_votes)

for i in range(1, Slime_anime_max_ep):
    posts = reddit.submission(id=Slime_anime[i])
    s_votes = posts.score
    Slime_votes.append(s_votes)

for i in range(1, Boogiepop_max_ep):
    posts = reddit.submission(id=Boogiepop[i])
    b_votes = posts.score
    Boogiepop_votes.append(b_votes)

epno_kaguya = np.arange(1,13)
epno_slime = np.arange(1,25)
epno_boogiepop = np.arange(1,19)

kag_ep = []
bog_ep = []
slime_ep = []

for i in epno_kaguya:
    k_name = 'Episode '+str(i)
    kag_ep.append(k_name)

for i in epno_slime:
    s_name = 'Episode '+str(i)
    slime_ep.append(s_name)

for i in epno_boogiepop:
    b_name = 'Episode '+str(i)
    bog_ep.append(b_name)


print(type(epno_kaguya.ravel()))

columns = {'Episodes','Upvotes'}
kaguya_df = pd.DataFrame({'Episodes' : kag_ep,'Upvotes' : Kaguya_votes})
kaguya_df.to_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Post Upvotes\ Kaguya-sama episodes posts votes .csv',encoding='utf-8-sig')


slime_df = pd.DataFrame({'Episodes' : slime_ep,'Upvotes' : Slime_votes})
slime_df.to_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Post Upvotes\ Tensei Slime episodes posts votes .csv',encoding='utf-8-sig')

boogiepop_df = pd.DataFrame({'Episodes' : bog_ep,'Upvotes' : Boogiepop_votes})
boogiepop_df.to_csv('D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Post Upvotes\ Boogiepop Wa Waranai episodes posts votes .csv',encoding='utf-8-sig')

import pandas as pd
import emoji

Stone_anime_max_ep = 25

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'',text)

for i in range(1,Stone_anime_max_ep):

 DrStone = r'D:\Python\Senku Sentiment Analyzer 2.0\Comment_Extractor\Dr.Stone\Dr.Stone Episode ' + str(i)+' Comment list.csv'

 df = pd.read_csv(DrStone)

 deletetag_index = df[df['Comment'] == '[deleted]'].index
 df.drop(deletetag_index, inplace=True)

 removetag_index = df[df['Comment'] == '[removed]'].index
 df.drop(removetag_index, inplace=True)

 df['Comment'] = df['Comment'].str.replace(r'\(https?:\/\/.*[\r\n\)]*\)', '')
 df['Comment'] = df['Comment'].str.replace(r'https?:\/\/.*[\r\n\)]*', '')

 df['Comment'] = df['Comment'].str.replace(r'\[[^)]*\(\/s', '')

 df['Comment'] = df['Comment'].str.replace(r'\>.*', ' ')
 df['Comment'] = df['Comment'].str.replace('\r', ' ')
 df['Comment'] = df['Comment'].str.replace('\n','  ')

 df['Comment'] = df['Comment'].str.replace(r'\[|\]|\)', '')

 df['Comment'] = df['Comment'].str.replace(r'\(\/s.|\(|\^|\☞ﾟヮﾟ☞|°|ʖ|¯|_|ツ|_|/|͡', '')
 df['Comment'] = df['Comment'].str.replace(r'|͜ ','')
 df['Comment'] = df['Comment'].str.replace(r'ゴ|ＴＨＩＳ 　ＭＵＳＴ 　ＢＥ 　ＴＨＥ 　ＷＯＲＫ 　ＯＦ 　ＡＮ 　ＥＮＥＭＹ 「ＳＴＡＮＤ」！！ ','')
 df['Comment'] = df['Comment'].str.replace(r'\*|\~|\#|\\', ' ')
 df['Comment'] = df['Comment'].str.replace('&x200B', ' ')
 df['Comment'] = df['Comment'].str.replace('& x200B', ' ')

 df['Comment'] = df['Comment'].apply(lambda x : remove_emoji(x))
 df['Comment'] = df['Comment'].drop_duplicates(keep='first')

 stri = "This comment has been removed.      Please keep all discussion of future events, comparisons with the source material or just general talk about the source material in the Source Material Corner.    ---  Have a question or think this removal was an error? Message the mods.     Don't know the rules? Read them hereranimewikirules."

 sourcetag = df[df['Comment'] == stri].index
 df.drop(sourcetag, inplace=True)

 df=df.dropna()

 stone_file = r'D:\Python\Senku Sentiment Analyzer 2.0\Data Cleaner\Dr.Stone Episode ' + str(i) + ' Comment list (cleaned).csv'
 df.to_csv(stone_file,encoding='utf-8-sig',index=False)



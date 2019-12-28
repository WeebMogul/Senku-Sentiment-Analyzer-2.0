import praw
import pandas as pd
import numpy as np
import enum
'Finally, note that the value of submission.num_comments may not match up 100% with the number of comments extracted via PRAW. ' \
'This discrepancy is normal as that count includes deleted, removed, and spam comments.'

'Source : https://praw.readthedocs.io/en/latest/tutorials/comments.html'

reddit = praw.Reddit(client_id='jL_MW1s2qeizKg',
                   client_secret='YiCB2KsvL4azFKPsxNA0KaIkpEo',
                   user_agent='qweqwdsafsa')

Stone_anime = {1: 'c9gm0u',
               2: 'ccbm1g',
               3: 'cf8ad3',
               4: 'ci3lz0',
               5: 'cl4wqu',
               6: 'co372p',
               7: 'cr756a',
               8: 'cuerhp',
               9: 'cxhrr4',
               10: 'd0hgfr',
               11: 'd3q43y',
               12: 'd6vtkq',
               13: 'da19xh',
               14: 'dd86iy',
               15: 'dgg1qa',
               16: 'djodqq',
               17: 'dmy9sy',
               18: 'dq52vc',
               19: 'dtgzdp',
               20: 'dwscka',
               21: 'e02lfb',
               22: 'e3g5n8',
               23: 'e708er',
               24: 'ea5gi2'

}

Stone_anime_max_ep = 25

for i in range(1,Stone_anime_max_ep):

 posts = reddit.submission(id=Stone_anime[i])
 upvotes = posts.score

 posts.comments.replace_more(limit=None)
 comment_array = []


 def get_replies(comment):
    for replies in comment.replies:
      print(replies.body)
      comment_array.append(replies.body)
      get_replies(replies)

 for comments in posts.comments:
     print(comments.body)
     comment_array.append(comments.body)
     get_replies(comments)

 #Comment_and_post_upvotes = 'Comments (Total post upvotes : '+ str(upvotes)+ ' ( '+ str(percent_of_upvotes) +'% ) )'

 columns = {'Comment'}
 df = pd.DataFrame(np.array(comment_array).reshape(len(comment_array), 1), columns=columns)

 file_name = 'Dr.Stone Episode ' + str(i) + ' Comment list.csv'
 df.to_csv(file_name, encoding='utf-8-sig')
 df.empty
 comment_array.clear()


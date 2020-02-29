import pandas as pd
import numpy as np
import matplotlib.pyplot as pypl

df = pd.read_csv(r'D:\Python\Senku Sentiment Analyzer 2.0\Post Upvotes\ Dr Stone episodes posts votes .csv',
                 encoding='utf-8-sig', index_col=0)

episodes = np.arange(1,4)
votes = df['Upvotes'].values

ef = df.sort_values(by='Upvotes',ascending=False)
print(ef)

pypl.title('Vote counts for Dr. Stone')
pypl.bar(episodes,votes)
pypl.xticks(episodes)
pypl.xlabel('Episodes')
pypl.ylabel('Number of votes')
pypl.show()


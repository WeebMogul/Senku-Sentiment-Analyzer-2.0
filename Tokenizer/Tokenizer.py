import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

Kaguya_sama_max_ep = 13
Boogiepop_max_ep = 19
Slime_anime_max_ep = 25

def penntag(pen):
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[pen[:2]]
    except:
        return 'n'

for ep in range(1,Slime_anime_max_ep):

 boogiepop_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Boogiepop cleaned\Cleaned_Boogiepop_Wa_Waranai_Episode_' + str(ep) + '_Comment_list.csv'
 kaguya_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Kaguya-sama cleaned\Cleaned_Kaguya_sama_Episode_' + str(ep) + '_Comment_list.csv'
 slime_file = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Data Cleaner\Tensei Slime cleaned\Cleaned_Tensei_Slime_Episode_' + str(ep) + '_Comment_list.csv'

 print('File '+str(ep)+ ' read')
 df = pd.read_csv(slime_file)

 # Convert dataframe values into string
 df['Comment'] = df['Comment'].astype(str)

 #Remove punctuation marks and tokenize each and every word
 df['Comment'] = df['Comment'].str.replace('[^\w\s]',' ')
 comment_words = df['Comment'].apply(word_tokenize)


 lemma = WordNetLemmatizer()
 stopword = set(stopwords.words('english'))

 # Put words into lowercase and remove stopwords
 lowercase_words = comment_words.apply(lambda x : [word.lower() for word in x])
 words_without_stopwords = lowercase_words.apply(lambda x : [word for word in x if word not in stopword])

 # remove further stopwords
 stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
 stopwords_jsa = set(stopwords_json['en'])
 stopword_removed = words_without_stopwords.apply(lambda x : [word for word in x if word not in stopwords_jsa])

 #Apply part-of-speech to tokenized words
 POS_words = stopword_removed.apply(nltk.pos_tag)
 samples = POS_words[:POS_words.shape[0]]

 list_of_lemmad_words = []

 for i in range(0,samples.shape[0]):
  #lemmatize the tokens using Penn Treebank POS tokens in the penntag() function
  lemma_words = [lemma.lemmatize(words.lower(),pos=penntag(tags)) for words,tags in samples[i]]
  list_of_lemmad_words.append([lemma_words])

  tokeniz = pd.DataFrame(list_of_lemmad_words,columns=['Values'])

 def return_back_df(doc):
    return doc

 #perfom TF-IDF on the tokens
 token_lemma_values = tokeniz['Values']
 tfidf = TfidfVectorizer(

 )
 response = tfidf.fit_transform(token_lemma_values)

 feature_names = tfidf.get_feature_names()

 dense = response.todense()
 wordlist = dense.tolist()

 #Put TF-IDF values onto the dataframe and sum up the values
 tfif_val = pd.DataFrame(wordlist,columns=feature_names)
 sum_of_tfidf = tfif_val.sum(numeric_only =True)

 #Normalize and sort the TF-IDF values
 normalized_values = (sum_of_tfidf - sum_of_tfidf.mean())/(sum_of_tfidf.max() - sum_of_tfidf.min())
 sorted_norms = normalized_values.sort_values(ascending=False)


 csv_file_name_kaguya = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Tokenizer\Kaguya-sama TF-IDF\TF-IDF words of Kaguya-sama Episode ' +str(ep)+' .csv'
 csv_file_name_boogiepop = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Tokenizer\Boogiepop Wa Waranai TF-IDF\TF-IDF words of Boogiepop Wa Waranai Episode ' + str(ep) + ' .csv'
 csv_file_name_slime = 'D:\Github Projects\Heriot-Watt-Msc-Project-Sentiment-Analysis\Tokenizer\Tensei Slime TF-IDF\TF-IDF words of Tensei Slime Episode ' +str(ep)+' .csv'

 #Write the TF-IDF values to the file.
 print('File ' + str(ep) + ' Written\n')
 sorted_norms.to_csv(csv_file_name_slime,encoding='utf-8-sig')





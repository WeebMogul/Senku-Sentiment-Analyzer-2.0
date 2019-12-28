# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y


print("There are 10 sentences of following three classes on which K-NN classification and K-means clustering" \
" is performed : \n1. Cricket \n2. Artificial Intelligence \n3. Chemistry")
path = "Sentences.txt"

train_clean_sentences = []
fp = open(path, 'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)

print(train_clean_sentences)
vector = TfidfVectorizer(stop_words='english')
x = vector.fit_transform(train_clean_sentences)
print(x)

y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

modelKnn = KNeighborsClassifier(n_neighbors=5)
modelKnn.fit(x,y_train)

test_sentences = ["Chemical compunds are used for preparing bombs based on some reactions",
                  "bat",
                  "Machine learning is a area of Artificial intelligence"]

test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+", "", cleaned)
    test_clean_sentence.append(cleaned)

print(train_clean_sentences)
Test = vector.transform(test_clean_sentence)
print(Test)
true_test_labels = ['Cricket', 'AI', 'Chemistry']
predicted_labels_knn = modelKnn.predict(Test)

print("\nBelow 3 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ", \
test_sentences[0], "\n2. ", test_sentences[1], "\n3. ", test_sentences[2])
print("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print("\n", test_sentences[0], ":", true_test_labels[np.int(predicted_labels_knn[0])],"\n")
print(test_sentences[1],":", true_test_labels[np.int(predicted_labels_knn[1])])
print("\n", test_sentences[2], ":", true_test_labels[np.int(predicted_labels_knn[2])])

"""### Import Statements"""

import os
import random
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


"""### Downloading extra dependencies from NLTK"""

nltk.download('punkt')
nltk.download('stopwords')


"""### Getting stopwords customized to your problem statement"""

#Use this function to create custom list of stop_words for your Project
path = r'C:\Users\devda\Desktop\Internship\Clustering_SpiderG_Tfid_method\Stopwords\stopwords_not_to_be_used.txt'
def get_stopwords(path):
  stopwords = nltk.corpus.stopwords.words('english')
  not_words = []
  with open(path,'r') as f:
    not_words.append(f.readlines())
  not_words = [word.replace('\n','') for words in not_words for word in words]
  not_words = set(not_words)
  stopwords = set(stopwords)
  customized_stopwords = list(stopwords - not_words)
  return stopwords,customized_stopwords

stop_words,customized_stopwords = get_stopwords(path)


"""### Loading the Data"""

path = r'C:\Users\devda\Desktop\Internship\DataSets\Para_Dataset_created\Articles'
seed = 137
def load_data(path,seed): 
  train_texts = []
  for fname in sorted(os.listdir(path)):
    if fname.endswith('.txt'):
      with open(os.path.join(path,fname),'r') as f:
        train_texts.append(f.read())
  random.seed(seed)
  random.shuffle(train_texts)
  return train_texts
train_texts = load_data(path,seed)


"""### Tokenizing the document and filtering the tokens"""

def tokenize(train_texts):
  filtered_tokens = []
  tokens = [word for sent in nltk.sent_tokenize(train_texts) for word in nltk.word_tokenize(sent)]
  for token in tokens:
    if re.search('[a-zA-Z]',token):
      filtered_tokens.append(token)
  return filtered_tokens


"""### Tokenizing and stemming using Snowball stemmer"""

def tokenize_stem(train_texts):
  tokens = tokenize(train_texts)
  stemmer = SnowballStemmer('english')
  stemmed_tokens = [stemmer.stem(token) for token in tokens]
  return stemmed_tokens



"""### Generating the vocab for problem statement"""

def generate_vocab(train_texts):
  vocab_tokenized = []
  vocab_stemmed = []
  total_words = []
  for text in train_texts:
    allwords_tokenized = tokenize(text)
    total_words.append(allwords_tokenized)
    vocab_tokenized.extend(allwords_tokenized)    
    allwords_stemmed = tokenize_stem(text)
    vocab_stemmed.extend(allwords_stemmed)
  return vocab_tokenized,vocab_stemmed,total_words
vocab_tokenized,vocab_stemmed,total_words = generate_vocab(train_texts)


"""### Calculating Tf-idf matrix"""

'''
Use stop_words = customized_stopwords if you want to use your own set of stopwords else leave it as it is.
Functions available for tokenizer -> 1)tokenize  2) tokenize_stem  3) Remove the attribute to use default function
''' 

def tfid_vector(train_texts):
  tfidf_vectorizer = TfidfVectorizer(max_df = 0.85, min_df = 0.13, sublinear_tf = True, stop_words = 'english', use_idf = True, tokenizer = tokenize, ngram_range = (1,10))
  tfidf_matrix = tfidf_vectorizer.fit_transform(train_texts)
  return tfidf_matrix
tfidf_matrix = tfid_vector(train_texts)


"""### Clustering Using K - Means"""

#Code For Elbow Method
nc = range(1,10)
kmeans = [KMeans(n_clusters = i, n_init = 100, max_iter = 500, precompute_distances = 'auto' ) for i in nc]
score = [kmeans[i].fit(tfidf_matrix).score(tfidf_matrix) for i in range(len(kmeans))]
plt.plot(nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


#Uncomment the below code after getting appropriate k value from the graph
'''
K_value = "Get the value from the graph"
km = KMeans(n_clusters = K_value, n_init = 2000, max_iter = 6000, precompute_distances = 'auto' )
clusters = km.fit_predict(tfidf_matrix)
clusters = list(clusters)
print(clusters)

'''






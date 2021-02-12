from gensim import utils
import gensim.models
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
import numpy as np
import gensim.downloader



class MyCorpus:
        """An iterator that yields sentences (lists of str)."""
        def __iter__(self):
            corpus_path = './bbc.lines'
            for line in open(corpus_path):
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)

# Build your own W2V model                
def build_w2v_model():
    sentences = MyCorpus()
    return gensim.models.Word2Vec(sentences=sentences)



# Turn a text into a concatenation of W2V vectors.
def word2vec_transformer(texts, w2v_model, dimension=100):
        vectors=[]
        for text in texts:
           words=text.split(" ")
           vector=[]
           for word in words[:50]:
                   if word in w2v_model.wv:
                         vector.extend(np.array(w2v_model.wv[word]))
                   else:
                         vector.extend(np.zeros(dimension))
           vectors.append(vector)
        return np.array(vectors)


# Option 1: build own model on BBC data
# --------------------------------------
#  model=build_w2v_model()


# Option 2:
# ---------------------------------------------
# Download existing (small) model and use that (model has 25-dimensional vectors)
model = gensim.downloader.load('glove-twitter-25')
    
df=pd.read_csv("bbc-all.csv",encoding= 'unicode_escape') # try encoding = 'utf-8'; encoding = 'ISO-8859-1' for unicode errors
X=df['news_item']
y=df['label']

# Split data into training, test, and training/test labels. Test=10% of all data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state = 3, stratify=y)

# TD.IDF
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()    
X_train_counts = count_vect.fit_transform(X_train)
X_train_vectors = tfidf_transformer.fit_transform(X_train_counts)    
X_test_counts = count_vect.transform(X_test)
X_test_vectors = tfidf_transformer.transform(X_test_counts)

    
# Word2vec: choose a model option (see above)
dimension=25
X_train_vectors=word2vec_transformer(X_train,model,dimension)
X_test_vectors=word2vec_transformer(X_test,model,dimension)

clf=Perceptron().fit(X_train_vectors, y_train)

X_test=list(X_test)

error=0

n=0

results=open("results.txt","w")

for (vector,label) in zip(X_test_vectors,y_test):
       text=X_test[n]
       n+=1
       pred=clf.predict(vector.reshape(1, -1))[0]
       if pred!=label:
               error+=1        
       print("\"%s\",%s,%s"%(text,label,pred),file=results)

results.close()

print("See results.txt. Error=",error)
     

             

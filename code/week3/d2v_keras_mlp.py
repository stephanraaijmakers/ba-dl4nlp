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

from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


def scale_data(X_train, X_test):
        min=np.min(np.array(X_train+X_test))
        max=np.max(np.array(X_train+X_test))
        X_train=(X_train-min)/(max-min)
        X_test=(X_test-min)/(max-min)
        return X_train, X_test
        

def create_keras_model(X_train, y_train):
        input_dim = X_train.shape[1]
        nb_classes = y_train.shape[1]
        print("Input dimension, nb_classes:",input_dim, nb_classes)
        
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim))
        model.add(Activation('relu'))
        #model.add(Dropout(0.15))

        #model.add(Dense(64))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.15))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.summary()
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
        print("Training...")
        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2)

        return model

def run_keras_model(model, X_test, y_test):
        results = model.evaluate(X_test, y_test, batch_size=64)
        print("Loss, accuracy:", results)
        print("Predicting...")
        preds = model.predict_classes(X_test, verbose=0)


# =====================================


class MyCorpus:
        """An iterator that yields sentences (lists of str)."""
        def __iter__(self):
            corpus_path = './bbc.lines'
            for line in open(corpus_path,'rb'): # Prevent unicode errors with 'rb'
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)

                
def build_d2v_model(dimension):
    texts= MyCorpus()
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=dimension, window=2, min_count=1, workers=4)
    return (documents, model)
    

def test_d2v_model(documents,str):
        tokens = str.split()
        sims = model.docvecs.most_similar(positive=[model.infer_vector(tokens)],topn=100)
        for (doc_id, value) in sims:
            print(' '.join(documents[doc_id].words))
                    
def doc2vec_transformer(texts, d2v_model, dimension=100):
        vectors=[]
        for text in texts:                
           tokens=text.split(" ")
           vector=model.infer_vector(tokens)
           vectors.append(np.array(vector))
        return np.array(vectors)


# Delete if  __name__ etc. when working in Colab
if __name__=="__main__":
   dimension=25
   (docs, model)=build_d2v_model(dimension)
   df=pd.read_csv("bbc-all.csv",encoding= 'ISO-8859-1') # try encoding = 'utf-8'; encoding = 'ISO-8859-1' for unicode errors

   X=df['news_item']
   y=df['label']

   # Split data into training, test, and training/test labels. Test=10% of all data.
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state = 3, stratify=y)
   label_encoder = LabelBinarizer()

   y_train = label_encoder.fit_transform(y_train)
   y_test=label_encoder.transform(y_test)

   X_train_vectors=doc2vec_transformer(X_train,model,dimension)
   X_test_vectors=doc2vec_transformer(X_test,model,dimension)

#   X_train_vectors, X_test_vectors=scale_data(X_train_vectors, X_test_vectors)
   
   model=create_keras_model(X_train_vectors,y_train)
   run_keras_model(model,X_test_vectors,y_test)


             

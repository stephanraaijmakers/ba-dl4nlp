import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding

from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer



def create_keras_model(X_train, y_train, vocab_len, max_len):
        nb_classes = y_train.shape[1]
        model = Sequential()
        model.add(Embedding(input_dim=vocab_len,output_dim=8,input_length=max_len))
        model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        
        print("Training...")
        model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)

        return model

def run_keras_model(model, X_test, y_test):
        results = model.evaluate(X_test, y_test, batch_size=64)
        print("Loss, accuracy:", results)
        print("Predicting...")
        preds = model.predict_classes(X_test, verbose=0)
        # Tensorflow >2.7: preds = np.argmax(model.predict(X_test),axis=1)


def embed_documents(texts):
        embedded_docs=[]
        Lexicon={}
        for doc in texts:
                for word in doc.split(" "):
                        Lexicon[word]=1
        vocab_len=len(Lexicon)
        max_len=0
        for doc in texts:
                embedded_docs.append(one_hot(doc,vocab_len))
                l=len(doc.split(" "))
                if l>max_len:
                        max_len=l
        # Padding
        embedded_docs=pad_sequences(embedded_docs,maxlen=max_len,padding='post',value=0.0)
        return (embedded_docs,vocab_len,max_len)
                        

# Delete if  __name__ etc. when working in Colab
if __name__=="__main__":
   df=pd.read_csv("bbc-all.csv",encoding= 'ISO-8859-1') # try encoding = 'utf-8'; encoding = 'ISO-8859-1' for unicode errors
   X=df['news_item']
   y=df['label']

   (X, vocab_len, max_len)=embed_documents(X)
   
   # Split data into training, test, and training/test labels. Test=10% of all data.
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state = 3, stratify=y)
   label_encoder = LabelBinarizer()

   y_train = label_encoder.fit_transform(y_train)
   y_test=label_encoder.transform(y_test)
   
   model=create_keras_model(X_train, y_train, vocab_len, max_len)
   run_keras_model(model,X_test,y_test)


             

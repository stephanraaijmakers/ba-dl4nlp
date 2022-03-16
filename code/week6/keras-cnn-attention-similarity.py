import tensorflow as tf
from tensorflow.keras import *
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_file
from sklearn.preprocessing import LabelBinarizer

import random
import sys


def get_data(fn): # sentence 1, sentence 2, label (0/1) for similarity
    X=[]
    y=[]
    with open(fn,"r") as fp:
        lines=fp.readlines()
    for line in lines:
        fields=line.rstrip().split("\t")
        X.append([fields[0].split(" "),fields[1].split(" ")])
        y.append(float(fields[2]))
    fp.close()    
    return X,y

def vectorize_documents(texts):
    vect_queries=[]
    vect_values=[]
    Lexicon={}
    for doc_pair in texts:
        for word in doc_pair[0]:
            Lexicon[word]=1
        for word in doc_pair[1]:
            Lexicon[word]=1
    vocab_len=len(Lexicon)
    max_len=0
    for doc_pair in texts:
        doc=' '.join([w for w in doc_pair[0]])
        vect_queries.append(one_hot(doc,vocab_len))
        l=len(doc)
        if l>max_len:
            max_len=l
        doc=' '.join([w for w in doc_pair[1]])
        vect_values.append(one_hot(doc,vocab_len))
        l=len(doc)
        if l>max_len:
            max_len=l
    vect_queries=pad_sequences(vect_queries,maxlen=max_len,padding='post',value=0.0)
    vect_values=pad_sequences(vect_values,maxlen=max_len,padding='post',value=0.0)
    return (np.array(vect_queries),np.array(vect_values),vocab_len,max_len)


def create_model(input_dim, vocab_size):
    query_input = tf.keras.Input(shape=(input_dim,), dtype='int32')
    value_input = tf.keras.Input(shape=(input_dim,), dtype='int32')

    token_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256)
    query_embeddings = token_embedding(query_input)
    value_embeddings = token_embedding(value_input)

    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention(use_scale=True)(
        [query_seq_encoding, value_seq_encoding])


    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    value_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        value_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    input=tf.keras.layers.Dense(128, activation="relu")(input)
    pred=tf.keras.layers.Dense(1,activation="softmax")(input)

    model=tf.keras.models.Model([query_input, value_input],pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



if __name__=="__main__":
    X,y=get_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state = 3, stratify=y)
    queries_train, values_train, vocab_len, max_len=vectorize_documents(X_train)
    y_train=np.array(y_train)
    model=create_model(max_len, vocab_len)
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png')
    model.fit([queries_train, values_train], y_train,epochs=10)

    



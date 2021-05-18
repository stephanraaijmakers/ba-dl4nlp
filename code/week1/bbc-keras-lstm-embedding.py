import sys
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Embedding, Activation, Conv1D, Flatten, MaxPooling1D, LSTM, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split as sk_train_test_split



def create_tokenizer(X_data, num_words=150000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_data)
    return tokenizer


def pad_data(X, tokenizer, max_len):
    X = tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(X, maxlen=max_len)
    return X

def split_data(X_data, Y_data, tokenizer, max_sequence_length, test_size=0.3):
    X_data = pad_data(X_data, tokenizer, max_sequence_length)
    Y_data = Y_data.astype(np.int32)
    X_train, X_test, Y_train, Y_test = sk_train_test_split(X_data, Y_data, test_size=test_size)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test



def load_embedding(tokenizer):
    embedding_dim = 100
    embeddings_index = {}

    embf = open('glove.6B.100d.txt') # See data in this github
    for line in embf:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    embf.close()

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector 
    return embedding_matrix, embedding_dim

def create_model(tokenizer, input_length, nb_classes):
    word_index = tokenizer.word_index
    embedding_matrix, embedding_dim = load_embedding(tokenizer)

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=True))
    model.add(SpatialDropout1D(0.3)) # try alternatives here
    model.add(LSTM(64, # try other values here 
                   activation='tanh',
                   dropout=0.2,
                   recurrent_dropout=0.5))
    # Maybe squeeze in a few Dense layers?
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def process_labels (Y_data):
    label_id = {}
    label_name = {}
    for index, c in enumerate(Y_data):
        if c in label_id:
            label = label_id[c]
        else:
            label = len(label_id)
            label_id[c] = label
            label_name[label] = c
    
        Y_data[index] = label
    return Y_data, label_name # for translating back labels to text

def main(fname): # ensure the file is permuted! 

    data = pd.read_csv(fname, encoding='ISO-8859-1')
    max_words=10000 # Play with different values
    max_len=1000 # idem

    X_data = data[['news_item']].to_numpy().reshape(-1)
    Y_data = data[['label']].to_numpy().reshape(-1)
     
    Y_data, label_name=process_labels(Y_data) 
    
    tokenizer = create_tokenizer(X_data)
    X_train, X_test, Y_train, Y_test = split_data(X_data, Y_data, tokenizer, max_len, test_size=0.2)

    nb_classes = np.max(Y_train) + 1
    model = create_model(tokenizer, max_len, nb_classes)
    batch_size = 32
    epochs = 10
    
    model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, validation_split=0.1)

    score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Use label_name ^^ to interpret predicted labels


if __name__=="__main__":
    fname=sys.argv[1]
    main(fname) # filename bbc data

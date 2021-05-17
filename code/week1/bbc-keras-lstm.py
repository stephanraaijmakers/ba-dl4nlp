import sys
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Embedding, Activation, Conv1D, Flatten, MaxPooling1D, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras


def split_data(data, train_perc=.08):
    train_size = int(len(data) * train_perc)
    train = data[:train_size]
    test = data[train_size:]
    x_train = train['news_item']
    y_train = train['label']
    x_test  = test['news_item']
    y_test = test['label']
    return x_train, x_test, y_train, y_test

# From words to integers
def tokenize_text(x_train, x_test, max_words=1000):
    tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, 
                                              char_level=False)
    tokenize.fit_on_texts(x_train)
    x_train = tokenize.texts_to_matrix(x_train) # produces a nested array [[doc1=word_id_1,...],[doc2=(etc)]]
    x_test = tokenize.texts_to_matrix(x_test)
    return x_train, x_test

# From labels to integers, from integers to one-hot
def encode_labels(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test  = encoder.transform(y_test)
    # Now we have integers, let's convert thenm to one-hot vectors of size <nb_classes>
    nb_classes = np.max(y_train) + 1
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return y_train, y_test, nb_classes

def main(fname): # ensure the file is permuted! 

    data = pd.read_csv(fname, encoding='ISO-8859-1')
    max_words=1000
    x_train,x_test,y_train,y_test=split_data(data, train_perc=.8)
    x_train, x_test=tokenize_text(x_train,x_test, max_words=max_words)
    y_train, y_test, nb_classes=encode_labels(y_train,y_test)

#    x_train = x_train.reshape(len(x_train), max_words, 1)
#    x_test = x_test.reshape(len(x_test), max_words, 1)
#    ^^ Use if LSTM is first layer

    
    # Model definition

    batch_size = 32
    epochs = 10
    model = Sequential()
    model.add(Embedding(input_dim=max_words, 
                           output_dim=64, 
                           input_length=max_words))
    #model.add(Flatten())
    model.add(Dense(64))
    model.add(LSTM(32))
   # model.add(Conv1D(32, 3, activation='relu'))
#    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__=="__main__":
    fname=sys.argv[1]
    main(fname) # filename bbc data

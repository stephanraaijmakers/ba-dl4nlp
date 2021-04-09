from tensorflow.keras.layers import Embedding, Dropout, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow
import tensorflow as tf
import string
from tensorflow.keras import Model,Input
import sys
import re


def text_normalization(input_data):
        lowercased = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercased, "[%s]" % re.escape(string.punctuation),"")



def get_data(directory):
    train_data = tf.keras.preprocessing.text_dataset_from_directory(
        directory,
        batch_size=32,
        validation_split=0.2,
        subset="training",
        seed=1337,
    )
    test_data = tf.keras.preprocessing.text_dataset_from_directory(
        directory, batch_size=32
    )
    return (train_data, test_data)


def create_model(train_data,test_data):
    max_tokens=10000
    max_length=100
    batch_size = 32
    embedding_dim=128

    vectorized_text_layer = TextVectorization(
            standardize=text_normalization,
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_length)

    def vectorize_text(text,label):
            text = tf.expand_dims(text, -1)
            return vectorized_text_layer(text),label

    X_train=train_data.map(vectorize_text)
    X_test=test_data.map(vectorize_text)
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    x = Embedding(max_tokens, embedding_dim)(inputs)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid", name="predictions")(x)    
    model = Model(inputs, predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    model.fit(X_train, epochs=5)
    model.evaluate(X_test)
            

def main(train,test):
     create_model(train,test)

if __name__=="__main__":
    (train,test)=get_data(sys.argv[1])
    main(train,test)
    

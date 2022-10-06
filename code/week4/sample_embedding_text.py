import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences



docs=[
    'terrible service!',
    'lame soup',
    'crude waitress',
    'food was cold',
    'great',
    'great food']

vocab_size=len(set([w for doc in docs for w in doc]))
one_hot=[one_hot(d,vocab_size) for d in docs]
print("One-hot:",one_hot)

max_length = 3
padded = pad_sequences(one_hot,maxlen=max_length,padding='post')
print("Padded:",padded)

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=3))

model.compile('rmsprop', 'mse')
emb = model.predict(padded)
print(emb)

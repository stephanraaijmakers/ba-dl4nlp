import numpy as np
from keras.models import Sequential
from keras.layers import Embedding

model = Sequential()
model.add(Embedding(10, 8, input_length=3))

input = np.random.randint(10, size=(1, 3))
model.compile('rmsprop', 'mse')
emb = model.predict(input)
print(emb)

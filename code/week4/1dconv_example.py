from tensorflow.keras import models
from tensorflow.keras import layers


model = models.Sequential()
model.add(layers.Embedding(1000,8, input_length=50))
model.add(layers.Conv1D(32, 3))

model.summary()

import sys
import numpy as np
import re
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, GRU, Embedding, Reshape, TimeDistributed, Bidirectional,InputLayer, Activation, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers

from tensorflow.keras.callbacks import Callback

import tensorflow
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from attention import Attention

import matplotlib.pyplot as plt
from matplotlib import cm
from keract import get_activations


from sklearn.preprocessing import minmax_scale

from nltk import tokenize

# See for attention: https://github.com/philipperemy/keras-attention-mechanism

filename = sys.argv[1]

review_texts = open(filename, 'r', encoding='ISO-8859-1').readlines()
Words={}
Labels={}

labels=[]
reviews=[]


class MyAttention(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        self._name='attention_weight'
        super(MyAttention, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self._trainable_weights = [self.W, self.b, self.u]
        super(MyAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]




for review in review_texts:
        m=re.match("^(.+),([^,]+)$",review.lstrip().rstrip())
        if m:
             sentences=tokenize.sent_tokenize(m.group(1))
             reviews.append(sentences)
             l=len(sentences)
             label=m.group(2)
             labels.append(label)
             Labels[label]=1
             for sentence in sentences:
                     words=sentence.lower().split(" ")
                     for word in words:
                             Words[word]=1

words = sorted(Words)
vocab_len=len(Words)+1
word_to_int = dict((w, i+1) for i, w in enumerate(words))
word_to_int[0]=0 # padding
int_to_word = dict((i+1, w) for i, w in enumerate(words))
int_to_word[0]="-PAD-" # padding

label_names=sorted(Labels)
label_to_int = dict((w, i) for i, w in enumerate(label_names))
int_to_label = dict((i, w) for i, w in enumerate(label_names))

for i in range(len(labels)):
        labels[i]=label_to_int[labels[i]]


max_review_len=10
max_sent_len = 100
max_words = 200000


data = np.zeros((len(reviews), max_review_len, max_sent_len), dtype='int32')
for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < max_review_len:
                k = 0
                for word in sent.lower().split(" "):
                        if k < max_sent_len and word_to_int[word] < max_words:
                                data[i, j, k] = word_to_int[word]
                                k = k + 1


labels=to_categorical(np.asarray(labels))                                
print(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
y=labels

#data=data[:100]
#y=y[:100]

validation_split=0.1
nb_validation_samples = int(validation_split * data.shape[0])


X_train = data[:-nb_validation_samples]
y_train = y[:-nb_validation_samples]
X_test=data[nb_validation_samples:]
y_test=y[nb_validation_samples:]

# Model (functional API)

embedding_layer=Embedding(vocab_len, 128,input_length=max_sent_len, mask_zero=False,trainable=True)
inp = Input(shape=(max_sent_len,), dtype='int32')
emb=embedding_layer(inp)
bi_lstm_word=Bidirectional(LSTM(max_sent_len, return_sequences=True))(emb)
#bi_lstm_word=LSTM(max_sent_len, return_sequences=True)(emb)
attention_words=MyAttention(max_sent_len)(bi_lstm_word)
sentence_encoder=Model(inp,attention_words)

review_inp=Input(shape=(max_review_len, max_sent_len),dtype='int32')
review_encoder=TimeDistributed(sentence_encoder)(review_inp)
bi_lstm_sentence=Bidirectional(LSTM(max_review_len, return_sequences=True))(review_encoder)
#bi_lstm_sentence=LSTM(max_sent_len, return_sequences=True)(review_encoder)
attention_sentence=MyAttention(max_review_len)(bi_lstm_sentence)
predictions=Dense(3, activation='softmax')(attention_sentence)
model=Model(review_inp,predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

sentence_encoder.summary()
model.summary()

#plot_model(model, to_file='hierarchical_attention_sentence_encoder.png')
#plot_model(model, to_file='hierarchical_attention_model.png')


model.fit(X_train,y_train,epochs=5,batch_size=64)


#predictions=model.predict(X_test)

def normalize(probs):
        prob_factor = 1.0 / sum(probs)
        return [prob_factor * p for p in probs]


def normprobs(probs):
    return normalize(minmax_scale(probs))
    
def to_words(x):
        s=""
        for w in x:
            if w!=0: # padding
               s+=int_to_word[w]+" "
        return(s.rstrip())

n=0    
for x_test in X_test[:20]:
        prediction=model.predict(np.array([x_test]))[0]
        sentence_attention = get_activations(model, np.array([x_test]), layer_names='my_attention_1')
        sent_probs=normprobs(sentence_attention['my_attention_1'][0])[:max_review_len]
        print(sent_probs)
        m=np.argmax(sent_probs,axis=0)
        most_important_sentence=x_test[m]
        print("MOST IMPORTANT SENT=%s (%f)"%(to_words(most_important_sentence),sent_probs[m]))
        word_attention = get_activations(sentence_encoder, np.array([most_important_sentence]), layer_names='my_attention')
        word_probs=normprobs(word_attention['my_attention'][0])[:max_sent_len]
    
        d=dict((i,v) for (i,v) in enumerate(word_probs) if most_important_sentence[i]!=0)
        sorted_probs=sorted(d.items(),key=lambda x:x[1],reverse=True)

        s=0.0
        for (i,prob) in sorted_probs:
            s+=prob
        for (i,prob) in sorted_probs:
            prob/=s
            print("%s:%f"%(int_to_word[most_important_sentence[i]],prob),end="|")    
        pred_label=int_to_label[np.argmax(prediction,axis=0)]
        gt_label=int_to_label[np.argmax(y_test[n],axis=0)]      
        print("\nLabel=%s Prediction=%s"%(gt_label,pred_label))
        n+=1

    

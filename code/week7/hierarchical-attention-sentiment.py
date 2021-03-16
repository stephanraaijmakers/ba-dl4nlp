import sys
import numpy as np
import re
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Embedding, Reshape, TimeDistributed, Bidirectional,InputLayer, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model


from tensorflow.keras.callbacks import Callback

import tensorflow
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from attention import Attention

import matplotlib.pyplot as plt
from matplotlib import cm
from keract import get_activations

from nltk import tokenize

# See for attention: https://github.com/philipperemy/keras-attention-mechanism

filename = sys.argv[1]

review_texts = open(filename, 'r', encoding='ISO-8859-1').readlines()
Words={}
Labels={}

max_sent_len=30
max_review_len=5

labels=[]
reviews=[]

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
int_to_word[0]=0 # padding

label_names=sorted(Labels)
label_to_int = dict((w, i) for i, w in enumerate(label_names))
int_to_label = dict((i, w) for i, w in enumerate(label_names))

for i in range(len(labels)):
        labels[i]=label_to_int[labels[i]]


max_review_len=15
max_sent_len = 100
max_words = 20000


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

validation_split=0.1
nb_validation_samples = int(validation_split * data.shape[0])

data=data[:1000]
y=y[:1000]


X_train = data[:-nb_validation_samples]
y_train = y[:-nb_validation_samples]
X_test=data[nb_validation_samples:]
y_test=y[nb_validation_samples:]

# Model (functional API)

embedding_layer=Embedding(vocab_len, 128,input_length=max_sent_len, mask_zero=True,trainable=True)
inp = Input(shape=(max_sent_len,), dtype='int32')
emb=embedding_layer(inp)
bi_lstm_word=Bidirectional(LSTM(256, return_sequences=True))(emb)
attention_words=Attention()(bi_lstm_word)
sentence_encoder=Model(inp,attention_words)

review_inp=Input(shape=(max_review_len, max_sent_len),dtype='int32')
review_encoder=TimeDistributed(sentence_encoder)(review_inp)
bi_lstm_sentence=Bidirectional(LSTM(256, return_sequences=True))(review_encoder)
attention_sentence=Attention()(bi_lstm_sentence)
predictions=Dense(3, activation='softmax')(attention_sentence)
model=Model(review_inp,predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

sentence_encoder.summary()
model.summary()


#plot_model(model, to_file='LSTM_pos_windowed_attention_model.png')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model.fit(X_train,y_train,epochs=1,batch_size=64)


#predictions=model.predict(X_test)

n=0
for x_test in X_test:
        #word_attention_map = get_activations(sentence_encoder, np.array([x_test]), layer_names='attention_weight')
        sentence_attention_map = get_activations(model, np.array([x_test]), layer_names='attention_weight')
        for sent in x_test:
                word_attention_map = get_activations(sentence_encoder, np.array([sent]), layer_names='attention_weight')
        # Next combine weights: per sentence, per word
        print("YES")
        exit(0)
        #print(attention_map['attention_weight'][0])
        a=attention_map['attention_weight'][0]
        print(a)
        exit(0)
        print_a=False
        for i in range(len(a)):
                if i!=focus_position-1:
                        if a[i]>0.05:
                                print_a=True
                
        if print_a:
                print("Found:",n)
                xvals=np.arange(len(x_test))
                words=tuple([int_to_word[w] for w in x_test])
                plt.xticks(xvals,words)
                plt.imshow(attention_map['attention_weight'], cmap=cm.afmhot) # cmap='hot')
                plt.title(f'Attention')
                plt.savefig(f'attention-%d.png'%(n))
                n+=1
        #plt.close()

        
n=0
for sentence in X_test:
        sentence_prediction=predictions[n]
        n+=1
        best_tag=int_to_tag[np.argmax(sentence_prediction,axis=0)]
        m=1
        for int_word in sentence:
            if m==focus_position:
                print("<<%s_%s>>"%(int_to_word[int_word],best_tag),end=" ")
            else:
                print(int_to_word[int_word],end=" ")
            m+=1
        print()
              

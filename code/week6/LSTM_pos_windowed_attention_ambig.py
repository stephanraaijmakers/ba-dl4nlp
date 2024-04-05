import sys
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Embedding, Reshape, TimeDistributed, Bidirectional,InputLayer,Activation
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

# See for attention: https://github.com/philipperemy/keras-attention-mechanism


filename = sys.argv[1]

pos_tagged_text = open(filename, 'r', encoding='ISO-8859-1').readlines()

Words={}
Tags={}
HasTag={}

n_words=0
n_sentences=0

maxlen=0

for line in pos_tagged_text:
        if line[0]==' ':
            next
        n_sentences+=1
        words=line.rstrip().lower().split(" ")
        l=len(words)        
        if l>maxlen:
                maxlen=l
        n_words+=l
        if words[0]=='':
            next
        for word in words:
            m=re.match("^([^\_]+)\_([^\s]+)",word)
            if m:
                word=m.group(1)
                tag=m.group(2)
                Words[word]=1
                if word not in HasTag:
                        HasTag[word]=[tag]
                elif tag not in HasTag[word]:
                        HasTag[word].append(tag)
                Tags[tag]=1

AmbigTag={}
for word in HasTag:
        HasTag[word]='|'.join(sorted(HasTag[word]))
        AmbigTag[HasTag[word]]=1

ambig_tags=sorted(AmbigTag)
ambig_to_int = dict((w, i+1) for i, w in enumerate(ambig_tags))
ambig_to_int[0]=0 # padding
int_to_ambig = dict((i+1, w) for i, w in enumerate(ambig_tags))
int_to_ambig[0]=0 # padding
vocab_len=len(AmbigTag)+1

words = sorted(Words)
word_to_int = dict((w, i+1) for i, w in enumerate(words))
word_to_int[0]=0 # padding
int_to_word = dict((i+1, w) for i, w in enumerate(words))
int_to_word[0]=0 # padding

tags=sorted(Tags)
tag_to_int = dict((t, i+1) for i, t in enumerate(tags))
tag_to_int[0]=0 # padding 
int_to_tag = dict((i+1, t) for i, t in enumerate(tags))
int_to_tag[0]=0 # padding

X=[]
y=[]

focus_position=3

for line in pos_tagged_text:
        if line[0]==' ':
            next
        sent=[]
        words=line.rstrip().lower().split(" ")
        n=1
        int_tag=0
        for word in words:
            m=re.match("^([^\_]+)\_(.+)",word)            
            if m:
                word=m.group(1)
                tag=m.group(2)
                sent.append(ambig_to_int[HasTag[word]])
                if n==focus_position:                    
                    int_tag=tag_to_int[tag]
            n+=1
        X.append(sent)
        y.append(int_tag)

X = pad_sequences(X, maxlen=maxlen, padding='post')
#y = pad_sequences(y, maxlen=maxlen, padding='post')

X=np.array(X)
y=np.array(y)

model = Sequential()
model.add(InputLayer(input_shape=(maxlen, )))
model.add(Embedding(vocab_len, 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Attention())
model.add(Dense(len(Tags)+1))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model.summary()


#plot_model(model, to_file='LSTM_pos_windowed_attention_model.png')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model.fit(X_train,to_categorical(y_train),epochs=2,batch_size=64)

#model.evaluate(X_test,to_categorical(y_test))


predictions=model.predict(X_test)


def normalize(probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

n=0
for x_test in X_test[:10]:
        attention_map = get_activations(model, np.array([x_test]), layer_names='attention')
        a=attention_map['attention'][0]
        total=0.0
        for i in range(len(a)):
                if i==focus_position-1:
                        continue
                total+=a[i]
        for i in range(len(a)):
                if i==focus_position-1:
                        continue
                a[i]/=total
        print(a)
        a=np.array([a])
        xvals=np.arange(len(x_test))
        words=tuple([int_to_ambig[w] for w in x_test])
        plt.xticks(xvals,words,rotation='vertical')
        plt.imshow(a, cmap=cm.afmhot,vmin=0, vmax=1) # cmap='hot')
        plt.title(f'Attention')
        plt.savefig(f'norm-attention-%d.png'%(n))
        plt.close()
        n+=1
        
n=0
for sentence in X_test:
        sentence_prediction=predictions[n]
        n+=1
        best_tag=int_to_tag[np.argmax(sentence_prediction,axis=0)]
        m=1
        for int_word in sentence:
            if m==focus_position:
                print("<<%s_%s>>"%(int_to_ambig[int_word],best_tag),end=" ")
            else:
                print(int_to_ambig[int_word],end=" ")
            m+=1
        print()
              

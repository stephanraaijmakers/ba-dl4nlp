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

max_sent_len=100
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
int_to_word[0]="-PAD-" # padding

label_names=sorted(Labels)
label_to_int = dict((w, i) for i, w in enumerate(label_names))
int_to_label = dict((i, w) for i, w in enumerate(label_names))

for i in range(len(labels)):
        labels[i]=label_to_int[labels[i]]


max_review_len=15
max_sent_len = 100
max_words = 50000


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

#data=data[:1000]
#y=y[:1000]

validation_split=0.1
nb_validation_samples = int(validation_split * data.shape[0])


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

model.fit(X_train,y_train,epochs=10,batch_size=64)


#predictions=model.predict(X_test)

def normalize(probs):
        prob_factor = 1.0 / sum(probs)
        return [prob_factor * p for p in probs]

def to_words(x):
        s=""
        for w in x:
                if w!=0: # padding
                        s+=int_to_word[w]+" "
        return(s.rstrip())

n=0
for x_test in X_test[:10]:
        prediction=model.predict(np.array([x_test]))[0]
        sentence_attention = get_activations(model, np.array([x_test]), layer_names='attention_weight')
        sent_probs=sentence_attention['attention_weight'][0]
        print(sent_probs)
        for x in x_test:
                print(to_words(x),end=" ")
        print()
        m=np.argmax(sent_probs,axis=0)
        most_important_sentence=x_test[m]
        print("MOST IMPORTANT SENT=% (%f)"%(to_words(most_important_sentence),sent_probs[m]))
        word_attention = get_activations(sentence_encoder, np.array([most_important_sentence]), layer_names='attention_weight')
        word_probs=word_attention['attention_weight'][0]
        word_probs=normalize(word_probs)
        print("NORM wprobs:",word_probs)
        i=0
        for word in word_probs:
                if most_important_sentence[i]!=0: #padding
                        print("%s:%f"%(int_to_word[most_important_sentence[i]],word_probs[i]),end="|")
                i+=1
        pred_label=int_to_label[np.argmax(prediction,axis=0)]
        gt_label=int_to_label[np.argmax(y_test[n],axis=0)]      
        print("\nLabel=%s Prediction=%s"%(gt_label,pred_label))
        
        #for sent in x_test:
                #word_attention_map = get_activations(sentence_encoder, np.array([sent]), layer_names='attention_weight')
        # Next combine weights: per sentence, per word

        xvals=np.arange(len(x_test))
        words=tuple([int_to_word[w] for w in most_important_sentence])
        plt.xticks(xvals,words)
        wp=np.array([word_probs])
        plt.imshow(wp, cmap=cm.afmhot) # cmap='hot')
        plt.title(f'Attention gt=%s pred=%s'%(gt_label,pred_label))
        plt.savefig(f'hierarchical-attention-%d.png'%(n))
        
        n+=1

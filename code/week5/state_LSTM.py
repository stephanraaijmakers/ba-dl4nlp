import sys
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


def factorize(X, y, window_size):
        new_X = []
        new_y = []
        for i, sequence in enumerate(X):
            l = len(sequence)
            for start in range(0, l - window_size + 1):
                end = start + window_size
                sub_sequence = sequence[start:end]
                new_X.append(sub_sequence)
                new_y.append(y[i])
        return np.array(new_X), np.array(new_y)

def generate_data(n_samples, max_len, window_size):
        X=np.zeros(shape=(n_samples,max_len))
        one_indexes = np.random.choice(n_samples, int(n_samples/2), replace=False)
        X[one_indexes, 0] = 1
        y=X[:,0]
        X,y=factorize(X,y,window_size)
        X=X.reshape((X.shape[0],window_size,1))
        return (X,y)


def stateless(X_train,y_train,X_test,y_test,window_size,epochs):
        batch_size=10
        model = Sequential()
        model.add(LSTM(10, input_shape=(window_size, 1), return_sequences=False, stateful=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test), shuffle=False)

        print(model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0))


def stateful(X_train,y_train,X_test,y_test,window_size,epochs):       
        batch_size=10
        model = Sequential()
        model.add(LSTM(10, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        for epoch in range(epochs):
                print("Epoch %d/%d: training fase..."%(epoch+1,epochs),end='')
                for i in range(len(X_train)):
                        y_true = y_train[i]
                        for j in range(window_size):
                                model.train_on_batch(np.expand_dims(np.expand_dims(X_train[i][j], axis=1), axis=1),
						   np.array([y_true]))
                        model.reset_states()
                mean_te_acc = []
                mean_te_loss = []
                print("testing fase")
                for i in range(len(X_test)):
                        for j in range(window_size):
                                te_loss,te_acc=model.test_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1),
                                                                        np.array([y_test[i]]))
                                mean_te_acc.append(te_acc)
                                mean_te_loss.append(te_loss)
                        model.reset_states()
        print('Accuracy test data = {}'.format(np.mean(mean_te_acc)))
        print('___________________________________')


if __name__=="__main__":
        n_samples=int(sys.argv[1])
        max_len=int(sys.argv[2])
        window_size=int(sys.argv[3])
        
        X,y=generate_data(n_samples, max_len,window_size)

        n_train=int(0.8*n_samples)
        
        X_train=X[:n_train]
        y_train=y[:n_train]
        X_test=X[n_train:]
        y_test=y[n_train:]

        epochs=5

        print("Stateless LSTM")
        stateless(X_train,y_train,X_test,y_test,window_size,epochs)

        print("Stateful LSTM")
        X,y=generate_data(n_samples, 1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        stateful(X_train,y_train,X_test,y_test,1,epochs)


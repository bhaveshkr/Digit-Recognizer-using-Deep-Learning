
# coding: utf-8

__author__ = 'Bhavesh Kumar'

import pandas as pd
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# prepare data - training
data_frame = pd.read_csv('train.csv')
data_matrix = data_frame.as_matrix()
x_train = data_matrix[:,1:]
y_train = data_matrix[:,0]
X_train, X_test, y_train, y_test = train_test_split(x_train,y_train)
#normalize
X_train = X_train/255
X_test = X_test/255

# prepare data - testing
test_data_frame = pd.read_csv('test.csv')
test_data_matrix = test_data_frame.as_matrix()
#normalize
test = test_data_matrix/255

# convert to categories binay matrix using One Hot encoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
y_train = oneHotEncoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = oneHotEncoder.fit_transform(y_test.reshape(-1,1)).toarray()

# Neural Network
prediction_network = Sequential()
prediction_network.add(Dense(units=10, kernel_initializer='uniform', activation='relu', input_dim=784))
prediction_network.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))
prediction_network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting model
prediction_network.fit(X_train, y_train, batch_size=10, epochs=10)

# evaluate model
scores = prediction_network.evaluate(X_test, y_test)
print("Baseline Error Result: %.2f%%" % (100-scores[1]*100))

def one_hot_decode(encoded_seq):
    for index, vector in enumerate(encoded_seq):
            label = argmax(vector)
            csv_row = {'ImageId':index,'Label':label}
            df = pd.DataFrame(csv_row, index=[0])
            with open('result.csv', 'a') as f:
                df.to_csv(f, index=False, header=False)
            f.close()
            
y_pred = prediction_network.predict(test)
one_hot_decode(y_pred)


#Artificial Neural Network

#Data Preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,1] = labelEncoder_x.fit_transform(x[:,1])
labelEncoder_x1 = LabelEncoder()
x[:,2] = labelEncoder_x1.fit_transform(x[:,2])
oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()

#Dummy variable trap
x = x[:,1:]

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initailizing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(6,activation='relu',input_shape=(11,)))

#Adding second hidden layer
classifier.add(Dense(6,activation='relu'))

#Adding output layer
classifier.add(Dense(1,activation='sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

#Predicting the Regression model
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
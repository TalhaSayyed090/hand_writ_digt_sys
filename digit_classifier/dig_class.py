#importing the required dependences
import tensorflow
from tensorflow import keras
from keras.layers import Dense,Conv2D,Flatten, AveragePooling2D
from keras import Sequential
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import pickle

#loading the dataset and spliting in tho training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)


#creating the model
model = Sequential()

model.add(Conv2D(6, kernel_size=(5,5), padding='valid', activation='tanh', input_shape = (28, 28, 1)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(16, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(120, activation='tanh'))
model.add(Dense(82, activation='tanh'))
model.add(Dense(10, activation='softmax'))

print(model.summary())  #Printng the over all record the created model


model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs = 5, batch_size = 1000)  #training the model through trainng data

y_prob = model.predict(x_test)  #predecting the probabality of the model of testing data

print(y_prob)
y_predic = y_prob.argmax(axis=1)  #finding of the model through testing data
print('Probabality of the first prdecton of test data_set', y_prob[0]) 
print(y_prob.argmax(axis=1))   

#Finding the Accuracy Score the Model
accu_score = accuracy_score(y_test,y_predic)
print('The Accuracy Score of the Model is : ',accu_score)

#saving it in the pickle file
pickle.dump(model, open('model.pkl', 'wb'))



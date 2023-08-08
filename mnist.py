#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Author: Devan Blanscett
' Program name: Hello Numerical Classification
' Program Description: This is the "hello world" of machine learning. This algorithm/script
' 							  imports the kera library, imports the mnist data set then trains and 
'							  tests an algorithm to identify hand written numerical characters.
'
'
'							  Code copied from "deep learning in Python" by Francois Chollet
'							  Comments are my own
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#28 by 28 because each image is 28 pixels by 28 pixels
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

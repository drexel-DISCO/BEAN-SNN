'''
MIT License

Copyright (c) 2022 Drexel Distributed, Intelligent, and Scalable COmputing (DISCO) Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Author: Anup Das
Email : anup.das@drexel.edu
'''
import keras
from keras import callbacks
from keras.optimizers import SGD, Adadelta, Adagrad

import tensorflow as tf

import numpy as np
from sklearn.metrics import log_loss, accuracy_score

#load and process the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

nb_classes = len(np.unique(y_train))
x_train     = x_train.astype('float32')
x_test      = x_test.astype('float32')
x_train     /= 255.0
x_test      /= 255.0

x_train     = np.moveaxis(np.expand_dims(x_train, -1),3,1)
x_test      = np.moveaxis(np.expand_dims(x_test, -1),3,1)
y_train     = keras.utils.to_categorical(y_train, nb_classes)
y_test      = keras.utils.to_categorical(y_test, nb_classes)

#load the model
model_name = 'bean_lenet'
json_fname = 'lenet/' + model_name + '.json'
h5_fname   = 'lenet/' + model_name + '.h5'
from utils import load_trained_model
model = load_trained_model(json_fname,h5_fname)
print(model.summary())

#hyperparameters
batch_size = 100
learning_rate = 0.0005

#model optimizer
model.compile(
    optimizer=keras.optimizers.Adam(lr=learning_rate),
    loss='categorical_crossentropy',
    metrics='accuracy')

#evaluate the model
y_predict       = np.argmax(model.predict(x_test, batch_size=batch_size, verbose=1)[3],axis=1)
y_test_labels   = np.argmax(y_test,axis=1)
score           = accuracy_score(y_test_labels,y_predict)
print('Accuracy = ',score)

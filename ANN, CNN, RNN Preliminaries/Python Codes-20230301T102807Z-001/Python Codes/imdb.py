# -*- coding: utf-8 -*-


from tensorflow.keras.datasets import imdb
#Loading the datasets
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)#The argument num_words=10000 means you’ll only 
#keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded

train_data[0]
train_labels[0]
max([max(sequence) for sequence in train_data])
word_index = imdb.get_word_index()
# creating a dict with index number and the text
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



import numpy as np
#vectorization
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))#creates a matrix with len(seq) rows and dimension columns
    for i, sequence in enumerate(sequences):#creates count value pair        
        results[i, sequence] = 1.#when you get i row and seq col value,replace a matrix with 1
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_train[0]

#converting the inputs to float type
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


#defining the model
from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))#converting the int data to tuple
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#from keras import optimizers
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

##splitting the data into traning and validation

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))

history_dict = model.history
history_dict.keys()

#Plotting validation scores
import matplotlib.pyplot as plt
acc = model.history['accuracy']
val_acc = model.history['val_accuracy']
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs= range(1,len(acc)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
'''The attribute Loc in legend() is used to specify the location of the legend.Default value of loc is loc=”best” (upper left)'''
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#hyperparameter tuning the model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

model.predict(x_test)

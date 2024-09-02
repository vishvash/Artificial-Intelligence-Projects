# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:09:16 2024

@author: Lenovo
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset using pandas
# Assuming you have a CSV file named 'your_dataset.csv'
df = pd.read_csv(r'C:/Users/Lenovo/Downloads/Study material/AI/ANN, CNN, RNN Preliminaries/Assignment/fedex.csv')

# Assuming your dataset contains features and labels
# Replace 'features_column_names' and 'label_column_name' with actual column names from your dataset
# Split the dataset into features and labels
features = df.iloc[:, :6]
labels = pd.DataFrame(df['Delivery_Status'])

# Assuming you have already preprocessed your features and labels as required

# Split the dataset into training and testing sets
# You can adjust the test_size parameter to change the ratio of training and testing data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


#standardizing the dataset
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test /= std

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU

# Define your model
model = models.Sequential()
model.add(layers.Dense(16, activation=LeakyReLU(alpha=0.1)))
model.add(layers.Dense(16, activation=LeakyReLU(alpha=0.1)))
model.add(layers.Dense(1, activation='tanh'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


x_val = x_train[:300]
partial_x_train = x_train[300:]
y_val = y_train[:300]
partial_y_train = y_train[300:]


# Train the model
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=100, validation_data=(x_val, y_val))

# Plotting validation scores
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Hyperparameter tuning the model
# Assuming you have already defined and compiled your model
# Train the model with your full dataset
model.fit(x_train, y_train, epochs=10, batch_size=100)

# Evaluate the model on the test data
results = model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)

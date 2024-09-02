import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical


# Define the alphabet sequence
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Create a mapping from characters to integers
char_to_int = {char: i for i, char in enumerate(alphabet)}
int_to_char = {i: char for i, char in enumerate(alphabet)}

# Prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []

for i in range(0, len(alphabet) - seq_length):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))

# Normalize the input
X = X / float(len(alphabet))

# One-hot encode the output variable
y = to_categorical(dataY)
# Define the alphabet sequence
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Create a mapping from characters to integers
char_to_int = {char: i for i, char in enumerate(alphabet)}
int_to_char = {i: char for i, char in enumerate(alphabet)}

# Prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []

for i in range(0, len(alphabet) - seq_length):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))

# Normalize the input
X = X / float(len(alphabet))

# One-hot encode the output variable
y = to_categorical(dataY)

# Define a more complex LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=1000, batch_size=10, verbose=2)

# Function to predict the next character
def predict_next_char(input_char):
    x = np.array([[char_to_int[input_char]]]) / float(len(alphabet))
    x = np.reshape(x, (1, seq_length, 1))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    return int_to_char[index]

# Test the model
for char in alphabet:
    next_char = predict_next_char(char)
    print(f"Input: {char}, Predicted Next Char: {next_char}")

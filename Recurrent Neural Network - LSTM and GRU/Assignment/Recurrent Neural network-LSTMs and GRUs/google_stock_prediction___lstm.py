import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pi
import plotly.express as px
import datetime
from datetime import timedelta


# Load the stock data from the provided CSV file

stock_data = pd.read_csv('C:/Users/Lenovo/Downloads/Study material/AI/Recurrent Neural Network - LSTM and GRU/Assignment/Recurrent Neural network-LSTMs and GRUs/GOOG.csv')

stock_data.head()

stock_data.info()

stock_data.describe().T

stock_data.isna().sum()

fig = px.line(stock_data, x='date', y='volume', title='Daily Trading Volume')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Volume')
fig.update_layout(template='plotly_dark')
fig.show()

fig = px.line(stock_data, x='date', y='close', title='Closing Prices Over Time')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Closing Price')
fig.update_layout(template='plotly_dark')
fig.show()

fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['open'], mode='lines+markers', name='Open'))
fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['high'], mode='lines+markers', name='High'))
fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['low'], mode='lines+markers', name='Low'))
fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], mode='lines+markers', name='Close'))

fig.update_layout(title='Stock Price Analysis',
                  xaxis_title='Date',
                  yaxis_title='Price')

fig.show()

daily_changes = stock_data['close'].diff()
fig = px.histogram(daily_changes, nbins=50, title='Histogram of Daily Price Changes')
fig.update_xaxes(title='Daily Price Change')
fig.update_yaxes(title='Frequency')
fig.update_layout(template='plotly_dark')
fig.show()

stock_data['20-day MA'] = stock_data['close'].rolling(window=20).mean()

fig = go.Figure(data=[go.Candlestick(
    x=stock_data['date'],
    open=stock_data['open'],
    high=stock_data['high'],
    low=stock_data['low'],
    close=stock_data['close'],
    name="Candlesticks",
    increasing_line_color='green',
    decreasing_line_color='red',
    line=dict(width=1),
    showlegend=False
)])

fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['20-day MA'], mode='lines', name='20-day Moving Average', line=dict(color='rgba(255, 255, 0, 0.3)')))

fig.update_layout(
    title="Google Stock Price Candlestick Chart with Moving Average",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
)

fig.show()

stock_data = stock_data.drop('20-day MA', axis=1)

"""## <span style="color: #d62d20">Data Preprocessing</span>

<span style="color: blue">1.</span> Convert the 'date' column in the stock data to datetime format using `pd.to_datetime`.
<br>
<span style="color: blue">2.</span> Sort the stock data by date in ascending order using `sort_values`.
<br>
<span style="color: blue">3.</span> Create a new DataFrame 'stock' containing the selected columns (`date`, `close`, `high`, `low`, `open`, `volume`) for further analysis.

"""

stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data = stock_data.sort_values('date')

stock = stock_data[['date', 'close', 'high', 'low', 'open', 'volume']]

"""## <span style="color: #d62d20">Data Normalization</span>

<span style="color: blue">1.</span> Initialize a Min-Max Scaler using `MinMaxScaler()`.
<br>
<span style="color: blue">2.</span> Create a copy of the 'stock' DataFrame containing columns ('open', 'high', 'low', 'volume', 'close').
<br>
<span style="color: blue">3.</span> Use the scaler to fit and transform the data, performing Min-Max normalization.

"""

scaler = MinMaxScaler()
normalized_data = stock[['open', 'high', 'low', 'volume', 'close']].copy()
normalized_data = scaler.fit_transform(normalized_data)

"""## <span style="color: #d62d20">Data Splitting</span>

<span style="color: blue">1.</span> Split the normalized data into training and testing sets using `train_test_split`. The testing set size is set to 20% of the data, and `shuffle` is set to `False` to maintain the chronological order.

<span style="color: blue">2.</span> Create a DataFrame 'train_df' containing the training data with columns (`date`, `close`, `high`, `low`, `open`, `volume`).

<span style="color: blue">3.</span> Create a DataFrame 'test_df' containing the testing data with columns (`date`, `close`, `high`, `low`, `open`, `volume`).

"""

train_data, test_data = train_test_split(normalized_data, test_size=0.2, shuffle=False)

train_df = pd.DataFrame(train_data, columns=['open', 'high', 'low', 'volume', 'close'])
test_df = pd.DataFrame(test_data, columns=['open', 'high', 'low', 'volume', 'close'])

"""## <span style="color: #d62d20">Sequence Generation</span>

<span style="color: blue">1.</span> Define a function named `generate_sequences` that takes a DataFrame `df` and an optional parameter `seq_length` (default is 50).

<span style="color: blue">2.</span> Extract the relevant columns ('open', 'high', 'low', 'volume', 'close') from the DataFrame `df` and reset the index.

<span style="color: blue">3.</span> Initialize empty lists `sequences` and `labels` to store the sequences and labels for training.

<span style="color: blue">4.</span> Iterate through the data using a sliding window approach. For each index, append the next `seq_length` rows as a sequence and the corresponding last row as the label.

<span style="color: blue">5.</span> Convert the lists of sequences and labels into NumPy arrays.

<span style="color: blue">6.</span> Return the generated sequences and labels.

Then we generate sequences and labels for training data using `generate_sequences` function on 'train_df' DataFrame.

And next, we generate sequences and labels for testing data using `generate_sequences` function on 'test_df' DataFrame.

"""

def generate_sequences(df, seq_length=50):
    X = df[['open', 'high', 'low', 'volume', 'close']].reset_index(drop=True)
    y = df[['open', 'high', 'low', 'volume', 'close']].reset_index(drop=True)

    sequences = []
    labels = []

    for index in range(len(X) - seq_length + 1):
        sequences.append(X.iloc[index : index + seq_length].values)
        labels.append(y.iloc[index + seq_length - 1].values)

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels

train_sequences, train_labels = generate_sequences(train_df)
test_sequences, test_labels = generate_sequences(test_df)

"""## <span style="color: #d62d20">Model Architecture</span>

<span style="color: blue"> 1. </span> Create a Sequential model.

<span style="color: blue">2.</span> Add the first LSTM layer with 50 units and return sequences. The input shape is set to (50, 5), where 50 is the sequence length and 5 is the number of features ('open', 'high', 'low', 'volume', 'close').
   - **LSTM Layer 1:**
      - Units: 50
      - Input Shape: (50, 5)
      - Return Sequences: True

<span style="color: blue">3.</span> Apply dropout regularization with a rate of 0.2 to mitigate overfitting.
   - **Dropout Layer 1:**
      - Rate: 0.2

<span style="color: blue">4.</span> Add the second LSTM layer with 50 units and return sequences.
   - **LSTM Layer 2:**
      - Units: 50
      - Return Sequences: True

<span style="color: blue">5.</span> Apply dropout regularization with a rate of 0.2.
   - **Dropout Layer 2:**
      - Rate: 0.2

<span style="color: blue">6.</span> Add the third LSTM layer with 50 units.
   - **LSTM Layer 3:**
      - Units: 50

<span style="color: blue">7.</span> Apply dropout regularization with a rate of 0.2.
   - **Dropout Layer 3:**
      - Rate: 0.2

<span style="color: blue">8.</span> Add a fully connected Dense layer with 5 units as the output layer.
   - **Dense Layer:**
      - Units: 5 (output)

This model architecture comprises multiple LSTM layers with dropout regularization to prevent overfitting.

"""

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(50, 5)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=5)
])

"""## <span style="color: #d62d20">Compile and Summary</span>

<span style="color: blue">1.</span> Compile the model for training using the mean squared error as the loss function and the Adam optimizer. Additionally, track the mean absolute error as a metric.
   - **Loss Function:** Mean Squared Error
   - **Optimizer:** Adam
   - **Metrics:** Mean Absolute Error

<span style="color: blue">2.</span> Display a summary of the model's architecture and parameters.

"""

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

"""## <span style="color: #d62d20">Model Training</span>

<span style="color: blue"> 1. </span> Set the number of training epochs to 200 and the batch size to 32 for training the model.

<span style="color: blue">2.</span> Train the model using the training sequences and labels. During training, the model will run for the specified number of epochs, updating its weights to minimize the loss.
   - **Number of Epochs:** 200
   - **Batch Size:** 32
   - **Training Data:** `train_sequences` and `train_labels`
   - **Validation Data:** `test_sequences` and `test_labels`
   - **Verbose:** Display training progress information.

"""

epochs = 200
batch_size = 32

history = model.fit(
    train_sequences,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_sequences, test_labels),
    verbose=1
)

train_predictions = model.predict(train_sequences)
test_predictions = model.predict(test_sequences)

"""## <span style="color: #d62d20">Training Data Predictions</span>"""

fig = make_subplots(rows=1, cols=1, subplot_titles=('Close Predictions'))

train_close_pred = train_predictions[:, 0]
train_close_actual = train_labels[:, 0]

fig.add_trace(go.Scatter(x=np.arange(len(train_close_actual)), y=train_close_actual, mode='lines', name='Actual', opacity=0.9))
fig.add_trace(go.Scatter(x=np.arange(len(train_close_pred)), y=train_close_pred, mode='lines', name='Predicted', opacity=0.6))

fig.update_layout(title='Close Predictions on Train Data', template='plotly_dark')
fig.show()

"""### <span style="color: #d62d20">Next 10 Days Predictions</span>

<span style="color: blue">1.</span> Initialize an empty list `latest_prediction` to store the model's predictions.

<span style="color: blue">2.</span> Extract the last sequence of the test data using `test_sequences[:-1]`.

<span style="color: blue">3.</span> Loop 10 times to predict the next values. In each iteration, predict the next sequence using the model and append the prediction to `latest_prediction`.
"""

latest_prediction = []
last_seq = test_sequences[:-1]

for _ in range(10):
    prediction = model.predict(last_seq)
    latest_prediction.append(prediction)

pi.templates.default = "plotly_dark"

predicted_data_next = np.array(latest_prediction).reshape(-1, 5)
last_date = stock['date'].max()
next_10_days = [last_date + timedelta(days=i) for i in range(1, 11)]

for i, feature_name in enumerate(['open', 'high', 'low', 'volume', 'close']):
    if feature_name in ['volume', 'close']:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=next_10_days, y=predicted_data_next[:, i],
                                 mode='lines+markers', name=f'Predicted {feature_name.capitalize()} Prices'))

        fig.update_layout(title=f'Predicted {feature_name.capitalize()} Prices for the Next 10 Days',
                          xaxis_title='Date', yaxis_title=f'{feature_name.capitalize()} Price')

        fig.show()

"""<div style="width: 90%; height: 100px; background-color: #ffffff; border: 3px solid #d62d20; text-align: center; line-height: 100px; color: #0057e7; font-size: 24px; font-weight: bold; border-radius:6px;">
    Thanks for paying attention ❤️
</div>
"""
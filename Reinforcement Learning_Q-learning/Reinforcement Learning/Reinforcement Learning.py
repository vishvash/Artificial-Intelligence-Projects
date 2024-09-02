# Data for analysis
import pandas as pd
alldata = pd.read_pickle('C:/Users/Bharani Kumar/nasdaq100_6y.pkl')
symbol = 'AAL'
data = alldata[symbol].values
data = pd.DataFrame(data)
data.columns = ['close']
close = data.close

import talib

macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# Cancel NaN values
import numpy as np
macdhist = macdhist[~np.isnan(macdhist)]
macd = macd[-len(macdhist):]
macdsignal = macdsignal[-len(macdhist):]

from sklearn import preprocessing
#Sclaing MACD hist to '[0, 1] range
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
macdhist_norm = min_max_scaler.fit_transform(np.expand_dims(macdhist, axis=1))

# Implement strategy
start_sell = 0.4
stop_sell = 0.1
start_buy = -0.4
stop_buy = -0.1

y = np.full(len(macdhist), np.nan)
y[0] = 0

for i in range(1, len(macdhist)):

    if y[i-1] == 0:
        if (macdhist_norm[i] >= start_sell):
            # Enter sell position
            y[i] = -1
        elif (macdhist_norm[i] <= start_buy):
            # Enter buy position
            y[i] = 1
        else:
            y[i] = 0
    elif y[i-1] == -1:
        if macdhist_norm[i] > stop_sell:
            # Stay in sell position
            y[i] = -1
        else:
            # Leave sell position
            y[i] = 0
    else:
        if macdhist_norm[i] < stop_buy:
            # Stay in buy position
            y[i] = 1
        else:
            # Leave buy position
            y[i] = 0

# Plot strategy
import matplotlib.pyplot as plt
dates = np.arange(len(macdhist))
plt.plot(dates, y,'g', label='Strategy Positions')
plt.bar(dates, macdhist_norm[:, 0], width=1, color='blue', label='MACD histogram')
plt.plot(dates, start_sell * np.ones(len(macdhist)), 'k--', lw=1)
plt.plot(dates, stop_sell * np.ones(len(macdhist)), 'k--', lw=1)
plt.plot(dates, start_buy * np.ones(len(macdhist)), 'k--', lw=1)
plt.plot(dates, stop_buy * np.ones(len(macdhist)), 'k--', lw=1)
plt.xlabel('Days')
plt.xlim((300, 600))
plt.legend()
plt.savefig('AAL_macd.png', bbox_inches='tight')
plt.show()
    
# Generate input data - technical indicators
ind1 = talib.MIDPOINT(close)    # Overlap: MidPoint over period
ind2 = talib.HT_DCPERIOD(close) # Cycle Indicator Functions:  Hilbert Transform - Dominant Cycle Period
ind3 = talib.MAX(close)         # Math Operator: Highest value over a specified period
ind4 = talib.SIN(close)         # Math Transform: Vector Trigonometric Sin
ind5 = talib.APO(close)         # Momentum: Absolute Price Oscillator

X = np.vstack((macdhist, macd, macdsignal, ind1[-len(macdhist):], ind2[-len(macdhist):],
               ind3[-len(macdhist):], ind4[-len(macdhist):], ind5[-len(macdhist):]))
X = X.T
print(X.shape)

# Data Preparation
# Split dataset
n_train = int(X.shape[0] * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# Normalize data
scaler = preprocessing.MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode trading signal with integers between 0 and n-1 classes
le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(le.classes_)


import numpy as np
import random

from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


# Deep Q-learning Agent
class Agent:

    def __init__(self, look_back, action_size, n_features):

        self.look_back = look_back          # fixed window of historical prices
        self.action_size = action_size      # buy, sell, hold
        self.n_features = n_features
        
        self.memory = deque(maxlen=3000)    # list of experiences

        self.gamma = 0.95                   # discount rate
        self.epsilon = 1.0                  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self.create_DQN()


    def create_DQN(self):
        """
        Function create_DQN to implement the deep Q-network as a MLP
        """

        # input: stock price
        # output: decision

        model = Sequential()
        model.add(Dense(30, input_shape=(self.look_back, self.n_features), activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        return model


    def remember(self, state, action, reward, next_state, done):
        """
        Function remember to store states, actions ad rewards by appending elements to the memory list
        """
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        """
        Function replay to train the deep Q-network according to the experience replay strategy
        """
        
        # Random minibatch of experiences
        mini_batch = random.sample(self.memory, batch_size)

        # Information from each experience
        for state, action, reward, next_state, done in mini_batch:

            if done:
                # End of episode, make our target reward
                target = reward

            else:
                # estimate the future discounted reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Calculate the target associated to the current state
            target_f = self.model.predict(state)
            # Update the Q-value for the action according to Belmann equation
            target_f[0][action] = target

            # Train the DQN with the state and target_t
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def act(self, state):
        """
        Function act to decide which action to take according to the epsilon-greedy policy
        """

        if np.random.rand() <= self.epsilon:
            # The agent acts at random
            return np.random.randint(self.action_size)

        # Predict the Q-values based on the current state
        act_values = self.model.predict(state)

        # The agent take the action with higher Q-value
        return np.argmax(act_values[0])

## Create and train the Agent
# Variable definiton
episodes = 25
look_back = 15
batch_size = 32
action_size = len(le.classes_)
n_features = X_train.shape[1]

# Create Agent
agent = Agent(look_back, action_size, n_features)

def run(agent, dataX, dataY, episodes, look_back):
    """
    Function run to train the agent
    """
    
    # Length of dataset
    times = len(dataX)

    # List of total rewards
    total_reward_list = []

    for ep in range(episodes):

        # print('Episode: ' + str(ep))
        
        # Initial state and position
        state = dataX[:look_back, :][np.newaxis, :, :]
        pos = dataY[look_back - 1]

        done = False
        total_reward = 0

        for t in range(1, times - look_back + 1):

            # Predict action based on the current state
            action = agent.act(state)

            # Calculate reward
            if action == pos:   # 0:-1      1:0     2:1
                reward = +1

            elif (pos == 0 or pos == 2):
                if action == 1:
                    reward = 0
                else:
                    reward = -1
            else:
                reward = -1

            total_reward += reward

            # Final state
            if t == times - look_back:
                done = True

            # Receive next state and position
            next_state = dataX[t:t + look_back, :][np.newaxis, :, :]
            next_pos = dataY[t + look_back - 1]
            
            # Remember current experience
            agent.remember(state, action, reward, next_state, done)
            
            # Make next_state the new current state; the same for pos
            state = next_state
            pos = next_pos

            if done:
                print('Episode: %i ---> Total Reward: %i' %(ep, total_reward))
                total_reward_list.append(total_reward)

            # Train the agent with previous experiences
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if (ep + 1) % 5 == 0 and ep > 0:
            file = 'AAL_robot_checkpoint' + str(ep + 1)
            # Serialize weights to HDF5
            agent.model.save_weights(file + ".h5")
            # Save epsilon
            pickle.dump(agent.epsilon, open(file + "_epsilon.pickle", "wb"))

    # Save list of rewards along the epochs
    np.savetxt(file + '_total_reward.txt', total_reward_list)

    return

# Train Agent
run(agent, X_train, y_train, episodes, look_back)

# Load rewards
total_reward_list = np.loadtxt('checkpoints/AAL_robot_checkpoint' + str(episodes) + '_total_reward.txt')
# Plot
plt.figure()
plt.plot(np.arange(1, episodes+1), total_reward_list)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()

def evaluate(agent, dataX, dataY, look_back):
    """
    Function run to evaluate the trained agent
    """
    
    # Length of dataset
    times = len(dataX)

    # Initial state and position
    state = dataX[:look_back, :][np.newaxis, :, :]
    pos = dataY[look_back - 1]

    # List of predicted positions
    pos_list = []
    
    done = False
    total_reward = 0
    
    for t in range(1, times - look_back + 1):

        # Predict action based on the current state
        action = agent.act(state)

        # Calculate reward
        if action == pos:   # 0:-1      1:0     2:1
            reward = +1

        elif (pos == 0 or pos == 2):
            if action == 1:
                reward = 0
            else:
                reward = -1
        else:
            reward = -1

        pos_list.append(action)
        total_reward += reward

        # Final state
        if t == times - look_back:
            done = True

        # Receive next state and position
        next_state = dataX[t:t + look_back, :][np.newaxis, :, :]
        next_pos = dataY[t + look_back - 1]

        # Remember current experience
        agent.remember(state, action, reward, next_state, done)

        # Make next_state the new current state; the same for pos
        state = next_state
        pos = next_pos

        if done:
            print('Total Reward: %i' % total_reward)

    return np.array(pos_list)

# Evaluate the model
# Make predictions
y_pred_test = evaluate(agent, X_test, y_test, look_back)

# Calculate and print accuracy
acc = accuracy_score(y_test[look_back-1:-1], y_pred_test)

print('Accuracy: %.2f %%' % (acc*100))

# Calculate and print precision, recall, f1 score and support
p, r, f1, s = precision_recall_fscore_support(y_test[look_back-1:-1], y_pred_test, average=None)
results = pd.DataFrame({'1-Precision': p, '2-Recall': r, '3-F1 score': f1, '4-Support': s}, index=le.classes_)

print(results.round(decimals=3))

# Decodificate labels
y_true_test = le.inverse_transform(y_test[look_back-1:-1])
y_pred_test = le.inverse_transform(y_pred_test)

# Plot strategy
plt.figure()
plt.plot(y_true_test, label='true')
plt.plot(y_pred_test, label='pred')
plt.legend()
plt.show()








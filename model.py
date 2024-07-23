import numpy as np
import pandas as pd
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# catch multithreading error
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

class SchoolPicker:
    def __init__(self, drop_columns=False):
        self.lookup = pd.read_csv('data/collegedata.csv',index_col='INSTNM')
        self.data = pd.read_csv('data/collegedata.csv', index_col='INSTNM')
        if drop_columns:
            self._drop_columns()
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.names = self.data.index.tolist()
        self.features = self.data.values
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.num_features = self.features.shape[1]
        self.num_schools = self.features.shape[0]
        self.restart()

    def _drop_columns(self):
        missing_perc = self.data.isnull().mean() * 100
        self.data = self.data.loc[:, missing_perc < 50]
        print(f"Removed some columns. Now we have {self.data.shape[1]} columns left.")
        self.features = self.data.values

    def restart(self):
        self.remaining = set(self.names)
        self.rejected = set()
        self.current_state = np.zeros(self.num_features)
        self.current_school = self._get_best_school()
        if self.current_school is not None:
            self.current_state = self.features[self.name_to_index[self.current_school]]
        return self.current_state

    def _get_best_school(self):
        valid_schools = self.remaining - self.rejected
        if not valid_schools:
            return None
        scores = [np.dot(self.features[self.name_to_index[name]], self.current_state) for name in valid_schools]
        best_school = max(valid_schools, key=lambda name: np.dot(self.features[self.name_to_index[name]], self.current_state))
        return best_school

    def make_choice(self, school_name):
        if school_name not in self.remaining:
            return self.current_state, 0, True

        index = self.name_to_index[school_name]
        reward = np.dot(self.features[index], self.current_state)
        self.remaining.remove(school_name)
        if reward < 0:
            self.rejected.add(school_name)
        done = (len(self.remaining) == 0)
        if not done:
            self.current_school = self._get_best_school()
            if self.current_school is not None:
                self.current_state = self.features[self.name_to_index[self.current_school]]
        else:
            self.current_state = np.zeros(self.num_features)
        return self.current_state, reward, done

    def information_on_current_school(self):
        school_row = self.lookup.loc[self.current_school]
        return school_row

    def display(self):
        if self.current_school is not None:
            print(f"Current pick: {self.current_school}")
        else:
            print("No more schools to show.")

class DQNStudent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2500)
        self.gamma = 0.95
        self.alpha = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, env):
        if np.random.rand() <= self.epsilon:
            return env._get_best_school()
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        best_action_index = np.argmax(q_values[0])
        return list(env.remaining)[best_action_index]

    def train(self, batch_size, env):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0))
            action_index = list(env.remaining).index(action)
            target_f[0][action_index] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def adjust_epsilon(self, got_feedback):
        if got_feedback:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            if self.epsilon < 1.0:
                self.epsilon += (1 - self.epsilon) * 0.1
       
def process_swipe(feedback, agent, state, env):
    action = agent.choose_action(state, env)
    next_state, reward, done = env.make_choice(action)
    
    if feedback == True:
        reward = 1
        got_feedback = True
        agent.remember(state, action, reward, next_state, done)
        agent.train(32, env)
        agent.adjust_epsilon(got_feedback)
        state = next_state
    elif feedback == False:
        reward = -1
        got_feedback = False
        agent.remember(state, action, reward, next_state, done)
        agent.train(32, env)
        agent.adjust_epsilon(got_feedback)
        state = next_state
    else:
        print("Invalid input. Defaulting to no.")
        reward = -1
        got_feedback = False
        agent.remember(state, action, reward, next_state, done)
        agent.train(32, env)
        agent.adjust_epsilon(got_feedback)
        state = next_state


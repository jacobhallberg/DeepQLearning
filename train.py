import numpy as np
import gym
from keras.layers import Dense, LSTM, ConvLSTM2D, Activation, Flatten
from keras.models import Sequential
from preprocessImages import preprocess
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

environment = gym.make("Pong-v0")

total_episodes = 20000
learning_rate = 10e-3
discount_rate = 0.99
batch_size = 10

# Exploitation vs Exploration params.
exploration_probability = 1
max_exploration_probability = 1
min_exploration_probability = 0.01
exploration_probability_decay_rate = 0.005

# Game Actions:
move_up = 2
move_down = 3
possible_actions = [move_up, move_down]

# Image params
image_size = (80,80,1)
memory_size = 11

def deep_q(input_shape, num_actions, loss, optimizer):
    model = Sequential()
    
    model.add(ConvLSTM2D(32, 4, input_shape=input_shape))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss=loss, optimizer=optimizer)
    return model

class ReplayMemory():
    def __init__(self, max_replay_memory_size):
        self.current_state = deque(maxlen = max_replay_memory_size)
        self.action = deque(maxlen = max_replay_memory_size)
        self.reward = deque(maxlen = max_replay_memory_size)
        self.next_state = deque(maxlen = max_replay_memory_size)
        self.done = deque(maxlen = max_replay_memory_size)
            
    def add_memory(self, current_state, reward, action, next_state, done):
        self.current_state.append(current_state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)
    
    def sample_memory(self, batch_size):
        samples = np.random.choice(np.arange(len(self.current_state)), size = batch_size, replace=False)
        
        current_states = np.array([self.current_state[sample] for sample in samples])
        rewards = np.array([self.reward[sample] for sample in samples])
        actions = np.array([self.action[sample] for sample in samples])
        next_states = np.array([self.next_state[sample] for sample in samples])
        dones = np.array([self.done[sample] for sample in samples])
        
        return current_states, rewards, actions, next_states, dones
    
def next_action(model, state, episode, possible_actions, exploration_probability, max_exploration_probability, min_exploration_probability, exploration_probability_decay_rate):
    exploration_exploitation_value = np.random.uniform(0,1)
    
    if exploration_exploitation_value > exploration_probability:
        action = np.random.choice(possible_actions)
    else:
        action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
        action = possible_actions[action]
    
    exploration_probability = min_exploration_probability + \
    (max_exploration_probability - min_exploration_probability) * \
    np.exp(-exploration_probability_decay_rate * episode)
    
    return action, exploration_probability

memory = ReplayMemory(memory_size)
model = deep_q((2,80,80,1), 2, "mse", "adam")

for episode in tqdm(range(total_episodes)):
    previous_state = preprocess(environment.reset())
    current_state, _, _, _ = environment.step(np.random.choice(possible_actions))
    current_state =  preprocess(current_state)
    current_state = np.array([previous_state, current_state])
    
    step = 0
    episode_rewards = 0
    done = False
    
    while not done:
        action, exploration_probability = next_action(model, current_state, episode, 
                                        possible_actions, exploration_probability, max_exploration_probability, 
                                        min_exploration_probability, exploration_probability_decay_rate)
        previous_state = current_state
        current_state, reward, done, _ = environment.step(action)
        current_state = preprocess(current_state)
        current_state = np.array([previous_state[-1], current_state])
        
        episode_rewards += reward
        
        # Add state to memory to be used for replay training.
        memory.add_memory(previous_state, reward, action, current_state, done)
        
        if done:
            # Replay Learning
            current_states, rewards, actions, next_states, dones = memory.sample_memory(batch_size)
            
            Q_next_state = model.predict(next_states)
            target_Qs = []
            
            for i in range(batch_size):
                if dones[i]:
                    target_Qs.append(rewards[i])
                else:
                    target_Qs.append(rewards[i] + discount_rate * np.max(Q_next_state[i]))
            target_Qs = np.array(target_Qs)
            
            print(current_states.shape, target_Qs.shape)
            model.train_on_batch(current_states, target_Qs)
        
        
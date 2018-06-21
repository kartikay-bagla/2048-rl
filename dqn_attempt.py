import Game2048
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from drawnow import drawnow

class Brain:
    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.model = self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Dense(64, activation = "relu", input_dim = self.stateCount))
        model.add(Dense(64, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation = "relu"))
        model.add(Dense(self.actionCount, activation = "softmax"))
        model.compile(loss = "mean_absolute_error", optimizer = "adam")
        return model

    def train(self, x, y, epoch = 1, verbose = 0):
        self.model.fit(x, y, batch_size=64, epochs = epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(np.array(s).reshape(1, self.stateCount)).flatten()

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
    
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

class Agent:
    MEMORY_CAPACITY = 100000
    BATCH_SIZE = 64
    GAMMA = 0.99
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    LAMBDA = 0.001

    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount
        self.brain = Brain(self.stateCount, self.actionCount)
        self.memory = Memory(self.MEMORY_CAPACITY)
        self.epsilon = self.MAX_EPSILON
        self.steps = 0

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):
        self.memory.add(sample)

        self.steps += 1
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(self.BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCount)

        states = np.array([i[0] for i in batch])
        states_ = np.array([(no_state if i[3] is None else i[3]) for i in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = np.zeros((batchLen, self.stateCount))
        y = np.zeros((batchLen, self.actionCount))

        for i in range(batchLen):
            j = batch[i]
            s = j[0]; a = j[1]; r = j[2]; s_ = j[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

class Environment:
    def __init__(self):
        self.env = Game2048.Game2048()

    def run(self, agent):
        s = self.env.reset()
        R = 0
        while True:
            a = agent.act(s)
            s_, r, done = self.env.step(a)
            if done:
                s_ = None
            agent.observe((s, a, r, s_))
            agent.replay()
            s = s_
            R += r
            if done:
                break
        return R

env = Environment()

stateCount = 16
actionCount = 4
result_list = []
def plotter():
    plt.plot(range(len(result_list)), result_list, marker=".")

plt.ion()
fig = plt.figure()

agent = Agent(stateCount, actionCount)

for i in range(1000000):
    R = env.run(agent)
    print("Iteration:", i+1, "Reward:", R)
    result_list.append(R)
    if i%5 == 0:
        drawnow(plotter)
    
    if i%1000 == 0:
        with open("logdir.txt", "a") as f:
            f.write("File Name,2048_base_"+ str(i) +"_iter.h5" + ",Result," + str(sum(result_list[-500::])/len(result_list[-500::])))
        agent.brain.model.save("2048_base_"+ str(i) +"_iter.h5")

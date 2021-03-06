import os
import sys
import pandas as pd
import random


class SinoEnv(object):
    def __init__(self, actionMoney, dataset, dataset_variation, dates, train_ratio):
        self.count = 0
        self.action_space = actionMoney
        train_len = round(len(dataset) * train_ratio)
        self.observation_space = [123]
        self.train_data = dataset[:train_len]
        self.test_data = dataset[train_len:]
        self.train_variation = dataset_variation
        self.test_variation = dataset_variation[train_len:]

        # self.variation = dataset_variation
        self.dates = dates[train_len:]
        self.TEST = False

        print("訓練筆數: ", train_len)
        print("測試筆數: ", len(dataset) - train_len)

    def reset(self):
        self.count = 0
        # self.count = random.randint(0,len(self.train_data)-1)
        self.observation_space = [self.train_data[self.count]]
        self.variation = self.train_variation
        return self.observation_space

    def reset_test(self):
        self.count = 0
        self.observation_space = [self.test_data[self.count]]
        self.variation = self.test_variation
        return self.observation_space

    def step(self, state, action):  # Calculate reward
        state = state[0]
        done = False

        nextstate = state + \
            self.variation[self.count] + self.action_space[action]
        human_reward = -225
        interest = (nextstate - 27000000) * 0.03 * 0.01 * 1/365
        risk = -(18775829 - nextstate) * 0.3 * \
            0.01 if nextstate < 18775829 else 0
        stock = -(nextstate - 27000000) * 0.3 * \
            0.01 if nextstate > 27000000 else 0

        if action == 0:
            car_reward = 0
        else:
            car_reward = -2000

        if nextstate < 0:
            reward = -10000000000
            nextstate = 1
            # print("OHHHHHHHH!")
            done = True

        elif nextstate > 60000000:
            reward = -10000000000
            nextstate = 60000000
            # print("OHHHHHHHH!")
            done = True

        # else:
        # if nextstate > 60000000 :
        #     reward = -10000000000
        # elif nextstate < 0 :
        #     reward = -10000000000
        else:
            # reward = car_reward + human_reward * self.trans_discrete_state(nextstate)*0.9 + risk + stock
            reward = car_reward + human_reward * nextstate/10000000 + risk + stock
            if(self.TEST):
                if self.action_space[action] > 0:
                    print("第", self.count, "天，提",
                          self.action_space[action], "date: ", self.dates[self.count], " !")
                elif self.action_space[action] < 0:
                    print("第", self.count, "天，解",
                          self.action_space[action], "date: ", self.dates[self.count], " !")
        self.count += 1
        nextstate = [nextstate]
        # if self.count == len(self.variation):
        #     done = True
        return nextstate, reward, action, done

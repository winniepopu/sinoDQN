import tensorflow as tf
import collections
import random
import numpy as np
import pandas as pd
import sys

GAMMA = 0.9  # discount factor for target Q
# INITIAL_EPSILON = 0.9  # starting value of epsilon
# FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 100  # experience replay buffer size
BATCH_SIZE = 64  # size of minibatch


class DQN():
    def __init__(self, env, action_num, GAMMA=0.85, EPS_END=0.001, EPS_DECAY=0.999, LEARNING_RATE=0.1):
        self.actions = np.arange(0, action_num)
        self.discrete_state = [0,10000000, 13000000, 15000000, 17846836.5, 18775829, 19000000, 20000000, 23000000, 25000000, 26312565.85,
        28000000, 29000000, 30000000, 32951914.5, 35000000, 38000000, 40000000, 42000000, 50000000, 10000000000]
        # self.Qlearning_table = pd.DataFrame(np.zeros((6, len(self.actions))),columns=self.actions,index=[0,1,2,3,4,5])
        self.gamma_reward_decay = GAMMA
        self.eps_start = 1.0
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.epsilon = self.eps_start
        self.learning_rate = LEARNING_RATE

        self.replay_buffer = collections.deque()  # 經驗池
        self.time_step = 0
        # self.state_dim = len(env.observation_space)  # 狀態維度
        self.state_dim = len(self.discrete_state)-1  # 狀態維度
        self.action_dim = len(env.action_space)  # 動作維度
        self.create_Q_network()  # 建立Q網路
        self.create_training_method()  # 健立訓練方法
        # 初始化tensorflow session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        # network weights.
        # print("self.state_dim: ", self.state_dim)
        unit1 = 100
        unit2 = 200
        W1 = self.weight_variable([self.state_dim, unit1])
        # W1 = self.weight_variable([6, unit1])
        b1 = self.bias_variable([unit1])
        W2 = self.weight_variable([unit1, unit2])
        b2 = self.bias_variable([unit2])
        W3 = self.weight_variable([unit2, self.action_dim])
        b3 = self.bias_variable([self.action_dim])

        # 輸入層
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # self.state_input = tf.placeholder("float", [None, 6])
        # 隱藏層
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        h_layer = tf.nn.relu(tf.matmul(h_layer, W2) + b2)
        keep_prob = tf.placeholder_with_default(0.8, shape=())
        h_layer = dropout(h_layer, keep_prob)

        # 輸出層
        self.Q_value = tf.matmul(h_layer, W3) + b3

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.multiply(
            self.Q_value, self.action_input), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):

        state = self.trans_discrete_state(state)
        next_state = self.trans_discrete_state(next_state)

        one_hot_state = np.zeros(self.state_dim)  # 轉成one-hot型式
        one_hot_state[state] = 1

        one_hot_nextstate = np.zeros(self.state_dim)  # 轉成one-hot型式
        one_hot_nextstate[next_state] = 1

        one_hot_action = np.zeros(self.action_dim)  # 轉成one-hot型式
        one_hot_action[action] = 1

        self.replay_buffer.append(
            (one_hot_state, one_hot_action, reward, one_hot_nextstate, done))  # 新增記憶

        # self.replay_buffer.append(
        #     (state, one_hot_action, reward, next_state, done))  # 新增記憶

        if len(self.replay_buffer) > REPLAY_SIZE:  # 限制記憶容量
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:  # 累積一些size在train
            self.train_Q_network()
        # print("self.replay_buffer.: " , self.replay_buffer)

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: 從經驗池隨機選擇minibatch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: 計算y
        y_batch = []
        # print("state_batch: ", state_batch)
        # print("next_state_batch: ", next_state_batch)
        Q_value_batch = self.Q_value.eval(
            feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
        if done:
            y_batch.append(reward_batch[i])
            # print("y_batch: ", y_batch)
            # print(minibatch)
        else:
            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    # Choose actions

    def choose_action(self, state):
        # print("state :", state)
        state = self.trans_discrete_state(state)
        one_hot_state = np.zeros(self.state_dim)  # 轉成one-hot型式
        one_hot_state[state] = 1
        a = self.Q_value.eval(feed_dict={self.state_input: [one_hot_state]})
        # print("a :", a)
        Q_value = self.Q_value.eval(
            feed_dict={self.state_input: [one_hot_state]})[0]
        r = np.random.uniform(0, 1)
        # print("## self.epsilon: ", self.epsilon)
        if  r < self.epsilon:
            # print("r: ", r)
            action = random.randint(0, self.action_dim - 1)
        else:
            action = np.argmax(Q_value)

            print("Q_value: ",Q_value)
            print("state " , state)
            print("action:　", action)
            print("-"*10)
        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

        return action

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)  # 常態分布
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)  # 常數 起始維0.01
        return tf.Variable(initial)

    def trans_discrete_state(self, todayCash):
        todayCash = pd.array(todayCash)
        # op_labels = ['0', '1', '2', '3', '4', '5']
        # category = [0, 17846836.5, 18775829, 26312565.85,
        #             30000000, 32951914.5, 10000000000]

        op_labels = np.arange(0, len(self.discrete_state)-1)


        discreted_state = pd.cut(todayCash,labels=op_labels, bins=self.discrete_state)[0]
        return [discreted_state]

    # def learn(self, state, action, reward, nextstate):
    #     state = self.trans_discrete_state(state)

    #     nextstate = self.trans_discrete_state(nextstate)

    #     q_predict = self.Qlearning_table.loc[state, action]
    #     if nextstate != 'terminal':
    #         q_target = reward + self.gamma_reward_decay * self.Qlearning_table.loc[nextstate, :].max()  # next state is not terminal
    #     else:
    #         q_target = reward  # next state is terminal

    #     self.Qlearning_table.loc[state, action] += self.learning_rate * (q_target - q_predict)


def dropout(nodes, keep_prob):
    if keep_prob == 0:
        return tf.zeros_like(nodes)

    mask = tf.random_uniform(tf.shape(nodes)) < keep_prob
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.divide(tf.multiply(mask, nodes), keep_prob)

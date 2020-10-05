import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

# np.random.seed(1)
# tf.set_random_seed(1)
EPS_END = 0.001
EPS_DECAY = 0.9999

"""state 狀態"""
# 1. 原始連續變數 ex: [24555689]
ONE_HOT_ENCODING = False
DISCRETE = False

# 2. 離散化 ex: [2]
# ONE_HOT_ENCODING = False
# DISCRETE = True

# 3. 離散化並one hot encoding ex: [0 1 0 0 0 0 0 0 0 0 0 0 0 ]
# ONE_HOT_ENCODING = True
# DISCRETE = True


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            # e_greedy_increment=None,
            output_graph=True,
    ):
        self.discrete_state = [0, 10000000, 13000000, 15000000, 17846836.5, 18775829, 19000000, 20000000, 23000000, 25000000, 26312565.85,
                               28000000, 29000000, 30000000, 32951914.5, 35000000, 38000000, 40000000, 42000000, 50000000, 10000000000] if DISCRETE == True else []
        self.n_actions = n_actions
        self.n_features = 1 if ONE_HOT_ENCODING == False else len(
            self.discrete_state)-1

        self.eps_start = 1.0
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.epsilon = self.eps_start
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # 前後觀測值維度+action+reward
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e)
                                  for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 500, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
                    0.1)  # config of layers # n_l1: 用幾個神經元  ##默認參數

            # 第一層layer: first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [
                                     self.n_features, n_l1], initializer=w_initializer, collections=c_names)  # c_name: 變量所屬集合
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 第二層layer: second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params',
                       tf.GraphKeys.GLOBAL_VARIABLES]  # 不同的collection

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        # print('state: ',s, ', action: ', a )
        s = self.trans_one_hot(
            s, self.n_features, ONE_HOT_ENCODING, DISCRETE)  # 轉成one-hot型式
        s_ = self.trans_one_hot(s_, self.n_features,
                                ONE_HOT_ENCODING, DISCRETE)  # 轉成one-hot型式

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0  # dataframe 第0行

        # transition = np.hstack((s, [a, r], s_))
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1  # 一直插入到 200 就開始替換

    def choose_action(self, observation):

        observation = self.trans_one_hot(
            observation, self.n_features, ONE_HOT_ENCODING, DISCRETE)

        # to have batch dimension when feed into tf placeholder ## 增加矩陣維度
        observation = [observation]
        # observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)

            # print("Q_value: ",actions_value)
            # print("state " , observation)
            # print("action:　", action)
            # print("-"*10)
        else:
            action = np.random.randint(0, self.n_actions)
        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

        return action

    def learn(self):
        # check to replace target parameters ## 檢查是否要換參數
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:  # 如果記憶庫沒那麼多記憶，那就抽取我們存下來的記憶

            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)

        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # fixed params #後面幾個 : s
                self.s_: batch_memory[:, -self.n_features:],
                # newest params #錢面幾個 : s_
                self.s: batch_memory[:, :self.n_features],
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + \
        #     self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def trans_discrete_state(self, todayCash):
        todayCash = pd.array(todayCash)
        # op_labels = ['0', '1', '2', '3', '4', '5']
        # category = [0, 17846836.5, 18775829, 26312565.85,
        #             30000000, 32951914.5, 10000000000]
        op_labels = np.arange(0, len(self.discrete_state)-1)
        discreted_state = pd.cut(
            todayCash, labels=op_labels, bins=self.discrete_state)[0]
        return [discreted_state]

    def trans_one_hot(self, observation, n_dim, one_hot_encoding=True, discrete=True):
        if discrete == True:
            state = self.trans_discrete_state(observation)
            if one_hot_encoding == True:
                one_hot_arr = np.zeros(n_dim)  # 轉成one-hot型式
                one_hot_arr[state] = 1
                return one_hot_arr
            else:
                return state
        else:
            return observation

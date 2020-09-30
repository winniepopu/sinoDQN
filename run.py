import collections
import numpy as np
import pandas as pd
from DQN import DQN
from env import SinoEnv
import tensorflow.compat.v1 as tf

GAMMA = 0.85
EPS_END = 0.001
EPS_DECAY = 0.999
LEARNING_RATE = 0.1
EPISODE = 100

banknum = 39
cashLimit = 30

train_ratio = 0.9

# actionMoney = [0, 10000000, -10000000] ##  for 有 3 個action時  [不動作, 提取10000000, 解繳10000000 ]
# actionMoney = [0, 10000000, 15000000, -10000000, -15000000]   ## for 有 5 個action時  [不動作, 提取10000000, 提取15000000, 解繳10000000, 解繳15000000 ]
# actionMoney = [0, 10000000, 11000000, 12000000, 13000000, -10000000, -11000000, -12000000, -13000000]
# actionMoney = [0, 6000000, 8000000, 10000000,
#                12000000, -6000000, -8000000, -10000000, -12000000]
# actionMoney = [0, 5000000, 6000000, 7000000, 8000000,-5000000, -6000000, -7000000, -8000000,]
# actionMoney = [0, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, -5000000, -6000000, -7000000, -8000000, -9000000, -10000000]
actionMoney = [0, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, -6000000, -7000000, -8000000, -9000000, -10000000, -11000000, -12000000,]


def data_preprocessing(banknum, cashLimit):
    banknum = 39
    dataset = pd.read_csv('./allbank_15-19_庫現(決策2).csv', usecols=[
                          0, 1, 2, 8, 9, 10, 11, 12, 13], encoding='UTF-8')  # usecols=[1,2,3],
    df = dataset[dataset["分行別"] == banknum]
    df = df.drop(['分行別'], axis=1)

    df['date'] = df['資料日'].astype(int) % 100
    df['資料日'] = pd.to_datetime(df['資料日'], format="%Y%m%d")
    df.set_index('資料日', inplace=True)

    # df['庫存限額'].astype('int')
    df = df[df['庫存限額'] <= cashLimit]
    df = df.fillna(value=0)
    df["前日收支變動"] = df["本日庫現餘額"] - df["前日庫現餘額"] - df["提取金額"] + df["解繳金額"]
    dataset = df['本日庫現餘額'].values
    # dataset = df[['本日庫現餘額','date']].values

    dates = df['date'].values[1:]
    dataset_variation = df['前日收支變動'].values[1:]  # 要算當天收支變動，不包含第一個值

    return dataset, dataset_variation, dates

def main():
    # 資料前處理
    dataset, dataset_variation, dates = data_preprocessing(banknum, cashLimit)
    # 定義env
    env = SinoEnv(actionMoney, dataset, dataset_variation, dates, train_ratio)
    # 建立一個DQN
    rl = DQN(env, len(actionMoney), GAMMA, EPS_END, EPS_DECAY, LEARNING_RATE)
    # 初始化Tensorflow
    tf.global_variables_initializer()

    # Training
    for episode in range(EPISODE):
        s = env.reset()  # reset env
        total_reward = 0
        print("epsilon: ", rl.epsilon)

        # while True:
        for i in range(len(env.train_data)-2):
            # 選擇動作
            a = rl.choose_action(s)
            # print("a: ", a)
            # take action and get nextstate and reward
            s_, r, a, done = env.step(s, a)
            

            # 記憶和學習
            # rl.learn(s, a, r, s_ )
            rl.perceive(s, a, r, s_, done)

            if done:
                print("爆掉了")
                # print('第', episode+1 , '回合訓練損益是:', total_reward, '元')
                # print(rl.Qlearning_table)
                # print("-"*60)
                break

            # 否則轉換 s -> s_
            s = s_

            total_reward += r
        if not done:
            print('第', episode+1, '回合訓練損益是:', total_reward, '元')
            # print(rl.Qlearning_table)
        print("-"*60)

    # Testing
    for episode in range(1):
        env.TEST = True
        s = env.reset_test()  # reset env
        # print("initial state: ", s)
        print("-"*60)
        print("訓練", EPISODE, "次")
        print("有 %s 個action," % (len(actionMoney)))
        print(actionMoney)
        print("")
        total_reward = 0
        test_len = len(env.test_variation)
        # while True:
        for i in range(test_len):

            # 選擇動作
            a = rl.choose_action(s)

            # take action and get nextstate and reward
            s_, r, a, done = env.step(s, a)

            # 記憶和學習
            # rl.learn(s, a, r, s_ )
            rl.perceive(s, a, r, s_, done)

            if done:
                print("爆掉了")
                break

            # 否則轉換 s -> s_
            s = s_

            total_reward += r

        print('第', episode+1, '回合測試損益是:', total_reward, '元')
        # print(rl.Qlearning_table)
        print("-"*60)

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape)#常態分布
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)#常數 起始維0.01
    return tf.Variable(initial)

# def test():
#     session = tf.InteractiveSession()

#     state_input = tf.placeholder("float",[None,10])
#     state = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#     # h_layer = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
#     W1 = weight_variable([10,100])
#     b1 = bias_variable([100])
#     W2 = weight_variable([100,2])
#     b2 = bias_variable([2])
#     h_layer = tf.nn.relu(tf.matmul(state_input,W1) + b1)
#     Q_value = tf.matmul(h_layer,W2) + b2

#     print(Q_value)

#     session.run(tf.global_variables_initializer())
#     a = Q_value.eval(feed_dict = {state_input:[state]})[0]

#     print(a)


    
if __name__ == '__main__':
    main()
    # test()


import pandas as pd
def cal_origin_reward():
    nextstate = state + self.variation[self.count] + self.actionMoney[action]

    human_reward = -225
    # interest = (nextstate - xx) * 0.25 * 0.01 * 1/365
    risk = -(18775829 - nextstate) * 0.3 * 0.01 if nextstate<18775829 else 0
    stock = -(nextstate - 30000000) * 0.03 * 0.01 if nextstate>30000000 else 0    
    reward = car_reward + human_reward * self.trans_discrete_state(nextstate)*0.9 + risk + stock

def data_preprocessing():
    banknum = 39
    dataset = pd.read_csv('./allbank_15-19_庫現(決策2).csv', usecols=[0,1,2,8,9,10,11,12,13], encoding='UTF-8') #usecols=[1,2,3],
    df = dataset[dataset["分行別"]==banknum]
    df = df.drop(['分行別'],axis = 1)

    df['資料日']=pd.to_datetime(df['資料日'],format="%Y%m%d")
    # df = df.drop(['資料日'],axis = 1)
    df.set_index('資料日',inplace=True)

    # df['庫存限額'].astype('int')
    df = df[df['庫存限額'] <= 30]
    df = df.fillna(value=0)
    df["前日收支變動"] = df["本日庫現餘額"] - df["前日庫現餘額"] - df["提取金額"] + df["解繳金額"]
    dataset_cash = df['本日庫現餘額'].values
    action_variation = (df["提取金額"] - df["解繳金額"]).values

    train_len = round(len(dataset_cash)*0.9)
    # train_data = dataset_cash[:train_len]
    test_data = dataset_cash[train_len:]
    test_action_variation = action_variation[train_len:]

    print("訓練筆數: ", train_len)
    print("測試筆數: ", len(dataset_cash) - train_len)

    return test_data, test_action_variation

def trans_discrete_state(todayCash):
    todayCash = pd.array([todayCash])
    op_labels = ['0', '1', '2','3','4','5']
    category = [0,17846836.5,18775829,26312565.85,30000000,32951914.5,10000000000]
    discreted_state = pd.cut(todayCash,labels=op_labels, bins=category)[0]

    return int(discreted_state)
      
test_data, test_action_variation = data_preprocessing()
print("test: ", test_data)
print("test_action_variation: ", test_action_variation)

total_reward = 0
for i in range(1,len(test_data)):
    nextstate = test_data[i]
    human_reward = -225
    # interest = (nextstate - xx) * 0.25 * 0.01 * 1/365
    risk = -(18775829 - nextstate) * 0.3 * 0.01 if nextstate<18775829 else 0
    stock = -(nextstate - 27000000) * 0.03 * 0.01 if nextstate>27000000 else 0    
    
    if test_action_variation[i] != 0:
        car_reward = -2000
        print("第",i,"天", " risk:", risk , " stock: ", stock , "動作變動: ", test_action_variation[i])
    else:
        car_reward = 0 
    reward = car_reward + human_reward * nextstate/10000000 + risk + stock
   
    total_reward += reward

print("total_reward: ",total_reward)

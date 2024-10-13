import pandas as pd
import numpy as np
import pickle
task = 'tg'
ori = "./data_pyg/prediction/{}/{}_raw_1/{}_raw_120_tk.pkl".format(task,task,task)
with open(ori, 'rb') as file:
    ori_polymer_df =pickle.load(file)
des = './data_pyg/prediction/tg/oneToten_raw_120_tk.pkl'

train_idx = pd.read_csv('./rnn/train_idx.csv', header = None).values.T[0]

x_train_list = ori_polymer_df['tk'].iloc[train_idx].values.tolist()
y_train = ori_polymer_df['tg'].iloc[train_idx]
#print(y_train)
# padding training set to maximum length
input_len = 1587
for x in x_train_list:
    x += [0] * (input_len-len(x))
#print(x_train_list)
df = pd.DataFrame({'tk': x_train_list,'tg':y_train})
#print(train_df)
#train_df = x_train_df.join(y_train)

for tr in [2,3,4,5,6,7,8,9,10]:

    file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120_tk.pkl".format(task,task,tr,task)
    # import the ori test dataset
    with open(file, 'rb') as file:
        polymer_df =pickle.load(file)

    # obtain original training tokens
    cur_list = polymer_df['tk'].iloc[train_idx].values.tolist()
    #print(y_train)
    # padding training set to maximum length
    for x in cur_list:
        x += [0] * (input_len-len(x))
    cur_df = pd.DataFrame({'tk': cur_list,'tg':y_train})
    df = pd.concat([df,cur_df],axis = 0,ignore_index=True)
        #print(train_df)
    print(len(df))
with open(des,'wb') as file:
    pickle.dump(df,file)

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

def read_npy_to_df(dir_path):
    """Reads all .npy files in a directory and concatenates them into a DataFrame."""

    groundtruth = pd.DataFrame(np.load('./rnn/tg_test.npy'),columns=['groundtruth'])
    dfs = [groundtruth]

    for file in os.listdir(dir_path):
        if file.endswith(".npy"):
            file_path = os.path.join(dir_path, file)
            array = np.load(file_path)

            df = pd.DataFrame(array)
            print(df)
            dfs.append(df)
    print(dfs)
    if dfs:
        return pd.concat(dfs, axis=1, ignore_index=True)
    else:
        return pd.DataFrame()
    
def evaluate(y):
    try:
        mae = mean_absolute_error(y_test,y)
    except:
        mae = np.nan
    y1,y2 = [],[]

    for i in range(len(y_test)):
        if y_test[i] != np.nan and y[i] and y_test[i]/y[i]>0:
            y1.append(abs(y_test[i]))
            y2.append(abs(y[i]))
    try:
        lgmae = mean_absolute_error(np.log(y1),np.log(y2))
    except:
        lgmae = np.nan
    try:
        r2 = r2_score(y_test, y)
    except:
        r2=np.nan
    return mae, lgmae,r2    
'''
res_path = './rnn/learn_res'
# Example usage
df = read_npy_to_df(res_path) 
df.to_csv('./rnn/res.csv')
print(df)'''
if __name__=='__main__':
    y_test = np.load('./rnn/tg_test.npy')
    task = 'tg'
    rep = [1,2,3,4,5,6,7,8,9,10]
    for r in rep:
        results = {
                "task": [],
                "repeat_time":[],
                "test_mae":[],
                "test_lgmae":[],
                "test_r2":[],
            }
        df = pd.DataFrame(results)
        for i in range(10):
            res_file = "./rnn/res/res_{}{}.npy".format(r,i)
            cur_res = np.load(res_file)
            mae,lgmae,r2 = evaluate(cur_res)
            new_results = {
                "task": 'tg',
                "repeat_time":r,
                "test_mae":mae,
                "test_lgmae":lgmae,
                "test_r2":r2,
            }
            df = pd.concat([df, pd.DataFrame([new_results])], ignore_index=True)
        # Calculate mean and std, and format them as "mean±std".
        summary_cols = ["repeat_time", "task"]
        df_mean = df.groupby(summary_cols).mean().round(4)
        df_std = df.groupby(summary_cols).std().round(4)

        df_mean = df_mean.reset_index()
        df_std = df_std.reset_index()
        df_summary = df_mean[summary_cols].copy()
        # Format 'train', 'valid' columns as "mean±std".
        for metric in ['r2','mae','lgmae']:
            col_name = 'test_'+metric
            df_summary[col_name] = df_mean[col_name].astype(str) + "±" + df_std[col_name].astype(str)

        # Save and print the summary DataFrame.
        res_csv_name = "./rnn/result_summary_tg.csv"          
        if os.path.exists(res_csv_name):
            df_summary.to_csv(res_csv_name, mode="a", header=False, index=False)
        else:
            df_summary.to_csv(res_csv_name, index=False)
        print(df_summary)

            
                    


        

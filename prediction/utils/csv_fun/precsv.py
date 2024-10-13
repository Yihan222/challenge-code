import pandas as pd
import csv

df = pd.read_csv('../hyperparameters_summary.csv')
params = df[df['model_type']=='gin-virtual']
params = params[params['task']=='CO2']
bat = params['batch_size']
print(int(bat))
print(len(params['drop_ratio']))
    

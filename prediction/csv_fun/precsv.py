import pandas as pd
df = pd.read_csv('data_dev_co2.csv')
df.dropna(axis=0, how='any', inplace=True)
#print(df[200:250])

df.to_csv('CO2_raw.csv', index=False)
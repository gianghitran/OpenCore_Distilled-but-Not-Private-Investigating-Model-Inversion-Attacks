import pandas as pd

df = pd.read_csv('data/iotid20.csv')

sample_df = df.sample(n=100, random_state=42)

sample_df.to_csv('data/iotid20_sample_100.csv', index=False)
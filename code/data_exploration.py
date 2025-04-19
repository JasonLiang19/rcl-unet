import pandas as pd 
from data_loading import read_train_csv
import pickle

filepath = '../data/Uniprot Test Set.csv'

# df = pd.read_csv(filepath)
# print(len(df))
# print(len(df.dropna(subset=['rcl_seq'])))

# lengths = df.dropna(subset=['rcl_seq'])['Sequence'].str.len()
# print(lengths.describe())

# longest = lengths.nlargest(10)
# print(longest)

data_dict = read_train_csv(filepath, encoding='prottrans')
print(len(data_dict))
with open("../data/test_data.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print(data_dict)
# for i, residue in enumerate(data_dict['A0A0N8ERZ9']['sequence']):
#     print(f"{i}: {residue} {data_dict['A0A0N8ERZ9']['label'][i]}")
import pandas as pd
import os
import numpy as np

'''
make eSNLI data from .txt files
'''

data_path = 'data/eSNLI_full/'

train_path = os.path.join(data_path,'train.txt')
dev_path = os.path.join(data_path,'dev.txt')
test_path = os.path.join(data_path,'test.txt')

save_train_path = os.path.join(data_path,'train.csv')
save_dev_path = os.path.join(data_path,'dev.csv')
save_test_path = os.path.join(data_path,'test.csv')

exp_cols = ['explanation%d' % d for d in range(1,4)]
exp_col = ['explanation']
names = ['unique_key','label','premise','hypothesis']

train_df = pd.read_csv(train_path, sep = '\t', names = names+exp_col)
dev_df = pd.read_csv(dev_path, sep = '\t', names = names+exp_cols)
test_df = pd.read_csv(test_path, sep = '\t', names = names+exp_cols)

print(f"Saving data: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test")
train_df = train_df.to_csv(save_train_path, index = False)
dev_df = dev_df.to_csv(save_dev_path, index = False)
test_df = test_df.to_csv(save_test_path, index = False)
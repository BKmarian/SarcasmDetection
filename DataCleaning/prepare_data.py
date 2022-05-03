#%% md

# Introduction

We've used SARC dataset for sarcasm detection. <br>
Data source: https://www.kaggle.com/danofer/sarcasm

#%%

import pandas as pd
from sklearn.model_selection import train_test_split

#%%

data = pd.read_csv('data/raw/train-balanced-sarcasm.csv')

#%%

data.shape

#%%

data['label'].value_counts()

#%%

data_pos = data[data['label'] == 1]
data_neg = data[data['label'] == 0]

#%%

train_last_pos = 400000
cv_last_pos = 450000
test_last_pos = 500000

data_pos_tr = data_pos.iloc[:cv_last_pos]

data_pos_cv = data_pos_tr.iloc[train_last_pos:cv_last_pos]
data_pos_tr = data_pos_tr.iloc[:train_last_pos]

data_pos_test = data_pos.iloc[cv_last_pos:test_last_pos]

data_neg_tr = data_neg.iloc[:cv_last_pos]

data_neg_cv = data_neg_tr.iloc[train_last_pos:cv_last_pos]
data_neg_tr = data_neg_tr.iloc[:train_last_pos]

data_neg_test = data_neg.iloc[cv_last_pos:test_last_pos]

#%%

data_tr = pd.concat([data_pos_tr, data_neg_tr])
data_cv = pd.concat([data_pos_cv, data_neg_cv])
data_test = pd.concat([data_pos_test, data_neg_test])

#%%

print('Shape of train data: {}'.format(data_tr.shape))
print('Shape of CV data: {}'.format(data_cv.shape))
print('Shape of test data: {}'.format(data_test.shape))

#%%

data_tr.to_csv('data/raw/train.csv', index=False)
data_cv.to_csv('data/raw/cv.csv', index=False)
data_test.to_csv('data/raw/test.csv', index=False)

#%%



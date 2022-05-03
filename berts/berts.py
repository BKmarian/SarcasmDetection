#%%



#%%

!pip install simpletransformers -U --quiet
!pip install pytorch --quiet
!pip install Cython --quiet
#!pip uninstall setuptools
#!pip install setuptools==59.5.0
import os
import random
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import torch

from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import KFold

#%%

seed = 1236

f = open("results.txt", "a")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#%%

train = pd.read_csv('../input/rawdata/raw/train.csv')
cv = pd.read_csv('../input/rawdata/raw/cv.csv')
test = pd.read_csv('../input/rawdata/raw/test.csv')

train['comment'] = train['comment'].astype(str)
cv['comment'] = cv['comment'].astype(str)
test['comment'] = test['comment'].astype(str)

train = train.append(cv)
train_data = train[['comment', 'label']].sample(n = 50000)
test_data = test[['comment', 'label']].sample(n = 5000)
test_data['comment'] = test_data['comment'].astype(str)


#%%

custom_args = {'fp16': False, # not using mixed precision
               'train_batch_size': 1, # default is 8
               'gradient_accumulation_steps': 30,
               'do_lower_case': True,
               'max_seq_length':100,
               'learning_rate': 1e-05, # using lower learning rate
               'overwrite_output_dir': True, # important for CV
               'num_train_epochs': 2} # default is 1

# #%% md

# # 5-Fold CV

# #%%

# # n=5
# # kf = KFold(n_splits=n, random_state=seed, shuffle=True)
# # results = []
# #
# # for train_index, val_index in kf.split(train_data):
# #     train_df = train_data.iloc[train_index]
# #     val_df = train_data.iloc[val_index]
# #
# #     model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args)
# #     model.train_model(train_df)
# #     result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=sklearn.metrics.accuracy_score)
# #     print(result['acc'])
# #     results.append(result['acc'])

# #%%

# # for i, result in enumerate(results, 1):
# #     print(f"Fold-{i}: {result}")
# #
# # print(f"{n}-fold CV accuracy result: Mean: {np.mean(results)} Standard deviation:{np.std(results)}")

# #%% md

# # Full Training

#%%

model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args)
model.train_model(train_data)

predictions, _ = model.predict(test_data['comment'].tolist())
f.write('bert-base-uncased')
f.write(str(accuracy_score(test_data['label'], predictions)))
f.write(str(f1_score(test_data['label'], predictions)))

#%%

roberta = ClassificationModel("roberta", "roberta-base", args=custom_args)
roberta.train_model(train_data)


predictions,_ = roberta.predict( test_data['comment'].tolist())
f.write('Roberta')
f.write(str(accuracy_score(test_data['label'], predictions)))
f.write(str(f1_score(test_data['label'], predictions)))

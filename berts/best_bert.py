!pip install simpletransformers -U
# !pip install pytorch
# !pip install Cython

import warnings
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
#warnings.filterwarnings('ignore')

f = open("best-results.txt", "a")

seed = 1256
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# %%

train = pd.read_csv('../input/rawdata/raw/train.csv')
cv = pd.read_csv('../input/rawdata/raw/cv.csv')
test = pd.read_csv('../input/rawdata/raw/test.csv')

train['comment'] = train['comment'].astype(str)
cv['comment'] = cv['comment'].astype(str)
test['comment'] = test['comment'].astype(str)

train = train.append(cv)
train = train.sample(400000)
test = test.sample(40000)

train_data = train[['comment', 'label']]
test_data = test[['comment', 'label']]
train_data['comment'] = train_data['comment'].astype(str)
test_data['comment'] = test_data['comment'].astype(str)
test_data['label'] = test_data['label'].astype(str)


train_data.columns = ["text", "labels"]
test_data.columns = ["text", "labels"]
test_data.to_csv('test_l.csv', sep="\t",index = False)
train_data.to_csv('train_l.csv', sep="\t",index = False)


custom_args = {'fp16': False,  # not using mixed precision
               'train_batch_size': 2,
               'eval_batch_size': 2,
               'gradient_accumulation_steps': 30,
               'do_lower_case': True,
               'max_seq_length': 128,
               'learning_rate': 1e-05,  # using lower learning rate
               'overwrite_output_dir': True,  # important for CV
               "use_early_stopping": True,
               "early_stopping_delta": 0.01,
               "early_stopping_metric": "acc",
               "early_stopping_metric_minimize": False,
               "early_stopping_patience": 3,
               'num_train_epochs': 3,
               #"wandb_project": "bert-sarcasm",
               #"silent": True,
               "lazy_loading": True,
               "save_model_every_epoch": True,
               "save_eval_checkpoints": False
               }

roberta = ClassificationModel("roberta", "roberta-base",num_labels=2, args=custom_args, use_cuda=True)
#for t in train_data_batches:
print('Started Training')
roberta.train_model('/kaggle/working/train_l.csv')

print('Started Testing')
predictions, model_out = roberta.predict(test_data['text'].tolist())
test_data['labels'] = test_data['labels'].astype(int)
f.write(str(accuracy_score(test_data['labels'], predictions)))
f.write(str(f1_score(test_data['labels'], predictions)))
f.close()

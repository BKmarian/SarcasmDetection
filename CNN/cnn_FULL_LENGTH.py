#%%



#%%

!pip install pydotplus
%load_ext autoreload
%autoreload 2

#%%

import pickle
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus as pyd
import seaborn as sns
import tensorflow as tf
from gensim.models import KeyedVectors
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Flatten, Embedding, Conv1D, MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings('ignore')

%matplotlib inline

#%%

train = pd.read_csv('../input/sarcasm/data/clean/train.csv')
cv = pd.read_csv('../input/sarcasm/data/clean/cv.csv')
test = pd.read_csv('../input/sarcasm/data/clean/test.csv')

train.head()

#%%

train['comment'] = train['comment'].astype(str)
cv['comment'] = cv['comment'].astype(str)
test['comment'] = test['comment'].astype(str)

train['author'] = train['author'].astype(str)
cv['author'] = cv['author'].astype(str)
test['author'] = test['author'].astype(str)

#%%

t = Tokenizer()
t.fit_on_texts(train['comment'].values)
vocab_size = len(t.word_index) + 1
print(vocab_size)

#%%

encoded_comments_train = t.texts_to_sequences(train['comment'])
encoded_comments_cv = t.texts_to_sequences(cv['comment'])
encoded_comments_test = t.texts_to_sequences(test['comment'])

#%%

max_length = 512
padded_comments_train = pad_sequences(encoded_comments_train, maxlen=max_length, padding='post')
padded_comments_cv = pad_sequences(encoded_comments_cv, maxlen=max_length, padding='post')
padded_comments_test = pad_sequences(encoded_comments_test, maxlen=max_length, padding='post')

#%%

y_train = train['label'].values
y_cv = cv['label'].values
y_test = test['label'].values

y_train = to_categorical(y_train, num_classes=2)
y_cv = to_categorical(y_cv, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#%%

w2v_model = KeyedVectors.load_word2vec_format('../input/wordvec/GoogleNews-vectors-negative300.bin', binary=True)

#%%

# create a weight matrix for words in training docs
embedding_matrix_w2v = np.zeros((vocab_size, 300))
for word, i in t.word_index.items():
 try:
  embedding_vector = w2v_model[word]
 except:
  embedding_vector = [0]*300

 if embedding_vector is not None:
  embedding_matrix_w2v[i] = embedding_vector

embedding_matrix_w2v.shape

#%%

reduce_lr = ReduceLROnPlateau(monitor='val_f1_m',
                              mode = 'max',
                              factor=0.5,
                              patience=5,
                              min_lr=0.0001,
                              verbose=10)

checkpoint = ModelCheckpoint("model_01.h5",
                             monitor="val_f1_m",
                             mode="max",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_f1_m',
                          mode="max",
                          min_delta = 0,
                          patience = 5,
                          verbose=1)

#%% md

## Model 1: Baseline

#%%

input_data = Input(shape=(max_length,), name='main_input')
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix_w2v], trainable=False)(input_data)
conv_1 = Conv1D(filters=50, kernel_size=4, activation='relu')(embedding_layer)
max_1 = MaxPooling1D(pool_size=2)(conv_1)
conv_2 = Conv1D(filters=100, kernel_size=3, activation='relu')(max_1)
max_2 = MaxPooling1D(pool_size=2)(conv_2)
flatten = Flatten()(max_2)
dense = Dense(100, activation='relu', name='fully_connected')(flatten)
out = Dense(2, activation='softmax')(dense)

model_01 = Model(inputs=[input_data], outputs=[out])

print(model_01.summary())

#%%

#tensorboard = TensorBoard(log_dir='model_01_{}_{}_{}_{}'.format(filters1,filters2,kernel1,kernel2))

#%%

c = tf.keras.optimizers.Adam(lr = 0.0001)
model_01.compile(optimizer=c, loss='categorical_crossentropy', metrics=[f1_score, accuracy_score])

h1 = model_01.fit(padded_comments_train, y_train,
                  batch_size=64,
                  epochs=50,
                  verbose=1, callbacks=[ checkpoint, earlystop, reduce_lr],  #tensorboard
                  validation_data=(padded_comments_cv, y_cv))

#%%

score_1 = model_01.evaluate(padded_comments_test, y_test)
print(score_1)

#%%

cnf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model_01.predict(padded_comments_test), axis=1))

print(cnf_mat)
sns.heatmap(cnf_mat, annot=True, fmt='g', linewidths=.5, xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

#%%

plt.plot(h1.history['f1_m'][1:])
plt.plot(h1.history['val_f1_m'][1:])
plt.title('Model metric')
plt.ylabel('F1 metric')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.savefig('CNN_f1_50_4_100_3_512_length.png')
plt.clf()

#plt.show()

#%%

plt.plot(h1.history['loss'][1:])
plt.plot(h1.history['val_loss'][1:])
plt.title('Model Los')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.savefig('CNN_f1_50_4_100_3_512_length.png')
plt.clf()

#plt.show()


#%%

!cd /kaggle/working
!ls -l

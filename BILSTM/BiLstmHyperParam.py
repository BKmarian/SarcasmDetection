#%%



#%%

!pip install pydotplus
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
sns.set()
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Bidirectional,GRU
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import  Flatten
from sklearn.metrics import  confusion_matrix,f1_score,accuracy_score
warnings.filterwarnings('ignore')

#%%

train = pd.read_csv('data/clean/train.csv')
cv = pd.read_csv('data/clean/cv.csv')
test = pd.read_csv('data/clean/test.csv')

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

max_sent_length = 70
padded_comments_train = pad_sequences(encoded_comments_train, maxlen=max_sent_length, padding='post')
padded_comments_cv = pad_sequences(encoded_comments_cv, maxlen=max_sent_length, padding='post')
padded_comments_test = pad_sequences(encoded_comments_test, maxlen=max_sent_length, padding='post')

#%%

y_train = train['label'].values
y_cv = cv['label'].values
y_test = test['label'].values

y_train = to_categorical(y_train, num_classes=2)
y_cv = to_categorical(y_cv, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#%%

w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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

checkpoint = ModelCheckpoint("lstm_model.h5",
                             monitor="val_f1_m",
                             mode="max",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_f1_m',
                          mode="max",
                          min_delta = 0,
                          patience = 5,
                          verbose=1)

#%%

#print(lstm_model.summary())

for SIZE1 in [300]:
 # for SIZE2 in [50,100,200,300]:
 input_data = Input(shape=(max_sent_length,), name='main_input')
 embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix_w2v], trainable=False)(input_data)
 dropout_1 = Dropout(0.3)(embedding_layer)
 bilstm = Bidirectional(LSTM(SIZE1))(dropout_1)
 dropout_2 = Dropout(0.3)(bilstm)
 # gru = Bidirectional(LSTM(SIZE2))(dropout_2)
 # dropout_3 = Dropout(0.3)(gru)
 max_2 = Flatten()(dropout_2)
 flatten = Flatten()(max_2)
 out = Dense(2, activation='softmax', name='fully_connected')(flatten)

 lstm_model = Model(inputs=[input_data], outputs=[out])

 # print(lstm_model.summary())

 #keras.utils.vis_utils.pydot = pyd
 #plot_model(lstm_model, to_file='lstm_model.png')

 #%%

 lstm_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1_score,accuracy_score])

 lstm_model_h1 = lstm_model.fit(padded_comments_train, y_train,
                                batch_size=64,
                                epochs=50,
                                verbose=1, callbacks=[checkpoint, earlystop, reduce_lr], #tensorboard
                                validation_data=(padded_comments_cv, y_cv))

 #%%

 print(padded_comments_cv)
 print(y_cv)

 #%%

 score_1 = lstm_model.evaluate(padded_comments_test, y_test)
 print(score_1)

 #%%

 #         cnf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(lstm_model.predict(padded_comments_test), axis=1))

 #         print(cnf_mat)
 #         sns.heatmap(cnf_mat, annot=True, fmt='g', linewidths=.5, xticklabels=['Predicted 0', 'Predicted 1'],
 #                     yticklabels=['Actual 0', 'Actual 1'])

 #%%

 plt.plot(lstm_model_h1.history['f1_m'][1:])
 plt.plot(lstm_model_h1.history['val_f1_m'][1:])
 plt.title('Model metric')
 plt.ylabel('F1 metric')
 plt.xlabel('epoch')
 plt.legend(['train','Validation'], loc='upper left')
 plt.savefig('bilstm_f1_{}.png'.format(SIZE1))
 #plt.show()
 plt.clf()
 #%%

 plt.plot(lstm_model_h1.history['loss'][1:])
 plt.plot(lstm_model_h1.history['val_loss'][1:])
 plt.title('Model Los')
 plt.ylabel('Loss')
 plt.xlabel('epoch')
 plt.legend(['train','Validation'], loc='upper left')
 plt.savefig('Bilstm_Loss_{}.png'.format(SIZE1))
 #plt.show()
 plt.clf()
 #%%



#%%





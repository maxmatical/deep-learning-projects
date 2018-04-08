import pandas as pd
import numpy as np

# preprocessing tools for text and sequences
from keras.preprocessing.text import Tokenizer, one_hot, hashing_trick
from keras.preprocessing.sequence import pad_sequences

# loading keras core layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Flatten, GRU

# optimize model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.initializers import glorot_normal, he_normal
from keras.optimizers import SGD

# loading sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('nlp_train.csv', header=0)
comments= df.iloc[:,1]
max_words = 1000
t = Tokenizer(num_words=max_words)
t.fit_on_texts(comments)

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(comments, mode='count')
print(encoded_docs)

encoded_docs = pad_sequences(encoded_docs,maxlen=max_words)
np.shape(encoded_docs)
type(encoded_docs)

encoded_seq = tokenizer.text_to_sequences(comments)
encoded_seq = pad_sequences(encoded_seq)

#labels
y1 = df.iloc[:,2]
y1 = np.asarray(y1)


#train val split
xt, xv, yt, yv = train_test_split(encoded_docs, y1, test_size = 0.2)


# defining parameters for neural network
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

timestep = 10
data_dim = 1000 # place holder value
dim_per_timestep = int(data_dim/timestep)

# reshape data for lstm
xt_lstm = xt.reshape(len(xt), timestep, dim_per_timestep)
np.shape(xt_lstm)
xv_lstm = xv.reshape(len(xv), timestep, dim_per_timestep)

# reshape data for conv1d
xt_conv = np.expand_dims(xt, axis=2)
np.shape(xt)
xv_conv = np.expand_dims(xv, axis=2)

######################################
# 
# fitting 1st model (1 layer)
#
#####################################

model = Sequential()
model.add(LSTM(32, kernel_initializer='glorot_normal', input_shape = (timestep, dim_per_timestep),
            kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
#model.add(Flatten())
#fully connected layer
model.add(Dense(256, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01),
                activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(xt_lstm, yt, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size = 64)

              
              
######################################
# 
# 2nd model: using  CNN
#
#####################################

#######
# Note for Conv1D/LSTM: rehape data to (n, dim, 1)
# And use input shape as input_shape = (dim, 1)
#######

model = Sequential()
model.add(Conv1D(256, 5, input_shape = (data_dim, 1), kernel_initializer='glorot_normal',
                  kernel_regularizer=regularizers.l2(0.01), activation='relu' ))
model.add(Conv1D(256, 3, kernel_initializer='glorot_normal',
                  kernel_regularizer=regularizers.l2(0.01), activation='relu' ))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Flatten()) # need the flatten layer to convert to 1D
#fully connected layer
model.add(Dense(256, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01),
                activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(xt_conv, yt, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size = 64)


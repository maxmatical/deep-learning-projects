import pandas as pd
import numpy as np
from numpy import inf


from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import plot_model


# importing keras core layers
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization

# importing tools to optimize model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.initializers import glorot_normal, he_normal
from keras.optimizers import SGD

# defining parameters for neural network
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
filepath="model2weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


######
x_train = pd.read_csv('train.csv', header = 0)
y_train = pd.read_csv('labels.csv', header = 0)
x_test = pd.read_csv('test.csv', header = 0)

x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test= np.asarray(x_test).astype('float32')

log_x_train = np.log(x_train)#
log_x_train[log_x_train== -inf] = 0

log_x_test = np.log(x_test)#
log_x_test[log_x_test== -inf] = 0


#test
np.isinf(log_x_train)
np.isinf(log_x_test)

# splitting into training and validation data
xt, xv, yt, yv = train_test_split(log_x_train, y_train, test_size = 0.2)

n_cols = np.shape(xt)[1]
n_rows = np.shape(xt)[0]

#yt = np.ravel(yt)
#yv = np_utils.to_categorical(yv, 2)

# def plot history function
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()



######################################
# 
# fitting 1st model (1 dense layer)
#
#####################################
model1 = Sequential()
#dense layer
model1.add(Dense(64, input_dim = 23, activation='relu'))

model1.add(Dense(1, activation='sigmoid')) 

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist1 = model1.fit(xt, yt,validation_split=0.2, callbacks=callbacks_list,
            epochs=20, batch_size = 32)



            
                
plot_model_history(hist1)
scores1 = model1.evaluate(xv, yv, verbose=1)
print(scores1)
model1.predict(xt)
 # 0.78909090952439742 using early stopping
 
 # load weights
model1.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores1 = model1.evaluate(xv, yv, verbose=1)
print(scores1)

#visualizing model1
plot_model(model1, to_file='model.png')


######################################
# 
# Model 2 - Multiple FF layers
#
#####################################
model1 = Sequential()
#dense layer
model1.add(Dense(32, input_dim = 23, activation='relu', 
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(32, input_dim = 23, activation='relu',
                     kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(32, input_dim = 23, activation='relu',
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(32, input_dim = 23, activation='relu',
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid')) 

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist1 = model1.fit(xt, yt,validation_split=0.2, callbacks=[early_stopping],
            epochs=30, batch_size = 32)
            
                
# plot_model_history(hist1)
scores1 = model1.evaluate(xv, yv, verbose=1)
print(scores1)


######################################
# 
# Model 3 - Multiple configs
#
#####################################
model1 = Sequential()
#dense layer
model1.add(Dense(128, input_dim = 23, activation='selu', 
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(64,  activation='selu',
                     kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
# model1.add(Dense(32,  activation='selu',
#                     kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
# model1.add(Dropout(0.5))
model1.add(Dense(16,  activation='selu',
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(16,  activation='selu',
                     kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
# model1.add(Dense(32,  activation='selu',
#                      kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
# model1.add(Dropout(0.5))
model1.add(Dense(64,  activation='selu',
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(128, activation='selu',
                    kernel_initializer= 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid')) 

model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist1 = model1.fit(xt, yt,validation_split=0.2, callbacks=callbacks_list,
            epochs=30, batch_size = 32)
            
                
plot_model_history(hist1)


scores1 = model1.evaluate(xv, yv, verbose=1)
print(scores1)

 # load weights
model1.load_weights("model2weights.best.hdf5")
# Compile model (required to make predictions)
model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
scores1 = model1.evaluate(xv, yv, verbose=1)
print(scores1)



from keras.datasets import cifar10
import numpy as np

from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# importing keras core layers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization

# importing tools to optimize model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.initializers import glorot_normal, he_normal
from keras.optimizers import SGD
# importing data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_train, img_channels, img_rows, img_cols =  x_train.shape
num_test, _, _, _ =  x_test.shape
num_classes = len(np.unique(x_train))

#preprocess image data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# convert class labels to binary class labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# defining parameters for neural network
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

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
# fitting 1st model (1 conv layer + 1 dense layer)
#
#####################################
model1 = Sequential()
model1.add(Conv2D(32, (3,3), padding='same', input_shape = (img_channels, img_rows, img_cols),
                    activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.5))

#dense layer
model1.add(Flatten())
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes, activation='softmax')) 

model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
hist1 = model1.fit(x_train, y_train,validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size = 128)


# evaluating model
plot_model_history(hist1)
scores1 = model1.evaluate(x_test, y_test, verbose=1)
print(scores1)

# 0.4258 test set accuracy after 1 epoch 
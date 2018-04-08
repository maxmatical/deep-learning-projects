from keras.datasets import cifar10
import numpy as np

# utilities
from keras.utils import plot_model
from keras.utils import np_utils

# import functional API
from keras.models import Model

# core layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


# importing tools to optimize model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.initializers import glorot_normal, he_normal
from keras.optimizers import SGD

# defining model checking (for final model training)
filepath="modelweights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# defining early stopping (for model prototyping)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# defining SGD (for final model training)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_train, img_rows, img_cols, img_channels =  x_train.shape
num_test, _, _, _ =  x_test.shape
num_classes = len(np.unique(y_train))

#preprocess image data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# convert class labels to binary class labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


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
    
    
    
########################
#
# Model
#
########################

# input layer
input_layer = Input(shape=(img_rows, img_cols, img_channels))

# first conv block
conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
batchnorm1 = BatchNormalization()(conv1)
conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(batchnorm1)
batchnorm1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D((2, 2))(batchnorm1)
flat1 = Flatten()(pool1)

# second conv block
conv2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
batchnorm2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')(batchnorm2)
batchnorm2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D((2, 2))(batchnorm2)
flat2 = Flatten()(pool1)

# merge blocks
merge = concatenate([flat1, flat2])

# dense layer
hidden1 = Dense(32, activation='relu')(merge)

# prediction output
output_layer = Dense(num_classes, activation='softmax')(hidden1)

model = Model(inputs=input_layer, outputs=output_layer)

#visualizing model
# summarize layers
print(model.summary())

# compiling model
model.compile(optimizer=sgd,  loss='categorical_crossentropy', metrics=['accuracy'])

# training model
hist1 = model.fit(x_train, y_train,validation_split=0.2, callbacks=[early_stopping],
            epochs=2, batch_size = 64)


# Install packages on Rodeo (don't need pip3)
! pip install pandas


# deep-learning-projects

Notes on reshaping data for conv/rnn layers

# Sample
timestep = 10
data_dim = 1000 # place holder value
dim_per_timestep = int(data_dim/timestep)

# reshape data for lstm
xt_lstm = xt.reshape(len(xt), timestep, dim_per_timestep)
np.shape(xt_lstm)
xv_lstm = xv.reshape(len(xv), timestep, dim_per_timestep)

RNN Input layer
model.add(LSTM(32, kernel_initializer='glorot_normal', input_shape = (timestep, dim_per_timestep),
            kernel_regularizer=regularizers.l2(0.01)))
...
hist = model.fit(xt_lstm, yt, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size = 64)

# reshape data for conv1d
xt_conv = np.expand_dims(xt, axis=2)
np.shape(xt)
xv_conv = np.expand_dims(xv, axis=2)

Note for Conv1D/LSTM: rehape data to (n, dim, 1)
And use input shape as input_shape = (dim, 1)

model.add(Conv1D(256, 5, input_shape = (data_dim, 1), kernel_initializer='glorot_normal',
                  kernel_regularizer=regularizers.l2(0.01), activation='relu' ))
...
hist = model.fit(xt_conv, yt, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size = 64)
            
            
# Conv2d 
input_layer = Input(shape=(img_rows, img_cols, img_channels))

data in shape(n_samples, img_rows, img_cols, img_channels) <- 4D tensor

# pooling layers (upsampling) Conv1D
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
- pool_size: Integer, size of the max pooling windows. (Sliding window of size 2)
- strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.

# pooling layers (upsampling) Conv2D
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
- pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.
- strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.



# Padding/Stride (Conv layers)
strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
padding: One of "valid", "causal" or "same" (case-insensitive). "valid" means "no padding". "same" results in padding the input such that the output has the same length as the original input. 

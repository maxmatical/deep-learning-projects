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

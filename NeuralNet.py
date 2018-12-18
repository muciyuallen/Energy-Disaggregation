import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2

import sys
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y)) 

def build_nn_model(layers):
    nn_model = Sequential()
    for i in range(len(layers)-1):
        nn_model.add( Dense(input_dim=layers[i], output_dim= layers[i+1]) )#, W_regularizer=l2(0.1)) )
        nn_model.add( Dropout(0.5) )
        if i < (len(layers) - 2):
            nn_model.add( Activation('relu') )
    print(nn_model.summary())
    return nn_model

app = 'refrigerator_5'
path = 'train_test_data/'
#read data
df_train = pd.read_csv(path+'H1_train.csv')
df_val = pd.read_csv(path+'H1_val.csv')
df_test = pd.read_csv(path+'H1_test.csv')

print(df_test)
sys.exit()
X_train1 = df_train[['mains_1','mains_2']].values
y_train1 = df_train[app].values
X_val1 = df_val[['mains_1','mains_2']].values
y_val1 = df_val[app].values
X_test1 = df_test[['mains_1','mains_2']].values
y_test1 = df_test[app].values

#setting parameters
input_dim = 2
layer1_dim = 64
layer2_dim = 128
output_dim = 1

#Constructing Neural Network
layers = [input_dim, layer1_dim, layer2_dim, output_dim]
fc_model_1 = build_nn_model(layers)

adam = Adam(lr = 0.001)
fc_model_1.compile(loss='mean_squared_error', optimizer=adam)

#hold place forr time.time()
checkpointer = ModelCheckpoint(filepath="./fc_refrig_h1_2_new.hdf5", verbose=0, save_best_only=True)
hist_fc_1 = fc_model_1.fit( X_train1, y_train1,
                    batch_size=512, verbose=1, nb_epoch=100,
                    validation_split=0.33, callbacks=[checkpointer])

# fc_model_1 = load_model('fc_refrig_h1_2.hdf5')
pred_fc_1 = fc_model_1.predict(X_test1).reshape(-1)
mse_loss_fc_1 = mse_loss(pred_fc_1, y_test1)
mae_loss_fc_1 = mae_loss(pred_fc_1, y_test1)
print('Mean square error on test set: ', mse_loss_fc_1)
print('Mean absolute error on the test set: ', mae_loss_fc_1)

plot_each_app(df1_test, dates[1][17:], pred_fc, y_test[:,2], 
              'Real and predict Refrigerator on 6 test day of house 1', look_back = 50)


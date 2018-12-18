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
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#functions
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

def plot_each_app(df, num_date, predict, y_test, title, look_back = 0):
    num_date = len(dates)
    fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
    plt.suptitle(title, fontsize = '25')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    for i in range(num_date):
        if i == 0: l = 0
        ind = df.ix[dates[i]].index[look_back:]
        axes.flat[i].plot(ind, y_test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'True value')
        axes.flat[i].plot(ind, predict[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Predicted value')
        axes.flat[i].legend()
        l = len(ind)

app =  'refrigerator_5'
path = 'train_test_data/'

#read data
df_train = pd.read_csv(path+'H1_train.csv')
df_val = pd.read_csv(path+'H1_val.csv')
df_test = pd.read_csv(path+'H1_test.csv')

#turn first column into date_time index
df_list = [df_train, df_val, df_test]
for df in df_list:
    df['timestamp'] = pd.to_datetime(df['Unnamed: 0'])
    df = df.set_index('timestamp')

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
nn_model = build_nn_model(layers)

adam = Adam(lr = 0.001)
nn_model.compile(loss='mean_squared_error', optimizer=adam)
nn_model.fit( X_train1, y_train1,
                    batch_size=512, verbose=1, nb_epoch=20,
                    validation_split=0.33)

#hold place forr time.time()
# # checkpointer = ModelCheckpoint(filepath="./fc_refrig_h1_2_new.hdf5", verbose=0, save_best_only=True)
# hist_fc_1 = nn_model.fit( X_train1, y_train1,
#                     batch_size=512, verbose=1, nb_epoch=100,
#                     validation_split=0.33)

# nn_model = load_model('fc_refrig_h1_2_new.hdf5')
pred_fc_1 = nn_model.predict(X_test1).reshape(-1)
mse_loss_fc_1 = mse_loss(pred_fc_1, y_test1)
mae_loss_fc_1 = mae_loss(pred_fc_1, y_test1)
print('Mean square error on test set: ', mse_loss_fc_1)
print('Mean absolute error on the test set: ', mae_loss_fc_1)


output_path = 'results/'
#output the test results of the model on same house
SH_result_df = pd.DataFrame()
SH_result_df['time'] = df_test['Unnamed: 0']
SH_result_df['y_true'] = y_test1
SH_result_df['y_pred'] = pred_fc_1


SH_result_df.to_csv(output_path+'H1_OS_%s_results.csv'%app)

app2 = 'refrigerator_9'
#apply the trained model on different house
df2 = pd.read_csv(path + 'H2_test_full.csv')
df2['timestamp'] = pd.to_datetime(df2['Unnamed: 0'])
X_test2 = df2.set_index('timestamp')[['mains_1','mains_2']].values
y_test2 = df2[app2].values

pred_fc_2 = nn_model.predict(X_test2).reshape(-1)

DH_result_df = pd.DataFrame()
DH_result_df['time'] = df2.index.values
DH_result_df['y_true'] = y_test2
DH_result_df['y_pred'] = pred_fc_2


DH_result_df.to_csv(output_path+'H2_OS_%s_results.csv'%app)




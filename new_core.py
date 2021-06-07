from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import RobustScaler
import os
from keras.models import load_model
from keras.layers import Flatten
from keras import backend as K
import tensorflow as tf
import datetime
import ta
import joblib
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
pd.options.mode.use_inf_as_na = True
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def core(data):
 
  data= data.drop(["extra","_id"], axis=1, errors="ignore")

  data["new_open"]=data["open_x"].shift(periods=1)
  data['diff_open']=data['open_x']-data['new_open']
  data['sensitivity_open']=data['diff_open']/data['diff_volume']
  data['next_close']=data['close_x'].shift(periods=-1)
  data.drop(data.tail(1).index,inplace=True)

  xaseries = data[["open_x","high_x","low_x","close_x","volume_x", 'H-L' ,'O-C','SMA',"EMA","MACD","MACDEXT","STOCH","RSI","ADX","CCI","AROONUP","AROONDOWN","MSI","TRIX","BBANDSHIGH","BBANDSLOW","ATR","OBV","WILLR","sensitivity" , "diff_volume", "diff_close","sensitivity_open","diff_open" ]]
  yaseries=data[["next_close"]]

  yaseries=  yaseries.round(6)
  yaseries=yaseries.replace([np.inf, -np.inf], np.nan)
  yaseries=yaseries.fillna(0)

  xaseries=  xaseries.round(6)
  xaseries=xaseries.replace([np.inf, -np.inf], np.nan)
  xaseries=xaseries.fillna(0)


  x = xaseries.values
  y=yaseries.values

  ## scaler X
  scalerX= MinMaxScaler()
  x= x.reshape(x.shape[0], -1)
  scalerX.fit(x)
  x = scalerX.transform(x)

  ## scaler Y
  scalerY= MinMaxScaler()
  y= y.reshape(y.shape[0], -1)
  scalerY.fit(y)
  y = scalerY.transform(y)
  return x,y


def train(data):
    x=core(data)[0]
    y=core(data)[1]
    model = Sequential()
    model.add(Dense(100, input_dim=29, activation='linear'))
    model.add(Dense(120, activation='linear'))
    model.add(Dense(100, activation='linear'))
    model.add(Dense(60, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    model=load_model("modelclose3.h5")
    checkpoint = ModelCheckpoint('modelclose3.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(x, y, epochs=10, batch_size=1000,callbacks=callbacks_list)
    model.save("modelclose3.h5")

    clear_session()


def pred(data):
  data["new_open"]=data["open_x"].shift(periods=1)
  data['diff_open']=data['open_x']-data['new_open']
  data['sensitivity_open']=data['diff_open']/data['diff_volume']
  data['next_close']=data['close_x'].shift(periods=-1)

  model=load_model("modelclose3.h5")
  data=  data.round(5)
  data=data.replace([np.inf, -np.inf], np.nan)
  data=data.fillna(0)
  
  d1=data[["open_x","high_x","low_x","close_x","volume_x",'H-L' ,'O-C','SMA',"EMA","MACD","MACDEXT","STOCH","RSI","ADX","CCI","AROONUP","AROONDOWN","MSI","TRIX","BBANDSHIGH","BBANDSLOW","ATR","OBV","WILLR","sensitivity" , "diff_volume", "diff_close","sensitivity_open","diff_open" ]]
  d1=d1.values
  d2=data[['next_close']]
  
  d3=d2.values
 
  
  scalerx= MinMaxScaler()
  scalerX=scalerx.fit(d1)
  x= scalerX.transform(d1)
  ypred1=model.predict(x)



  scalery= MinMaxScaler()
#   y= d2.reshape(d2.shape[0], -1)
  scalery.fit(d3)
  ypred1=model.predict(x)
  ypred1=np.round_(ypred1,4) 
  dummy1= np.zeros(shape=(len(ypred1), 1) )
  dummy1[:,0] =ypred1[:,0]
  ypred1=scalery.inverse_transform(dummy1)
  ypred1= ypred1[:,0]
  print('prediction-----------------------')
  print(ypred1)
  
  

  yreal=d2['next_close'].tolist()
  ypandas= pd.DataFrame(list(zip(yreal,ypred1)), columns=["Real", "P1"])
  ypandas=ypandas.round(3)
  print(ypandas)
  ypandas.to_csv("pred9.csv")
  return None

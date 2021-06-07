from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import RobustScaler
import os
from keras.models import load_model
from keras.layers import Flatten
from keras import backend as K
import tensorflow as tf
import datetime
from ta import momentum,volume,volatility,trend
import joblib
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
pd.options.mode.use_inf_as_na = True
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras.layers import Bidirectional
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



BATCH_SIZE=100
TIME_STEPS=50
INVESTMENTS=10000
sensitivity=2

def get_target(data):
  data['target']=None
  for i in data.index:
    if data['future_close'][i]>=data['close'][i]:
      data['target'][i]=1#buy
    else:
      data['target'][i]=0#sell_value
  
  return data

def normalized_close(data):
  data['new_close']=None

  for ind in data.index:
    
    if ind==0:
      compare_price=data['close_x'][ind]
      data.at[ind,'new_close']=0
      #data.set_value('new_close', ind, 0) 
    else:
      change= (compare_price/data['close_x'][ind])-1
      data.at[ind,'new_close']=change
  return data,compare_price

def inverse_normalize(data,first_value):
  print(data)
  data['R']=None
  data['P']=None
  for ind in data.index:
    data.at[ind,'R']=first_value*(data['Real'][ind]+1)
    data.at[ind,'P']=first_value*(data['P1'][ind]+1)

      
  
  return data



def core(data):
  try:
    value=[]
    #REMOVING EXTRA
    data= data.drop(["extra","_id"], axis=1, errors="ignore")
    data["new_open"]=data["open_x"].shift(periods=1)
    data['diff_open']=data['open_x']-data['new_open']
    data['sensitivity_open']=data['diff_open']/data['diff_volume']
    data.fillna(0, inplace = True) 
    # data['future_open']= data["open_x"].shift(periods=-1)
    # data['future_close']=data["close_x"].shift(periods=-1)
    # data.drop(data.tail(1).index,inplace=True)
    data,compare_price=normalized_close(data)
    # New Indicators 
    data['SMA1'] = trend.sma_indicator(data["close_x"], n=60, fillna=True)
    data['awesome_oscialltor']=momentum.ao(data['high_x'], data['low_x'], 5, 34, False)
    data['kama_indicator']=momentum.kama(data['close_x'], 10, 2, 30, False)
    data['rate_of_change']=momentum.roc(data['close_x'], 12, False)
    data['stoch)Signal']=momentum.stoch_signal(data['high_x'], data['low_x'], data['close_x'], n=14, d_n=3, fillna=False)
    data['tsi']=momentum.tsi(data['close_x'], 25, 13, False)
    data['uo']=momentum.uo(data['high_x'], data['low_x'], data['close_x'],7, 14, 28, 4.0,2.0,  1.0, False)
    #2 volume indicators
    data['adi']=volume.acc_dist_index(data['high_x'],data['low_x'], data['close_x'], data['volume_x'], fillna=False)
    data['chaikin']=volume.chaikin_money_flow(data['high_x'], data['low_x'], data['close_x'], data['volume_x'], n=20, fillna=False)
    data['emv']=volume.ease_of_movement(data['high_x'],data['low_x'], data['volume_x'], n=14, fillna=False)
    data['force_index']=volume.force_index(data['close_x'], data['volume_x'], n=13, fillna=False)
    data['mfi']=volume.money_flow_index(data['high_x'], data['low_x'],data['close_x'],data['volume_x'], n=14, fillna=False)
    data['nvi']=volume.negative_volume_index(data['close_x'], data['volume_x'], fillna=False)
    data['vpt']=volume.volume_price_trend(data['close_x'], data['volume_x'], fillna=False)
    # 3  Volatility Trends

    data['bbands_high_indicator']=volatility.bollinger_hband_indicator(data['close_x'], n=20, ndev=2, fillna=False)
    data['bbands_low_indicator']=volatility.bollinger_lband_indicator(data['close_x'], n=20, ndev=2, fillna=False)
    data['bband_avg']=volatility.bollinger_mavg(data['close_x'], n=20, fillna=False)
    data['bband_percentage']=volatility.bollinger_pband(data['close_x'], n=20, ndev=2, fillna=False)
    data['bband_width']=volatility.bollinger_wband(data['close_x'], n=20, ndev=2, fillna=False)
    data['dc']=volatility.donchian_channel_hband(data['close_x'], n=20, fillna=False)
    data['dc_hban']=volatility.donchian_channel_hband_indicator(data['close_x'], n=20, fillna=False)
    data['dc_lband']=volatility.donchian_channel_lband(data['close_x'], n=20, fillna=False)
    data['dc_lband_indicator']=volatility.donchian_channel_lband_indicator(data['close_x'], n=20, fillna=False)
    data['kc_hband']=volatility.keltner_channel_hband(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    data['kc_hband_indicator']=volatility.keltner_channel_hband_indicator(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    data['kc_lband']=volatility.keltner_channel_lband(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    data['kc_lband_indicator']=volatility.keltner_channel_lband_indicator(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    data['kc_mband']=volatility.keltner_channel_mband(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    data['kc_pband']=volatility.keltner_channel_pband(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    data['kc_wband']=volatility.keltner_channel_pband(data['high_x'], data['low_x'], data['close_x'], n=10, fillna=False, ov=True)
    #4  Trend Indicator
    data['adx_negative']=trend.adx_neg(data['high_x'], data['low_x'], data['close_x'], n=14, fillna=False)
    data['adx_pos']= trend.adx_pos(data['high_x'], data['low_x'], data['close_x'], n=14, fillna=False)
    data['aroon_down']=trend.aroon_down(data['close_x'], n=25, fillna=False)
    data['aroon_up']=trend.aroon_up(data['close_x'], n=25, fillna=False)
    data['cci']=trend.cci(data['high_x'], data['low_x'], data['close_x'], n=20, c=0.015, fillna=False)
    data['dpo']=trend.dpo(data['close_x'], n=20, fillna=False)
    data['ema_indicator']=trend.ema_indicator(data['close_x'], n=12, fillna=False)
    data['ichimoku_a']=trend.ichimoku_a(data['high_x'], data['low_x'], n1=9, n2=26, visual=False, fillna=False)
    data['ichimoku_b']=trend.ichimoku_b(data['high_x'], data['low_x'],  n2=26, n3=52, visual=False, fillna=False)
    data['kst']=trend.kst(data['close_x'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False)
    data['kst_sig']=trend.kst_sig(data['close_x'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False)
    data['macd_signal']=trend.macd_signal(data['close_x'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
    data['mass_index']=trend.mass_index(data['high_x'], data['low_x'], n=9, n2=25, fillna=False)
    data['pasr_down']=trend.psar_down(data['high_x'], data['low_x'], data['close_x'], step=0.02, max_step=0.2)
    data['psar_down_indicator']=trend.psar_down_indicator(data['high_x'], data['low_x'], data['close_x'], step=0.02, max_step=0.2)
    data['pasr_up']=trend.psar_up(data['high_x'], data['low_x'], data['close_x'], step=0.02, max_step=0.2)
    data['psar_down_indicator']=trend.psar_up_indicator(data['high_x'], data['low_x'], data['close_x'], step=0.02, max_step=0.2)
    data['trix']=trend.trix(data['close_x'], n=15, fillna=False)
    data['vortex_indicator_neg']=trend.vortex_indicator_neg(data['high_x'], data['low_x'], data['close_x'], n=14, fillna=False)
    data['vortex_indicator_pos']=trend.vortex_indicator_pos(data['high_x'], data['low_x'], data['close_x'], n=14, 
    fillna=False)

  
    xaseries = data[['H-L' ,'O-C','SMA1',"EMA","MACD","MACDEXT","STOCH","RSI","ADX","CCI","AROONUP","AROONDOWN","MSI","TRIX","BBANDSHIGH","BBANDSLOW","ATR","OBV","WILLR","sensitivity" , "diff_volume", "diff_close","sensitivity_open","diff_open",'awesome_oscialltor','kama_indicator','rate_of_change','stoch)Signal','tsi','uo','adi','chaikin','emv','force_index','mfi','nvi','vpt','bbands_high_indicator','bbands_low_indicator','bband_avg','bband_percentage','bband_width','dc','dc_hban','dc_lband','dc_lband_indicator','kc_hband','kc_hband_indicator','kc_lband','kc_lband_indicator','kc_mband','kc_pband','kc_wband','adx_negative','adx_pos','aroon_down','aroon_up','cci','dpo','ema_indicator','ichimoku_a','ichimoku_b','kst','kst_sig','macd_signal','mass_index','pasr_down','psar_down_indicator','pasr_up','psar_down_indicator','trix','vortex_indicator_neg','vortex_indicator_pos']]
    yaseries=data[["new_close"]]
    xaseries=round(xaseries,2)
    
    

    yaseries=  yaseries.round(6)
    yaseries=yaseries.replace([np.inf, -np.inf], np.nan)
    yaseries=yaseries.fillna(0)

    xaseries=  xaseries.round(6)
    xaseries=xaseries.replace([np.inf, -np.inf], np.nan)
    xaseries=xaseries.fillna(0)


    x = xaseries.values
    y=yaseries.values

    # scaler X
    transformer = Normalizer().fit(x) 
    matx=transformer.transform(x)

    # scaler Y
    # scalerY= MinMaxScaler()
    maty= y.reshape(y.shape[0], -1)
    matx= matx.reshape(matx.shape[0], -1)
    # scalerY.fit(maty)

    #maty= scalerY.transform(maty)

    dim_0 = matx.shape[0] - TIME_STEPS
    dim_1 = matx.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros(dim_0,)

    for i in (range(dim_0)):
      x[i] = matx[i:TIME_STEPS+i]
      y[i]= maty[TIME_STEPS+i, 0]
     


    return x, y, compare_price
  except Exception as e :
    print('error')
    print(e)
    pass

def train(data):
  try:
    x=core(data)[0]
    y=core(data)[1]
    compare_price = core(data)[2]
    lstm_model = Sequential()
    lstm_model.add(LSTM(100,return_sequences=True, stateful=False,batch_input_shape=(BATCH_SIZE,TIME_STEPS, x.shape[2]), dropout=0.0, recurrent_dropout=0.0,kernel_initializer='random_uniform'))
    lstm_model.add(LSTM(100, return_sequences=True,dropout=0.2))
    lstm_model.add(LSTM(100, return_sequences=False, dropout=0.2))
    lstm_model.add(Dense(1, activation='linear'))
    lstm_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    lstm_model=load_model("model_lstm.h5")
    checkpoint = ModelCheckpoint('model_lstm.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    lstm_model.fit(x, y, epochs=1,batch_size=BATCH_SIZE,callbacks=callbacks_list)
    lstm_model.save("model_lstm.h5")
    clear_session()
  except Exception as e:
    print(e)
    pass 

  
def pred(data):
  x=core(data)[0]
  y=core(data)[1]
  print(y)
  compare_price = core(data)[2]

  model=load_model('model_lstm.h5')
  y_pred = model.predict(x,batch_size=BATCH_SIZE)

  y_pred=y_pred.flatten()


  ypandas= pd.DataFrame(list(zip(y,y_pred)), columns=["Real", "P1"])
  ypandas=ypandas.round(8)


  ypandas=inverse_normalize(ypandas,compare_price)
  ypandas=ypandas.round(2)
  ypandas=ypandas[['R','P']]
  print(ypandas)



  

  ypandas.to_csv("pred9.csv")

  return None



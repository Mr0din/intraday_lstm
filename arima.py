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
from ta import momentum,volume,volatility,trend
import joblib
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
pd.options.mode.use_inf_as_na = True
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE=100
TIME_STEPS=50
INVESTMENTS=10000
sensitivity=2


def get_pnl(no_of_shares,secondlast_prediction,last_prediction,sell,call):
    turnaround=no_of_shares*(secondlast_prediction+last_prediction)
    brokerage= min(20,turnaround*0.0001) 
    stt= round((sell*no_of_shares)*0.00025)
    tc=round(turnaround*0.0000325,2)
    gst=round(0.18*(brokerage+tc),2)
    sebi=round(0.000001*turnaround,2)
    stamp=round(0.000002*turnaround,2)
    total=brokerage+stt+tc+gst+sebi+stamp
    final_value=round((abs(secondlast_prediction-last_prediction)*no_of_shares)-total,2)
    print('Received from trade: ', round(abs(secondlast_prediction-last_prediction)*no_of_shares),2)
    print('total cost: ',total)
    print('pnl: ',final_value)
  
    if final_value<sensitivity:
        final_call1=2
    else:
        final_call1=call

    return final_call1

# def final_call(data):
#   data['no_of_shares']=round(INVESTMENTS/data['future_open'])
#   data['sell_value']=0#dummy
#   data['make_call']=4#dummy
#   data['final_value']=5#dummy
  
#   for ind in data.index:
#     if data['future_open'][ind]<=data['future_close'][ind]:
#       data['sell_value'][ind]=data["future_close"][ind]
#       data['make_call'][ind]=1
#     else:
#       data['sell_value'][ind]=round(data["future_open"][ind],2)
#       data['make_call'][ind]=0
#     data['final_value'][ind]=get_pnl(data['no_of_shares'][ind],data['future_open'][ind], data['future_close'][ind],data['sell_value'][ind] ,data['make_call'][ind])


  
  

#   return data
    
    


def core(data):
  value=[]
  #REMOVING EXTRA
  data= data.drop(["extra","_id"], axis=1, errors="ignore")

  data["new_open"]=data["open_x"].shift(periods=1)
  data['diff_open']=data['open_x']-data['new_open']
  data['sensitivity_open']=data['diff_open']/data['diff_volume']
  data['future_open']= data["open_x"].shift(periods=-1)
  data['future_close']=data["close_x"].shift(periods=-1)
#   data.drop(data.tail(1).index,inplace=True)
#   for_finalvalue=data[['future_open','future_close','time']]
#   data1=final_call(for_finalvalue)
#   data = pd.merge(data, data1, left_index=True, right_index=True, how='inner')


 

  # New Indicators 
  # 1 Mometum indicartors
  data['awesome_oscialltor']=momentum.ao(data['high'], data['low'], 5, 34, False)
  
  data['kama_indicator']=momentum.kama(data['close'], 10, 2, 30, False)
  data['rate_of_change']=momentum.roc(data['close'], 12, False)
  data['stoch)Signal']=momentum.stoch_signal(data['high'], data['low'], data['close'], n=14, d_n=3, fillna=False)
  data['tsi']=momentum.tsi(data['close'], 25, 13, False)
  data['uo']=momentum.uo(data['high'], data['low'], data['close'],7, 14, 28, 4.0,2.0,  1.0, False)
  #2 volume indicators
  data['adi']=volume.acc_dist_index(data["high"],data["low"], data["close"], data["volume"], fillna=False)
  data['chaikin']=volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'], n=20, fillna=False)
  data['emv']=volume.ease_of_movement(data["high"],data["low"], data["volume"], n=14, fillna=False)
  print(data['emv'])
  data['force_index']=volume.force_index(data['close'], data['volume'], n=13, fillna=False)
  data['mfi']=volume.money_flow_index(data['high'], data['low'],data['close'],data['volume'], n=14, fillna=False)
  data['nvi']=volume.negative_volume_index(data['close'], data['volume'], fillna=False)
  data['vpt']=volume.volume_price_trend(data['close'], data['volume'], fillna=False)
  # 3  Volatility Trends

  data['bbands_high_indicator']=volatility.bollinger_hband_indicator(data['close'], n=20, ndev=2, fillna=False)
  data['bbands_low_indicator']=volatility.bollinger_lband_indicator(data['close'], n=20, ndev=2, fillna=False)
  data['bband_avg']=volatility.bollinger_mavg(data['close'], n=20, fillna=False)
  data['bband_percentage']=volatility.bollinger_pband(data['close'], n=20, ndev=2, fillna=False)
  data['bband_width']=volatility.bollinger_wband(data['close'], n=20, ndev=2, fillna=False)
  data['dc']=volatility.donchian_channel_hband(data['close'], n=20, fillna=False)
  data['dc_hban']=volatility.donchian_channel_hband_indicator(data['close'], n=20, fillna=False)
  data['dc_lband']=volatility.donchian_channel_lband(data['close'], n=20, fillna=False)
  data['dc_lband_indicator']=volatility.donchian_channel_lband_indicator(data['close'], n=20, fillna=False)
  data['kc_hband']=volatility.keltner_channel_hband(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  data['kc_hband_indicator']=volatility.keltner_channel_hband_indicator(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  data['kc_lband']=volatility.keltner_channel_lband(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  data['kc_lband_indicator']=volatility.keltner_channel_lband_indicator(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  data['kc_mband']=volatility.keltner_channel_mband(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  data['kc_pband']=volatility.keltner_channel_pband(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  data['kc_wband']=volatility.keltner_channel_pband(data['high'], data['low'], data['close'], n=10, fillna=False, ov=True)
  #4  Trend Indicator
  data['adx_negative']=trend.adx_neg(data['high'], data['low'], data['close'], n=14, fillna=False)
  data['adx_pos']= trend.adx_pos(data['high'], data['low'], data['close'], n=14, fillna=False)
  data['aroon_down']=trend.aroon_down(data['close'], n=25, fillna=False)
  data['aroon_up']=trend.aroon_up(data['close'], n=25, fillna=False)
  data['cci']=trend.cci(data['high'], data['low'], data['close'], n=20, c=0.015, fillna=False)
  data['dpo']=trend.dpo(data['close'], n=20, fillna=False)
  data['ema_indicator']=trend.ema_indicator(data['close'], n=12, fillna=False)
  data['ichimoku_a']=trend.ichimoku_a(data['high'], data['low'], n1=9, n2=26, visual=False, fillna=False)
  data['ichimoku_b']=trend.ichimoku_b(data['high'], data['low'],  n2=26, n3=52, visual=False, fillna=False)
  data['kst']=trend.kst(data['close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False)
  data['kst_sig']=trend.kst_sig(data['close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False)
  data['macd_signal']=trend.macd_signal(data['close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
  data['mass_index']=trend.mass_index(data['high'], data['low'], n=9, n2=25, fillna=False)
  data['pasr_down']=trend.psar_down(data['high'], data['low'], data['close'], step=0.02, max_step=0.2)
  data['psar_down_indicator']=trend.psar_down_indicator(data['high'], data['low'], data['close'], step=0.02, max_step=0.2)
  data['pasr_up']=trend.psar_up(data['high'], data['low'], data['close'], step=0.02, max_step=0.2)
  data['psar_down_indicator']=trend.psar_up_indicator(data['high'], data['low'], data['close'], step=0.02, max_step=0.2)
  data['trix']=trend.trix(data['close'], n=15, fillna=False)
  data['vortex_indicator_neg']=trend.vortex_indicator_neg(data['high'], data['low'], data['close'], n=14, fillna=False)
  data['vortex_indicator_pos']=trend.vortex_indicator_pos(data['high'], data['low'], data['close'], n=14, 
  fillna=False)

  xaseries = data[['H-L' ,'O-C','SMA',"EMA","MACD","MACDEXT","STOCH","RSI","ADX","CCI","AROONUP","AROONDOWN","MSI","TRIX","BBANDSHIGH","BBANDSLOW","ATR","OBV","WILLR","sensitivity" , "diff_volume", "diff_close","sensitivity_open","diff_open",'awesome_oscialltor','kama_indicator','rate_of_change','stoch)Signal','tsi','uo','adi','chaikin','emv','force_index','mfi','nvi','vpt','bbands_high_indicator','bbands_low_indicator','bband_avg','bband_percentage','bband_width','dc','dc_hban','dc_lband','dc_lband_indicator','kc_hband','kc_hband_indicator','kc_lband','kc_lband_indicator','kc_mband','kc_pband','kc_wband','adx_negative','adx_pos','aroon_down','aroon_up','cci','dpo','ema_indicator','ichimoku_a','ichimoku_b','kst','kst_sig','macd_signal','mass_index','pasr_down','psar_down_indicator','pasr_up','psar_down_indicator','trix','vortex_indicator_neg','vortex_indicator_pos']]
  yaseries=data[["close_x"]]




  yaseries=  yaseries.round(6)
  yaseries=yaseries.replace([np.inf, -np.inf], np.nan)
  yaseries=yaseries.fillna(0)

  xaseries=  xaseries.round(6)
  xaseries=xaseries.replace([np.inf, -np.inf], np.nan)
  xaseries=xaseries.fillna(0)
  x = xaseries.values
  y=yaseries.values
  return x,y

    


  # #return xaseries

def pred(data):
    xmat,ymat=core(data)
    yreal=ymat[0:len(ymat)-10]
    ypred=ymat[len(ymat)-10:len(ymat)]
  
    history1=[]
  


    history = [x for x in yreal]
    
    predictions = list()
    for t in range(len(ypred)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        
        actaul=round((output[0][0]),2)
        min_=round((output[2][0][0]),2)
        max_=round((output[2][0][1]),2)
        print(actaul,min_,max_)
        yhat = output[0]
        predictions.append(yhat)
        obs = ypred[t]
        history.append(obs)
        history1.append(obs)
    error = mean_squared_error(ypred, predictions)
    
    df1=pd.DataFrame(list(zip(predictions,history1)),columns=['pred','real'])
    print(df1)

    print('Testing Mean Squared Error-------------: %.3f' % error)

  
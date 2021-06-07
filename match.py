# importing pandas package 
import pandas as pd 
import numpy as np 
from pymongo import MongoClient
import threading
from datetime import datetime


# connection with databasecd ==
def conn(n):
    conn = MongoClient("192.168.0.102", 27017)
    db = conn.market
    collection = db[n]
    return collection
# fetching data in chunks by threading 
acc= list(conn("trend_data").find({}))
acc=pd.DataFrame(acc)

data= list(conn("market_data").find({"tikr":"ACC"}))
data=pd.DataFrame(data)
data["date"]= data["date"].astype("str")
data1= list(conn("market_data_nifty").find({}))
data1=pd.DataFrame(data1)

data1.rename(columns={'_id': '_id_n','tikr': 'tikr_n','date': 'date_n','time': 'time_n','open': 'open_n','high': 'high_n','low': 'low_n','close': 'close_n','volume': 'volume_n','extra': 'extra_n'}, inplace=True)

data1["date_n"]= data1["date_n"].astype("str")

new_df = pd.merge(data,acc,how='left', left_on=['date','time'], right_on = ['combined','time'])
new_df=new_df[["close","date", "high", "low","open", "tikr", "time","volume","acc"]]
new_df1 = pd.merge(new_df,data1,how='left', left_on=['date','time'], right_on = ['date_n','time_n'])
new_df1=new_df1.drop(["date_n", "time_n"], axis=1)
new_df1=new_df1.fillna(0)
new_df1=new_df1.drop_duplicates(subset=None, keep="first", inplace=False)
conn("intraday").insert_many(new_df1.to_dict("records"))

print(new_df1)
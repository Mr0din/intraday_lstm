# importing pandas package 
import pandas as pd 
import numpy as np 
from pymongo import MongoClient
from pymongo import ASCENDING
# import learning_core
from learning_core import core
from learning_core import pred
from learning_core import train

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib 


#scaler = MinMaxScaler()


# connection with databasecd ==
def conn(chunks):
    conn = MongoClient("192.168.0.105", 27017)
    db = conn.market
    collection = db[chunks]
    conn.close()
    return collection
# fetching data in chunks by threading 

count = 0
def idlimit(page_size, last_id, tikr):
       
        """Function returns `page_size` number of documents after last_id
        and the new last_id.
        """
        start=conn("with_bnknifty")
        if last_id is None:
            # When it is first page
            cursor = list(start.find().limit(page_size))
        else:
            cursor = list(start.find({'tikr' :tikr,'_id': {'$gt': last_id}, }).sort([("date", ASCENDING),("time", ASCENDING)]).limit(page_size))


        data = pd.DataFrame(cursor)
        minx= [x for x in cursor]
        last_id = minx[-1]['_id']

       
        try:
                data1=data[["open_y","high_y","low_y","close_y","volume_y", "open_x","high_x","low_x","close_x","volume_x", 'H-L' ,'O-C','SMA',"EMA","MACD","MACDEXT","STOCH","RSI","ADX","CCI","AROONUP","AROONDOWN","MSI","TRIX","BBANDSHIGH","BBANDSLOW","ATR","OBV","WILLR","sensitivity" , "diff_volume", "diff_close" , 'date','time',"open","high","low","close",'volume']]
                return last_id,data1
                
        except: 
                
                return last_id,pd.DataFrame()

def trainscaler(chunks):
        start=conn("with_bnknifty")
        total=start.count_documents({"tikr":"ACC"})
        itera= np.around(int(total/chunks))
        last_id=Non
        for i in tqdm(range (0,itera+1)):
                last_id,scaler1=idlimit(chunks, last_id)
     
        return scaler1

def trainsession(chunks):
        tikr_list=["ACC","ADANIENT","ADANIPORTS","AMBUJACEM","ARVIND","ASHOKLEY","ASIANPAINT","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BANKBARODA","BANKINDIA","BHARATFORG","BHARTIARTL","BHEL","BOSCHLTD","BPCL","CANBK","CENTURYTEX","CIPLA","COALINDIA","DISHTV","DLF","DRREDDY","EICHERMOT","GAIL","GOLDBEES","GRASIM","HCLTECH","HDFC","HDFCBANK","HDIL","HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR","IBREALEST","IBULHSGFIN","ICICIBANK","IDBI","IDEA","IDFC","INDUSINDBK","INFRATEL","INFY","IOC","ITC", "JINDALSTEL","JSWSTEEL","KOTAKBANK","LICHSGFIN","LT","LUPIN","M&M","MARUTI","NMDC","NTPC","ONGC","PFC", "PNB","POWERGRID","RCOM","RELCAPITAL","RELIANCE","RELINFRA","RPOWER","SAIL","SBIN","SIEMENS","SUNPHARMA","SUNTV","TATAGLOBAL","TATAMOTORS","TATAMTRDVR","TATAPOWER","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UNIONBANK","UPL","VEDL","VOLTAS","WIPRO","WOCKPHARMA","YESBANK","ZEEL"]
        start=conn("with_bnknifty")
        for tikr in tikr_list:
                total=start.count_documents({'tikr':tikr})
                itera= np.around(int(total/chunks))
                last_id=None
                for i in tqdm(range (0,itera+1)):
                        if i <= -1:
                                last_id,data=idlimit(chunks, last_id, tikr)

                                print('Skipping since already trained')
                                pass
                        else:

                                last_id,data=idlimit(chunks, last_id, tikr)
                                if data.empty:
                                        pass
                                else:
                                        train(data)

    

                         

def prd(n):

        start=conn("with_bnknifty")
        data= list(start.find({"tikr":"AXISBANK"}).sort([("date", ASCENDING ), ("time", ASCENDING)]).limit(n))
        data = pd.DataFrame(data)
        
        pred(data)




# print("Training Scaler")

#trainsession(1050)
# joblib.dump(scaler, 'my_dope_model
# .pkl') 







prd(150)


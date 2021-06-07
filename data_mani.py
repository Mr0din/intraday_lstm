from pymongo import MongoClient
from tqdm import tqdm 
import pandas as pd
import ta
import numpy as np
from pymongo import ASCENDING
import multiprocessing


tikr = ["AXISBANK",
    "BHARATFORG",
    "SBIN",
    "SUNTV",
    "HINDUNILVR",
    "IOC",
    "POWERGRID",
    "GRASIM",
    "INFY",
    "WOCKPHARMA",
    "ZEEL",
    "BHEL",
    "TATAMOTORS",
    "IDFC",
    "MARUTI",
    "INDUSINDBK",
    "JSWSTEEL",
    "INFRATEL",
    "HINDPETRO",
    "CIPLA",
    "HDFCBANK",
    "ADANIENT",
    "ITC",
    "BAJAJ-AUTO",
    "HDIL",
    "ADANIPORTS",
    "BHARTIARTL",
    "HEROMOTOCO",
    "HCLTECH",
    "JINDALSTEL",
    "DLF",
    "UPL",
    "VOLTAS",
    "SIEMENS",
    "ARVIND",
    "TECHM",
    "IDEA",
    "TATASTEEL",
    "USDINR_APR19",
    "BAJFINANCE",
    "CENTURYTEX",
    "NTPC",
    "SUNPHARMA",
    "AMBUJACEM",
    "LT",
    "TATAMTRDVR",
    "RELINFRA",
    "BANKINDIA",
    "BANKBARODA",
    "BPCL",
    "LUPIN",
    "PFC",
    "BOSCHLTD",
    "WIPRO",
    "DISHTV",
    "NMDC",
    "M&M",
    "CANBK",
    "ACC",
    "TATAPOWER",
    "VEDL",
    "HDFC",
    "AUROPHARMA",
    "LICHSGFIN",
    "ONGC",
    "IDBI",
    "HINDALCO",
    "TITAN",'YESBANK', 'UNIONBANK', 'ULTRACEMCO', 'TCS', 'TATAGLOBAL', 'SAIL', 'RPOWER', 'RELIANCE', 'RELCAPITAL', 'RCOM', 'PNB', 'KOTAKBANK', 'ICICIBANK', 'IBULHSGFIN', 'IBREALEST', 'GOLDBEES', 'GAIL', 'EICHERMOT', 'DRREDDY', 'COALINDIA', 'ASIANPAINT', 'ASHOKLEY']

# dates= list(collection.find({}).distinct("date"))
# print(dates)

dates= [20171004, 20171009, 20171019, 20171026, 20171030, 20171101, 20171102, 20171103, 20171106, 20171107, 20171108, 20171109, 20171110, 20171113, 20171114, 20171115, 20171116, 20171117, 20171120, 20171121, 20171122, 20171123, 20171124, 20171127, 20171128, 20171129, 20171130, 20180101, 20180102, 20180103, 20180104, 20180105, 20180108, 20180109, 20180110, 20180111, 20180112, 20180115, 20180116, 20180117, 20180118, 20180119, 20180122, 20180123, 20180124, 20180125, 20180129, 20180130, 20180131, 20180201, 20180202, 20180205, 20180206, 20180207, 20180208, 20180209, 20180212, 20180214, 20180215, 20180216, 20180219, 20180220, 20180221, 20180222, 20180223, 20180226, 20180227, 20180228, 20180301, 20180305, 20180306, 20180307, 20180308, 20180309, 20180312, 20180313, 20180314, 20180315, 20180316, 20180319, 20180320, 20180321, 20180322, 20180323, 20180326, 20180327, 20180328, 20180402, 20180403, 20180404, 20180405, 20180406, 20180409, 20180410, 20180411, 20180412, 20180413, 20180416, 20180417, 20180418, 20180419, 20180420, 20180423, 20180424, 20180425, 20180426, 20180427, 20180430, 20180502, 20180503, 20180504, 20180507, 20180508, 20180509, 20180510, 20180511, 20180514, 20180515, 20180516, 20180517, 20180518, 20180521, 20180522, 20180523, 20180524, 20180525, 20180528, 20180529, 20180530, 20180531, 20180601, 20180604, 20180605, 20180606, 20180607, 20180608, 20180611, 20180612, 20180613, 20180614, 20180615, 20180618, 20180619, 20180620, 20180621, 20180622, 20180625, 20180626, 20180627, 20180628, 20180629, 20180702, 20180703, 20180704, 20180705, 20180706, 20180709, 20180710, 20180711, 20180712, 20180713, 20180716, 20180717, 20180718, 20180719, 20180720, 20180723, 20180724, 20180725, 20180726, 20180727, 20180730, 20180731, 20180801, 20180802, 20180803, 20180806, 20180807, 20180808, 20180809, 20180810, 20180813, 20180814, 20180816, 20180817, 20180820, 20180821, 20180823, 20180824, 20180827, 20180828, 20180829, 20180830, 20180831, 20180903, 20180904, 20180905, 20180906, 20180907, 20180910, 20180911, 20180912, 20180914, 20180917, 20180918, 20180919, 20180921, 20180924, 20180925, 20180926, 20180927, 20180928, 20181001, 20181003, 20181004, 20181005, 20181008, 20181009, 20181010, 20181011, 20181012, 20181015, 20181016, 20181017, 20181019, 20181022, 20181023, 20181024, 20181025, 20181026, 20181029, 20181030, 20181031, 20181101, 20181102, 20181105, 20181106, 20181107, 20181109, 20181112, 20181113, 20181114, 20181115, 20181116, 20181119, 20181120, 20181121, 20181122, 20181126, 20181127, 20181128, 20181129, 20181130, 20181203, 20181204, 20181205, 20181206, 20181207, 20181210, 20181211, 20181212, 20181213, 20181214, 20181217, 20181218, 20181219, 20181220, 20181221, 20181224, 20181226, 20181227, 20181228, 20181231, 20190101, 20190102, 20190103, 20190104, 20190107, 20190108, 20190109, 20190110, 20190111, 20190114, 20190115, 20190116, 20190117, 20190118, 20190121, 20190122, 20190123, 20190124, 20190125, 20190128, 20190129, 20190130, 20190131, 20190201, 20190204, 20190205, 20190206, 20190207, 20190208, 20190211, 20190212, 20190213, 20190214, 20190215, 20190218, 20190219, 20190220, 20190221, 20190222, 20190225, 20190226, 20190227, 20190228, 20190301, 20190305, 20190306, 20190307, 20190308, 20190311, 20190312, 20190313, 20190314, 20190315, 20190318, 20190319, 20190320, 20190322, 20190325, 20190326, 20190327, 20190328, 20190329, 20190401, 20190402, 20190403, 20190404, 20190405, 20190408, 20190409, 20190410, 20190411, 20190412, 20190415, 20190416, 20190418, 20190422, 20190423, 20190424, 20190425, 20190426, 20190430, 20190502, 20190503, 20190506, 20190507, 20190508, 20190509, 20190510, 20190513, 20190514, 20190515, 20190516, 20190517, 20190520, 20190521, 20190522, 20190523, 20190524, 20190527, 20190528, 20190529, 20190530, 20190531, 20190603, 20190604, 20190606, 20190607, 20190610, 20190611, 20190612, 20190613, 20190614, 20190617, 20190618, 20190619, 20190620, 20190621, 20190624, 20190625, 20190626, 20190627, 20190628, 20190701, 20190702, 20190703, 20190704, 20190705, 20190708, 20190709, 20190710, 20190711, 20190712, 20190715, 20190716, 20190717, 20190718, 20190719, 20190722, 20190723, 20190724, 20190725, 20190726, 20190729, 20190730, 20190731, 20190801, 20190802, 20190805, 20190806, 20190807, 20190808, 20190809, 20190813, 20190814, 20190816, 20190819, 20190820, 20190821, 20190822, 20190823, 20190826, 20190827, 20190828, 20190829, 20190830, 20190903, 20190904, 20190905, 20190906, 20190909, 20190911, 20190912]


def conn(dbname):
    conn = MongoClient("192.168.0.105", 27017)
    db = conn.market
    collection = db[dbname]
    return collection
# # fetching data in chunks by threading 
def trainsession(chunks):
    last_id=0
    start=conn("experiment_data")
    total=start.count_documents({})
    itera= np.around(int(total/chunks))
    for i in range (0,itera+1):
        # data= list(start.find({"date":{"$gte": "20190801"}}).limit(n))
        data= list(start.find({}).sort([("date", ASCENDING ), ("time", ASCENDING)]).skip(last_id).limit(chunks))
        data = pd.DataFrame(data)
        data= data.drop(["tikr_n","volume_n","_id_n","_id","extra_n"],axis=1)
        last_id = last_id+chunks
        train(data)

def indicators(dataframe):
    data=dataframe
    data['H-L'] = data['high'] - data['low']
    data['O-C'] = data['close'] - data['open']
    data['SMA'] = data['close'].shift(1).rolling(window = 60).mean()
    data["EMA"]=ta.trend.ema_indicator(data["close"], n=60, fillna=True)
    # data["KAMA"]=ta.momentum. kama(data["close"], n=60, pow1=60, pow2=60, fillna=True)
    data["MACD"]=ta.trend.macd(data["close"], n_fast=12, n_slow=26, fillna=True)
    data["MACDEXT"]=ta.trend.macd_diff(data["close"], n_fast=12, n_slow=26, n_sign=9, fillna=True)
    data["STOCH"]=ta.momentum.stoch(data["high"], data["low"], data["close"], n=14, fillna=True)
    data["RSI"]=ta.momentum.rsi(data["close"], n=60, fillna=True)
    data["ADX"]=ta.trend.adx(data["high"], data["low"], data["close"], n=60, fillna=True)
    data["CCI"]=ta.trend.cci(data["high"], data["low"], data["close"], n=60, c=0.015, fillna=True)
    data["AROONUP"]=ta.trend.aroon_up(data["close"], n=60, fillna=True)
    data["AROONDOWN"]=ta.trend.aroon_down(data["close"], n=60, fillna=True)
    data["MSI"]=ta.momentum.money_flow_index(data["high"], data["low"], data["close"], data["volume"], n=60, fillna=True)
    data["TRIX"]=ta.trend.trix(data["close"], n=60, fillna=True)
    data["BBANDSHIGH"]=ta.volatility.bollinger_hband(data["close"], n=60, ndev=2, fillna=True)
    data["BBANDSLOW"]=ta.volatility.bollinger_lband(data["close"], n=60, ndev=2, fillna=True)
    data["ATR"]=ta.volatility.average_true_range(data["high"], data["low"], data["close"], n=60, fillna=True)
    # data["AD"]=ta.volume.chaikin_money_flow(data["high"], data["low"], data["close"], data["volume"], n=60, fillna=True)
    data["OBV"]=ta.volume.on_balance_volume(data["close"], data["volume"], fillna=True)
    data["WILLR"]=ta.momentum.wr(data["high"], data["low"], data["close"], lbp=60, fillna=True)
    return data

# change in per unit (positive or negative to change in price positive or negative )
def sensitivity(dataframe): 
    data=dataframe
    data=data.drop(["_id"],axis=1)
    data["previous_volume"]= data["volume"].shift(1)
    data["diff_volume"]= data["volume"]- data["previous_volume"]
    data["previous_close"]=data["close"].shift(1)
    data["diff_close"]= data["close"]- data["previous_close"]
    data["sensitivity"] = data["diff_close"]/data["diff_volume"]
    data= data.drop(["previous_volume","previous_close"], axis=1)
    data=data.fillna(0)
    return data

def match_nifty(dataframe,dt):
    data=dataframe
    data=data.drop(["_id"],axis=1, errors="ignore")
    nifty_data=pd.DataFrame(list(conn("nifty_data").find({"date":dt})))
    nifty_data=nifty_data.drop(["_id","tikr","date","extra"],axis=1, errors="ignore")
    if list(nifty_data)==[]:
        new_data=data
        pass
    else:
        new_data= data.merge(nifty_data,how="left", left_on='time', right_on='time')
    return new_data
    

def capturing():
    for company in (tikr):
        for date in tqdm(dates):
            if date <= 20190131:
                pass:
            else:
                data=pd.DataFrame(list(conn("with_nifty").find({"tikr":company,"date":date}).sort("time", ASCENDING)))
                if list(data)==[]:
                    pass
                else:
                    new_df=match_nifty(data,date)
                    new_df=new_df.to_dict(orient='records')
                    conn("with_nifty").insert_many(new_df)
            
           

if __name__ == '__main__':
  
    p1 = multiprocessing.Process(target=capturing)
    p1.start()
    p1.join()
    



        
    













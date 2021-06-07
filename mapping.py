from pymongo import MongoClient
from pymongo import ASCENDING
import pandas as pd

def conn(n):
    conn = MongoClient("192.168.0.102", 27017)
    db = conn.market
    collection = db[n]
    return collection
co=0
start = conn("market_data")
data= start.distinct("tikr")
data= pd.DataFrame(data,columns=["tikr"])
data["index"]=0
for i in range (data.shape[0]):
    data["index"][i]= i


add= conn("mapping")
add.insert_many(data.to_dict("records"))
print("done....................")

    
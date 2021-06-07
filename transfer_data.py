
# importing pandas package 
import pandas as pd 
import numpy as np 
from pymongo import MongoClient
from datetime import datetime
from tqdm import tqdm


# connection with databasecd ==
def local(n):
    conn = MongoClient("192.168.0.105", 27017)
    db1 = conn.market
    collection1 = db1[n]
    return collection1

def cluster():
    client = MongoClient("mongodb+srv://mongoadmin:Jockey1809@findata-gjf4a.mongodb.net/test?retryWrites=true&w=majority")
    db = client.share_data
    collection2 = db['share_market']
    return collection2






def iterator(page_size, last_id, tikr):
       

        start=conn("with_sense")
        if last_id is None:
            # When it is first page
            cursor = list(start.find().limit(page_size))
        else:
            cursor = list(start.find({'tikr' :tikr,'_id': {'$gt': last_id}, }).sort([("date", ASCENDING),("time", ASCENDING)]).limit(page_size))
        data = pd.DataFrame(cursor)
        minx= [x for x in cursor]
        last_id = minx[-1]['_id']       
        return last_id,data

def main_():
    count=local('with_sense').count_documents({})
    loops= count//5000
    last_id=None

    for i in tqdm(range(loops)):

        last_id,data= idlimit(5000,last_id)
        cluster().insert_many(data)


main_()


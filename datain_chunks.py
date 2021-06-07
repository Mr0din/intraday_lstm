
import pandas as pd
import numpy as np
import multiprocessing
from pymongo import MongoClient
from functools import partial
from itertools import chain

cl = MongoClient('localhost')   #make sure the name of this client is different from the one used inside the function
db = cl.local
collection = db.dummy
document_ids = collection.find().distinct('_id') #list of all ids

# takes a list and integer n as input and returns generator objects of n lengths from that list

        
def calculate(chunk, input):
  #define client inside function
  client = client = MongoClient('localhost',27017,maxPoolSize=10000)
  db = client.local
  collection = db.dummy
  chunk_result_list = pd.DataFrame()
  #loop over the id's in the chunk and do the calculation with each
  for id in chunk:
      result= list(collection.find({"_id":id}))
      result=pd.DataFrame(result)
      chunk_result_list=chunk_result_list.append(result, ignore_index=True)
      print(chunk_result_list)
     
  return chunk_result_list

# pool object creation
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=8) #spawn 8 processes
    calculate_partial  = partial(calculate, input = input) #partial function creation to allow input to be sent to the calculate function
    result= pool.map(calculate_partial,list(chunks(document_ids,100)))  #creates chunks of 1000 document id's
    # # #result is now a list of lists which needs to be converted into a single list
    # result_final = list(chain.from_iterable([r.items() for r in result]))
    print ("done")
    pool.close()
    



import numpy as np
import pandas as pd
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import pickle
import logging
import string
def punctuation_removal(x):
    return " ".join([a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()])
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
# uri (uniform resource identifier) defines the connection parameters
uri ="mongodb+srv://Manoj:manoj11mikasa@cluster0.mg1dw.mongodb.net/NLP?retryWrites=true&w=majority"

# start client to connect to MongoDB server
try:
    client = MongoClient(uri)
    logging.info("MongoDB data has been loaded successfully and the stats of the MongoDB client is \n {0}".format(
        str(client.stats)))
except:
    logging.info("Not able to get the data from MongoDb atlas")

test=client.NLP.news_test
train=client.NLP.news_train
df_test=pd.DataFrame((test.find({}, {"ArticleId":1,"Text":1, "_id":0})))
df_train=pd.DataFrame((train.find({},{"ArticleId":1,"Text":1,"Category":1,"_id":0})))
logging.info("Data Has been converted as a DataFrame Sucessfully")
df_train["Text"]=df_train["Text"].apply(punctuation_removal)
print(df_train.head())



try:

    Pipe_LR=Pipeline([
        ('tfidf',TfidfVectorizer(lowercase=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')),
        ('classifier', LogisticRegression(random_state=10))])



    Pipe_LR.fit(df_train["Text"],df_train["Category"])
    logging.info("Model Fitted in the Pipeline successfully")
except:
    logging.info("Model was not able to run properly")

#Pipe_LR.predict(df_train["Text"])
print(Pipe_LR.score(df_train["Text"],df_train["Category"]))
filename = 'final_model.pkl'

pickle.dump(Pipe_LR, open(filename, 'wb'))
logging.info("Model has been dumped as pickle file")



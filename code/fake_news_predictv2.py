from nltk.classify.naivebayes import NaiveBayesClassifier
import scipy
import sklearn
from sklearn.feature_selection import chi2
#from gensim.models.fasttext import FastText
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model, preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn import svm
from sklearn.svm import SVC
from scipy.sparse import hstack, coo_matrix
import pandas as pd
import numpy as np
import pickle
import sys
import re
from html.parser import HTMLParser
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from nltk import WordPunctTokenizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from text_cleaner import html_to_text, clean
#nltk.download('stopwords')
import dill
import os
import shutil
import tensorflow as tf
import scipy

def y_decode(output):
    new_output = []
    actual = {
       1:"Real",
       0:"Fake"
    }
    for i in output:
        new_output.append(actual[i])
    return new_output

def create_model(X):
    model = Sequential()
    model.add(Dense(500, input_dim=X.shape[1],activation='relu'))
    #model.add(Dense(250, activation='relu'))
    model.add(Dense(150, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(25, activation='relu'))
    # model.add(Dense(12, activation='relu'))
    # model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', \
        metrics=['accuracy',tf.keras.metrics.AUC(from_logits=True)])
    return model



if __name__ == "__main__":
    data = pd.read_csv (r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\data\fake_news_prediction.csv')


    # model = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\models\fake_news_model.pickle", 'rb'))
    # transformer_text = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_text_fake_news.pickle", 'rb'))
    # transformer_title = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_title_fake_news.pickle", 'rb'))
    # transformer_author = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_author_fake_news.pickle", 'rb'))
    # transformer_site_url = pickle.load(open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_site_url_fake_news.pickle',"rb"))
    # transformer_img_url = pickle.load(open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_img_url_fake_news.pickle',"rb"))
    
    transformer_text = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_text_fake_news_DNN.pickle", 'rb'))
    transformer_title = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_title_fake_news_DNN.pickle", 'rb'))
    transformer_author = pickle.load(open(fr"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_author_fake_news_DNN.pickle", 'rb'))
    transformer_site_url = pickle.load(open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_site_url_fake_news_DNN.pickle',"rb"))
    transformer_img_url = pickle.load(open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_img_url_fake_news_DNN.pickle',"rb"))
 



    label_maker = preprocessing.LabelEncoder()

    data['language'] = label_maker.fit_transform(data['language'])
    #data['site_url'] = label_maker.fit_transform(data['site_url'])


    columns = ['author', 'title', 'text', 'language', 'site_url', 'hasImage']

    data['text'] = [','.join(clean(str(doc))) for doc in data['text']]
    data['title'] = [','.join(clean(str(doc))) for doc in data['title']]
    data['site_url'] = [','.join(clean(str(doc))) for doc in data['site_url']]
    data['main_img_url'] = [','.join(clean(str(doc))) for doc in data['main_img_url']]


    tfidf_text = transformer_text.transform(data['text'])
    tfidf_title = transformer_title.transform(data['title'])
    tfidf_author = transformer_author.transform(data['author'])
    tfidf_site_url = transformer_site_url.transform(data['site_url'])
    tfidf_img_url = transformer_img_url.transform(data['main_img_url'])


    X = hstack([tfidf_text, tfidf_title, tfidf_author, np.array(data['language']).reshape(-1, 1), \
        tfidf_site_url, np.array(data['hasImage']).reshape(-1, 1), tfidf_img_url])

    #print(X)
    #X = tf.reorder(X)
    model = create_model(X)
    model.load_weights(r"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\models\checkpoint.ckpt").expect_partial()

    # print(model.summary())
    # sys.exit()
    predictions = model.predict(X.todense())

    #print(folder)
    print(predictions)
    # preds = y_decode(predictions)
    # print(preds)


# Fake News
import warnings
warnings.filterwarnings("ignore")
from matplotlib import units
from scipy import sparse
from sklearn.pipeline import Pipeline
from nltk import probability
#from sklearn.preprocessing import make_pipeline
from nltk.classify.naivebayes import NaiveBayesClassifier
import sklearn
import dill
from sklearn import linear_model, preprocessing
from sklearn.feature_selection import chi2
from tensorflow.python.keras.layers.core import Dropout
from torch.optim import adam
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix, classification_report
from sklearn import svm
from sklearn.svm import SVC
from scipy.sparse import hstack, coo_matrix, csr_matrix
import pandas as pd
import numpy as np
import pickle
import sys
import re
from html.parser import HTMLParser
from nltk.corpus import stopwords
import nltk
from nltk import WordPunctTokenizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from text_cleaner import html_to_text, clean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import SGD, adam, Adadelta, RMSprop
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
from keras.optimizer_v2 import adam
import scipy
from imblearn.under_sampling import NearMiss
from sklearn.neural_network import MLPClassifier
from keras.models import Model
import tensorflow as tf

def create_model():
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
    data = pd.read_csv(r"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\data\news_articles.csv")


    # tf-idf:
    #   text, author, title

    # drop:
    #   published, text_without_stopwords, type, main_img_url, title_without_stopwords
     
    og_y = data['label']
    
    label_maker = preprocessing.LabelEncoder()
    y = label_maker.fit_transform(og_y)
    decode_y = list(set(zip(y, og_y)))
    df = pd.DataFrame(decode_y, index=None)
    df.to_csv(r"C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\data\y_actuals.csv") 


    data['language'] = label_maker.fit_transform(data['language'])
    #data['site_url'] = label_maker.fit_transform(data['site_url'])


    columns = ['author', 'title', 'text', 'language', 'site_url', 'hasImage']
    
    data['text'] = [','.join(clean(str(doc))) for doc in data['text']]
    data['title'] = [','.join(clean(str(doc))) for doc in data['title']]
    data['site_url'] = [','.join(clean(str(doc))) for doc in data['site_url']]
    data['main_img_url'] = [','.join(clean(str(doc))) for doc in data['main_img_url']]


    transformer_text = TfidfVectorizer()
    transformer_title = TfidfVectorizer()
    transformer_author = TfidfVectorizer()
    transformer_site_url = TfidfVectorizer()
    transformer_img_url = TfidfVectorizer()

    tfidf_text = transformer_text.fit_transform(data['text'])
    tfidf_title = transformer_title.fit_transform(data['title'])
    tfidf_author = transformer_author.fit_transform(data['author'])
    tfidf_site_url = transformer_site_url.fit_transform(data['site_url'])
    tfidf_img_url = transformer_img_url.fit_transform(data['main_img_url'])



    X = hstack([tfidf_text, tfidf_title, tfidf_author, np.array(data['language']).reshape(-1, 1), \
        tfidf_site_url, np.array(data['hasImage']).reshape(-1, 1), tfidf_img_url])

    print(len(y))
    # print(X)
    # sys.exit()
    smote = SMOTE(sampling_strategy='not majority', k_neighbors=3)
    #smote = SMOTE(sampling_strategy='majority', k_neighbors=3)
    nm = NearMiss(n_neighbors=3)
    ros = RandomOverSampler(random_state=42)
    rus = RandomUnderSampler(random_state=0, replacement=True)
    #X, y = smote.fit_resample(X, y)
    #X, y = smote.fit_resample(X, y)
    X, y = nm.fit_resample(X, y)
    print("over: ", len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)
    
    models = [
    XGBClassifier(use_label_encoder=False),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    linear_model.LogisticRegression(max_iter=1000,),
    MultinomialNB(),
    KNeighborsClassifier(n_neighbors=3),
    svm.SVC(probability=True)
    ]


    for model in models:
        break
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        y_pred = model.predict_proba(X_test)
        roc = roc_auc_score(y_test, y_pred[:,1], multi_class='ovr')
        print(model, acc, roc)


    # model = KNeighborsClassifier()

    # params = {
    #     "n_neighbors":[3, 4, 5, 6], 
    #     "weights":['uniform', 'distance'], 
    #     "algorithm":['auto'], 
    #     "leaf_size":[30, 40, 50, 60], 
    #     "p":[1, 2], 
    #     "metric":['minkowski'], 
    #     "metric_params":[None], 
    #     "n_jobs":[None]
    # }

    # grid = GridSearchCV(model, params,refit = True, verbose = 3, n_jobs=-1, cv=5)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)

    model = KNeighborsClassifier(n_neighbors=2,
        weights='uniform', 
        algorithm='auto', 
        leaf_size=30, 
        p=2, 
        metric='minkowski', 
        metric_params=None, 
        n_jobs=None)

    model = KNeighborsClassifier(n_neighbors=3)

    model = create_model()
    
    
    model.fit(X_train, y_train, epochs=3, batch_size=7)
    eval = model.evaluate(X_test, y_test)
    roc = eval[1]
    print(eval)


    # model.fit(X_train, y_train)
    # acc = model.score(X_test, y_test)
    # y_pred = model.predict_proba(X_test)
    # roc = roc_auc_score(y_test, y_pred[:,1], multi_class='ovr')
    # cv = cross_val_score(model, X_test, y_test, cv=StratifiedKFold(shuffle=True, n_splits=5))
    # print(acc, roc)
    # print(cv)

    best = 0.97
    if roc > best:

    #    with open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\TNE_1.pickle', 'wb') as f:
        model.save_weights(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\models\checkpoint.ckpt')
        # with open(fr'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\models\fake_news_model2.pickle', 'wb') as f:
        # #with open(fr'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\file_organizer\models\file_org.pickle', 'wb') as f:
        #     dill.dump(model, f)
        # dill.dump(transformer_text, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_text_fake_news2.pickle',"wb"))
        # dill.dump(transformer_title, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_title_fake_news2.pickle',"wb"))
        # dill.dump(transformer_author, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_author_fake_news2.pickle',"wb"))
        # dill.dump(transformer_site_url, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_site_url_fake_news2.pickle',"wb"))
        # dill.dump(transformer_img_url, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_img_url_fake_news2.pickle',"wb"))
        # DNN 
        dill.dump(transformer_text, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_text_fake_news_DNN.pickle',"wb"))
        dill.dump(transformer_title, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_title_fake_news_DNN.pickle',"wb"))
        dill.dump(transformer_author, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_author_fake_news_DNN.pickle',"wb"))
        dill.dump(transformer_site_url, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_site_url_fake_news_DNN.pickle',"wb"))
        dill.dump(transformer_img_url, open(r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\Fakes-News\transformers\transformer_img_url_fake_news_DNN.pickle',"wb"))


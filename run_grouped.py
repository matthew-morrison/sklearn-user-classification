import sqlite3
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


from pprint import pprint
from time import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.decomposition import TruncatedSVD # like PCA

# create sparse matrix for bag-of-words models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler # because minmaxscaler doesn't support sparse input.

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB # apparantly good for word counts
from sklearn.linear_model import SGDClassifier # svm?
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier


# read in preprocessed train/test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# remove first row
train = train.iloc[2:]
test = test.iloc[2:]

# grab only useful columns
train = train[['authorID', 'serverID', 'content']]
test = test[['authorID', 'serverID', 'content']]

# convert to int, not float ( could be done in readcsv)
train[['authorID','serverID']] = train[['authorID','serverID']].astype(np.int)
test[['authorID','serverID']] = test[['authorID','serverID']].astype(np.int)

X_train = train['content']
y_train = train['authorID']
X_test = test['content']
y_test = test['authorID']


#============== This section contains the pipelines for various classification models

nb_count_tfidf = make_pipeline(
    CountVectorizer(lowercase=False),
    TfidfTransformer(),
    MultinomialNB(alpha=1e-5, fit_prior=False)
)

nb_hash_tfidf = make_pipeline(
#    HashingVectorizer(n_features=100000, alternate_sign=False, norm=None, binary=False, analyzer='word', lowercase=False),
    HashingVectorizer(n_features=1000000, alternate_sign=False, norm=None, binary=False, analyzer='word', lowercase=False),
    TfidfTransformer(),
    MultinomialNB(alpha=0.01, fit_prior=False)
)

nb_count = make_pipeline(
    CountVectorizer(lowercase=False),
#    MultinomialNB(alpha=0.01, fit_prior=False)
    MultinomialNB(alpha=1e-5, fit_prior=False)
)

nb_hash = make_pipeline(
    #HashingVectorizer(non_negative=True),
    HashingVectorizer(n_features=100000, alternate_sign=False, norm=None, binary=False, analyzer='word', lowercase=False),
    MultinomialNB(alpha=0.01, fit_prior=False)
)

sgdc_hash_tfidf = make_pipeline(
    HashingVectorizer(n_features=10000, alternate_sign=False, norm=None, binary=False),
    TfidfTransformer(),
    SGDClassifier(loss='hinge',alpha=0.0001, penalty='l2', max_iter=500, tol=None)
#    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
)

sgdc_count_tfidf = make_pipeline(
    CountVectorizer(lowercase=False),
    TfidfTransformer(),
    SGDClassifier(loss='hinge',alpha=0.0001, penalty='l2', max_iter=500, tol=None)
#    SGDClassifier(max_iter=5, tol=None)
#    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
)

sgdc_hash = make_pipeline(
    HashingVectorizer(n_features=10000, alternate_sign=False, norm=None, binary=False),
    SGDClassifier(loss='hinge',alpha=0.0001, penalty='l2', max_iter=500, tol=None)
#    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
)

sgdc_count = make_pipeline(
    CountVectorizer(lowercase=False),
    SGDClassifier(loss='hinge',alpha=0.0001, penalty='l2', max_iter=500, tol=None)
#    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
)

knn_count_tfidf = make_pipeline(
    CountVectorizer(lowercase=False),
    TfidfTransformer(),
    KNeighborsClassifier(weights='distance', n_neighbors=5)
#    KNeighborsClassifier(weights='distance', n_neighbors=5)
)

knn_count = make_pipeline(
    CountVectorizer(lowercase=False),
    KNeighborsClassifier(weights='distance', n_neighbors=5)
#    KNeighborsClassifier(weights='distance', n_neighbors=5)
)

knn_hash_tfidf = make_pipeline(
    HashingVectorizer(n_features=10000, alternate_sign=False, norm=None, binary=False),
    TfidfTransformer(),
    KNeighborsClassifier(weights='distance', n_neighbors=5)
#    KNeighborsClassifier(weights='distance', n_neighbors=5)
)

knn_hash = make_pipeline(
    HashingVectorizer(n_features=10000, alternate_sign=False, norm=None, binary=False),
    KNeighborsClassifier(weights='distance', n_neighbors=5)
#    KNeighborsClassifier(weights='distance', n_neighbors=5)
)


#============ extra models

model_svm_best = make_pipeline(
    CountVectorizer(lowercase=False),
    TfidfTransformer(),
    SGDClassifier(loss='hinge', max_iter=100, tol=None, alpha=0.0001, penalty='l2')
)


model_neural = make_pipeline(
    CountVectorizer(lowercase=False),
    TfidfTransformer(),
    MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,3), activation='relu')
)

model_trees = make_pipeline(
    HashingVectorizer(n_features=10000, alternate_sign=False, norm=None, binary=False),
    TfidfTransformer(),
    ExtraTreesClassifier(n_estimators=150)

)


def models():
    # list of model names used for printout
    model_names = np.array(['nb_hash_tfidf', 'nb_hash', 'nb_count_tfidf', 'nb_count', 'sgdc_hash_tfidf', 'sgdc_hash', 'sgdc_count_tfidf', 'sgdc_count', 'knn_hash_tfidf', 'knn_hash', 'knn_count_tfidf', 'knn_count'])
    
    # list of pipeline models corresponding to model names.
    models = [nb_hash_tfidf, nb_hash, nb_count_tfidf, nb_count, sgdc_hash_tfidf, sgdc_hash, sgdc_count_tfidf, sgdc_count, knn_hash_tfidf, knn_hash, knn_count_tfidf, knn_count]
    # extra models that have shown to be effective on other datasets. These models are not very effective here.
    extra_models = [model_svm_best, model_neural, model_trees]
    
    # create an array to store results.
    model_scores = np.zeros(len(model_names))
    y_test_sorted = y_test.value_counts(ascending=False, sort=False).sort_index()

    
    for i, m in enumerate(models): # test each model and store results to be printed.
        m.fit(X_train, y_train)
        
        p = pd.Series(m.predict(X_test)).value_counts(ascending=False, sort=False).sort_index()
        score = m.score(X_test, y_test)
        print(model_names[i]+': ', score)
        model_scores[i] = score
        
        y_predict = m.predict(X_test)
        del m
        #result_analysis(X_test , y_predict, y_test)
    score_frame = pd.DataFrame({'method':model_names, 'score':model_scores}).sort_values('score', ascending='True')
    
    #for i, m in enumerate(extra_models):
    #    m.fit(X_train, y_train)
    #    p = pd.Series(m.predict(X_test)).value_counts(ascending=False, sort=False).sort_index()
    #    score = m.score(X_test, y_test)
    #    print(model_names[i]+': ', score)
    #    y_predict = m.predict(X_test)
    #    del m
        
    print(score_frame)

def main():
    models()
    
if __name__=='__main__':
    main()





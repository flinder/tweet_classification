import csv
import spacy 
import re
import logging
import sys
import itertools
import Stemmer
import pickle
import os

import pandas as pd
import numpy as np

from gensim import corpora, matutils
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from multiprocessing import Pool


sys.path.append('../../dissdat/database/')
from db import make_session
from active_vs_random import *

def search_keywords(text, keywords):
    for word in keywords:
        m = re.search(word, text, re.IGNORECASE)
        if m is not None:
            return 1.0
    return 0.0

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    # Create sql connection
    logging.info('Connecting to DB...')
    _, engine = make_session()

    # Get the data  
    query = ('SELECT cr_results.tweet_id,' 
             '       cr_results.annotation,' 
             '       cr_results.trust,'
             '       tweets.text '
             'FROM cr_results '
             'INNER JOIN tweets on tweets.id=cr_results.tweet_id')
    
    logging.info('Getting the data...')
    df = pd.read_sql(sql=query, con=engine)
    
    df.replace(to_replace=['relevant', 'irrelevant', 'None'], 
               value=[1,0,np.nan], inplace=True)
    
    # Select only judgements from trusted contributors
    df = df[['tweet_id', 'annotation', 'text']].loc[df['trust'] > 0.8]

    # Aggregate to one judgement per tweet
    def f(x):
         return pd.Series({'annotation': x['annotation'].mean(),
                           'text': x['text'].iloc[0],
                           'tweet_id': x['tweet_id'].iloc[0]})

    logging.info('Aggregating...')
    df = df[['annotation', 'text', 'tweet_id']].groupby('tweet_id').apply(f)
    df = df[['annotation', 'text']]
    
    # Make annotations binary
    df.loc[df['annotation'] >= 0.5, 'annotation'] = 1
    df.loc[df['annotation'] < 0.5, 'annotation'] = 0

    # Load nlp pipeline
    logging.info('Loading nlp pipeline...')
    stemmer = Stemmer.Stemmer('german').stemWord 

    
    if not os.path.exists('oracle_clf.pkl'):
        # Explore most predictive words:
        
        ## Make term document matrices

        ### Dictionaries
        unigrams, bigrams = make_dictionaries(df.text, process_text, stemmer)    
        
        ### Create document term matrix
        logging.info('Generating unigram features...')
        grams = [make_ngrams(d, process_text, unigrams, bigrams, stemmer) 
                 for d in df.text]
        unigram_features = matutils.corpus2dense([x[0] for x 
                                                in grams],
                                                num_terms=len(unigrams)).transpose()
        X_full = pd.DataFrame(unigram_features)
        
         ### Get best classifier using crossvalidation
        params = {'loss': ['log'],
                      'penalty': ['elasticnet'],
                      'alpha': np.linspace(0.00001, 0.001, 10),
                      'l1_ratio': np.linspace(0,1,10)}
        
        mod = SGDClassifier()
        clf = GridSearchCV(estimator=mod, param_grid=params, n_jobs=10)

        clf.fit(X_full, df.annotation)
        # Store best clf
        pickle.dump(clf, open('oracle_clf.pkl', 'wb'))
    else:
        clf = pickle.load(open('oracle_clf.pkl', 'rb'))

    # Print words with largest coefficients
    coefs = clf.best_estimator_.coef_[0]
    # Get indices of largest coefs
    most_important_idx = [x[0] for x in sorted(enumerate(coefs), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:200]]
    most_important_words = [unigrams[x] for x in most_important_idx]    

    scores = []
    for i,k in enumerate(most_important_words):
        kw = most_important_words[:(i+1)]
        
        df['keyword_relevant'] = [search_keywords(x, kw) for x in df.text]
        n_clf_relevant = df.keyword_relevant.sum()
        out = get_metrics(df.annotation, df.keyword_relevant) + [n_clf_relevant]
        scores.append(out)
        p = i * 100 / len(most_important_words)
        print(f'{p} percent done')


    with open('../data/keyword_baseline_res.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['precision','recall','f1', 'n_clf_pos'])
        for s in scores:
            writer.writerow(s)

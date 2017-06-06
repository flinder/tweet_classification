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
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.utils import shuffle
from multiprocessing import Pool
from time import time


sys.path.append('../../dissdat/database/')
from db import make_session
from active_vs_random import *

def search_keywords(text, keywords):
    for word in keywords:
        m = re.search(word, text, re.IGNORECASE)
        if m is not None:
            return 1.0
    return 0.0


def make_doc_term_mat(df, process_text, stemmer, unigrams, bigrams):
    ## Make term document matrices
   
    ### Create document term matrix
    grams = [make_ngrams(d, process_text, unigrams, bigrams, stemmer) 
             for d in df.text]
    unigram_features = matutils.corpus2csc([x[0] for x 
                                            in grams],
                                            num_terms=len(unigrams)).transpose()
    #X_full = pd.DataFrame(unigram_features)

    return unigram_features

def transform_umlaute(text):
    text = re.sub('ä', 'a', text)
    text = re.sub('Ä', 'A', text)
    text = re.sub('ü', 'u', text)
    text = re.sub('Ü', 'U', text)
    text = re.sub('ö', 'o', text)
    text = re.sub('Ö', 'O', text)
    text = re.sub('ß', 's', text)
    return text

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

    # Transform umlaute
    df['text'] = [transform_umlaute(t) for t in df.text]

    df.reset_index(inplace=True)

    # Load nlp pipeline
    logging.info('Loading nlp pipeline...')
    stemmer = Stemmer.Stemmer('german').stemWord 
    
    # ============================================================
    # Train oracle model to get good keywords
    # ============================================================
    
    # Transform text to feature matrix
    unigrams, bigrams = make_dictionaries(df.text, process_text, stemmer)    
    X_full = make_doc_term_mat(df, process_text, stemmer, unigrams, bigrams)

    if not os.path.exists('oracle_clf.pkl'):
        ### Get best classifier using crossvalidation
        params = {'loss': ['log'],
                      'penalty': ['elasticnet'],
                      'alpha': np.linspace(0.00001, 0.001, 10),
                      'l1_ratio': np.linspace(0,1,10)}
        
        mod = SGDClassifier()
        clf = GridSearchCV(estimator=mod, param_grid=params, n_jobs=6)

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
    
    # Get the baseline scores (keywords additive in order)
    
    ## Prepare data with keyword indicator variables
    for i,k in enumerate(most_important_words):
        df[k] = [search_keywords(x, [k]) for x in df.text]

    if not os.path.exists('../data/keyword_baseline_res.csv'):
        logging.info('Getting baseline scores...')
        scores = []
        for i,k in enumerate(most_important_words):
            df['keyword_relevant'] = df.iloc[:, 3:(i+4)].sum(axis=1) != 0
            n_clf_relevant = df.keyword_relevant.sum()
            out = get_metrics(df.annotation, df.keyword_relevant) + [n_clf_relevant]
            scores.append(out)

        with open('../data/keyword_baseline_res.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(['precision','recall','f1', 'n_clf_pos'])
            for s in scores:
                writer.writerow(s)

    # Get the baseline scores (random draws of increasing number of keywords)
    scores = []
    for n_keywords in range(1, 51):
        for iteration in range(0, 50):

            keywords = list(np.random.choice(most_important_words[:50], n_keywords, 
                                        replace=False))
            column_idx = [list(df.columns).index(w) for w in keywords]
            df['keyword_relevant'] = df.iloc[:, column_idx].sum(axis=1) != 0
            
            n_clf_relevant = df.keyword_relevant.sum()
            metrics = get_metrics(df.annotation, df.keyword_relevant)
            out = metrics + [n_clf_relevant, n_keywords, iteration]
            scores.append(out)

    with open('../data/keyword_baseline_randomized_res.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['precision','recall','f1', 'n_clf_pos', 'n_keywords', 
                         'iteration'])
        for s in scores:
            writer.writerow(s)



    # Get the full system scores

    ## Initial (seed) query
    query = [most_important_words[0]]

    ## Everything we need for the classifiers
    params = {'loss': ['log'],
                  'penalty': ['elasticnet'],
                  'alpha': np.linspace(0.00001, 0.001, 5), #regulariz. weight
                  'l1_ratio': np.linspace(0,1,5)}          #balance l1 l2 norm
    
    mod = SGDClassifier()
    clf = GridSearchCV(estimator=mod, param_grid=params, n_jobs=10)

    annot_size = 10
    df['exp_annotated'] = False
    df['selected'] = 0
    
    scores = []

    # Initial query
    df['keyword_relevant'] = [search_keywords(x, query) for x in df.text]
    this_data = df[df.keyword_relevant == 1] 
    
    to_annotate = this_data.sample(50).index
    df.loc[to_annotate, 'exp_annotated'] = True
    
    try:
        for i in range(0, 199):
            
            logging.info(f'Iteration: {i}')
            # Train the model
            logging.info('train model on last iterations query...')
            annotated = np.where(df.loc[df.keyword_relevant == 1, 'exp_annotated'])[0]
            clf.fit(X_full[annotated, :], df.loc[df.exp_annotated, 'annotation'])

            logging.info('Query with new query terms...')
            query.append(most_important_words[i+1])
            df['keyword_relevant'] = [search_keywords(x, query) for x in df.text]
            
            logging.info('Classifying new tweets...')
            X_new = X_full[np.where(df.keyword_relevant == 1)[0], :]
            pred = clf.predict(X_new)         
            df.loc[df.keyword_relevant == 1, 'selected'] = pred

            
            # Assess performance
            out = get_metrics(df.annotation, df.selected) + [df.selected.sum()]

            # Choose tweets for next annotation step
            logging.info('Choosing tweets for annotation...')
            not_annotated_idx = df.loc[(df.keyword_relevant == 1) & 
                                       (df.exp_annotated == False)].index
            probs = clf.predict_proba(X_full[not_annotated_idx, :])[:, 1]
            to_annot = np.argsort((0.5 - probs)**2)[:annot_size]
            df.loc[not_annotated_idx[to_annot], 'exp_annotated'] = 1
            n_annotated = df.exp_annotated.sum()

            out = out + [n_annotated]
            scores.append(out)
    except KeyboardInterrupt:
        with open('../data/keyword_clf.csv', 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(['precision','recall','f1', 'n_selected', 'n_annotated'])
            for s in scores:
                writer.writerow(s)
        raise

    with open('../data/keyword_clf.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['precision','recall','f1', 'n_selected', 'n_annotated'])
        for s in scores:
            writer.writerow(s)



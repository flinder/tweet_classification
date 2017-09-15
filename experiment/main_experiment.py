#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
import pandas as pd
import copy
import logging
import numpy as np
import time
import string
import datetime
import itertools
import sys

from scipy.special import gammaln as G
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_fscore_support)
from stop_words import get_stop_words
from multiprocessing import Pool


def draw_keywords(n, terms):
    '''
    Draw words from collection of words with weights

    returns
    list of words
    collection of words with drawn words removed
    '''
    selected = terms.sample(n=n, replace=False, weights=terms['weight'])
    terms = terms.drop(selected.index)
    return list(selected['word']), terms

class SearchEngine(object):

    def __init__(self, documents, clf_documents, n_annotate, y):
        '''
        A search engine implementing all methods described in the paper
        documents: pandas.DataFrame, term document matrix
        clf_documents: pandas.DataFrame, term document matrix, normalized for 
            classifiers
        n_annotate: int, how many documents to annotate per iteration
        y: int array, annotation for each document
        '''
        self.data = documents
        self.clf_data = clf_documents
        self.valid_terms = set(self.data.columns)
        self.all_doc_ids = set(self.data.index)
        self.annotated = {'active': set(), 'random': set()}
        self.clfs = {'active': SGDClassifier(loss='log'), 
                     'random': SGDClassifier(loss='log')}
        self.n_annotate = n_annotate
        self.y = pd.Series(y, index=self.data.index)
        self.pos_index = set(self.y.index[self.y == 1])
        custom_sw = ['rt']
        sw = (get_stop_words('de') + list(string.punctuation) + [' '] + [''] 
              + custom_sw)
        self.expansion_stopwords = set(sw)
        
    def query(self, query_terms):
        '''
        Return the indices of documents that match the query
        query: iterable, search terms
        '''
        valid_query = [t for t in query_terms if t in self.valid_terms]
        selected = self.data[valid_query].sum(axis=1) != 0
        return self.data.loc[selected].index

    def random_annotate_docs(self, method, selection):
        '''
        Annotate data randomly

        method: str, 'active' or 'random'
        '''
        available = set(selection).difference(self.annotated[method])
        if len(available) < self.n_annotate:
            return []
        return np.random.choice(list(available), self.n_annotate, 
                                     replace=False)

    def active_annotate_docs(self, selection):
        '''
        Annotate data that clfs['active'] is most uncertain about
        '''
        available = list(set(selection).difference(self.annotated['active']))
        if len(available) < self.n_annotate:
            return []
        X = self.clf_data.iloc[available]
        pps = self.clfs['active'].predict_proba(X)[:, 1]        
        inv_priority = pd.Series((0.5-pps)**2, index=available)
        
        return inv_priority.sort_values(ascending=True).index[:self.n_annotate]

    def fit(self, method):
        '''
        Train a second step classification model. If the data is not suited for
        training, returns false
        '''
        positives = self.annotated[method].intersection(self.pos_index)
        if len(positives) < 1 or len(positives) == len(self.annotated[method]):
            return False
        X = self.clf_data.iloc[list(self.annotated[method])]
        y = self.y.iloc[list(self.annotated[method])]
        self.clfs[method].fit(X, y)
        return True

    def filter_query(self, selection, method):
        '''
        Use a classification model to remove potential false positives from a
        query result
        '''
        X = self.clf_data.iloc[selection]
        classification = self.clfs[method].predict(X)
        return selection[classification == 1]

    def expand_query(self, selection, scoring_method, method, expansion_query):
        '''
        Suggest new term for query
        selection: iterable, ids of docs returned by current query
        scoring_method: str, 'king', 'monroe', 'lasso'
        method: str, 'manual' or 'automatic': should terms be selected manually 
           or automatically
        '''
        X = self.clf_data.iloc[selection]
        classification = self.clfs['active'].predict(X)

        if scoring_method == 'king':
            scores = self._king_score(selection, classification)
        elif scoring_method == 'monroe':
            scores = self._monroe_score(selection, classification)
        else:
            scores = self._lasso_score(selection)

        if method == 'automatic':
            for word in scores.index:
                if word not in self.expansion_stopwords\
                and word not in expansion_query\
                and len(word) > 4:
                    return word
        else: 
            raise ValueError()

    def _king_score(self, selection, classification):
        '''
        Assign a score to each word in vocabulary base on the method proposed
        by King et al (2017)

        selection: iterable, ids of docs returned by current query
        classification: array of classification for each element in query
           selection
        '''

        X = self.clf_data.iloc[selection]
        classification = self.clfs['active'].predict(X)
        n_kw_rel = len(selection)
        clf_rel = classification == 1
        rel = self.data.iloc[selection].loc[clf_rel]
        n_match_rel = (rel != 0).sum(axis=0)
        n_not_match_rel = n_kw_rel - n_match_rel
        irrel = self.data.iloc[selection].loc[~clf_rel]
        n_match_irrel = (irrel != 0).sum(axis=0)
        n_not_match_irrel = n_kw_rel - n_match_irrel

        likelihood = ((G(n_match_rel + 1) * 
                       G(n_match_irrel + 1) * 
                       G(n_not_match_rel + 1) * 
                       G(n_not_match_irrel + 1)) /
                      (G(n_match_rel + n_match_irrel + 2) *
                       G(n_not_match_rel + n_not_match_irrel + 2)))
        
        return likelihood.sort_values(ascending=False)

    def _monroe_score(self, selection, classification):
        '''
        Assign a score to each word in vocabulary base on the method proposed
        by King et al (2017)
        selection: iterable, ids of docs returned by current query
        classification: array of classification for each element in query
           selection
        '''
        clf_rel = classification == 1
        X = self.data.iloc[selection].groupby(clf_rel).sum()
        n = X.sum(axis=1)
        N = n.sum()
        y = X.sum(axis=0)
        a0 = 1000
        a = (a0 * y) / N
        a0i = a0 * n / N
        row = X.loc[True]
        rel_ratios = (np.log((row+a)/(row.sum()+a0i.iloc[1]-row-a))
                      -np.log((y+a)/(N+a0-y-a)))
        
        return rel_ratios.sort_values(ascending=False) 
       
        
    def _lasso_score(self, selection):
        '''
        Assign a score to each word in vocabulary based on it's coefficient size
        in a regularized logistic model

        selection: iterable, ids of docs returned by current query
        '''
        X = self.data.iloc[selection]
        y = self.y.iloc[selection]
        clf = SGDClassifier().fit(X, y)        
        scores = pd.Series(clf.coef_[0])
        scores.index = self.data.columns

        return scores.sort_values(ascending=False)

    def reset(self):
        self.annotated = {'active': set(), 'random': set()}
        self.clfs = {'active': SGDClassifier(loss='log'), 
                     'random': SGDClassifier(loss='log')}

def expand_vector(sparse_vector, length):
    out = np.full(length, False, bool)
    out[sparse_vector] = True
    return out

def evaluate_selection(selection, ground_truth, df, dtm_hashtags, 
                       dtm_users):
    '''
    Calculate evaluation metrics for a selection of documents

    selection: iterable, indices of selected documents
    ground_truth: dict, containing ground truth vectors for hashtags, users and 
        timeline
    df: pandas.DataFrame, containing a column created_at for each document
        (datetime)
    '''

    gt_users = ground_truth['users']
    gt_hashtags = ground_truth['hashtags']
    gt_timeline = ground_truth['timeline']
    labels = df['annotation']

    if len(selection) > 0:

        y_pred = expand_vector(selection, len(labels))
        f1 = f1_score(labels, y_pred)
        prec = precision_score(labels, y_pred)
        rec = recall_score(labels, y_pred)

        # Difference in distributions for hashtags , users and timeline
        ht = (dtm_hashtags.loc[selection]
                          .sum(axis=0)
                          .values
                          .reshape(1, -1))
        ht_diff = cosine_similarity(ht, gt_hashtags)[0][0]
        us = (dtm_users.loc[selection]
                       .sum(axis=0)
                       .values
                       .reshape(1, -1))
        us_diff = cosine_similarity(us, gt_users)[0][0]
        selected_times = pd.DataFrame(df['created_at'].loc[selection])
        selected_per_day = (selected_times
                            .groupby(selected_times.created_at.dt.dayofyear)
                            .count())
        selected_per_day['count'] = selected_per_day['created_at']
        selected_per_day['date'] = selected_per_day.index.astype(int)
        selected_per_day['date'] = pd.DatetimeIndex(
                [datetime.datetime(2015, 1, 1) 
                 + datetime.timedelta(days=int(x-1)) for
                 x in selected_per_day['date']])
        every_day = pd.DataFrame({'date': pd.date_range(start='1-1-2015', 
                                                        end='12-31-2015')})
        tl = (every_day.merge(selected_per_day[['count', 'date']], how='left')
                       .fillna(0))
        tl = np.array(tl['count']).reshape(1, -1)
        tl_diff = cosine_similarity(tl, gt_timeline)[0][0]
    else:
        f1 = 0
        prec = 0
        rec = 0
        us_diff = 0
        ht_diff = 0
        tl_diff = 0
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'user_similarity': us_diff, 'hashtag_similarity': ht_diff, 
            'timeline_similarity': tl_diff
            }

def process_iteration(r):
    '''
    Process the results of a single iteration. Calculates all outcome metrics
    described in the paper (f1, precision, recall, cosine similarity with ground
    truth for users, hashtags and timeline).
    '''
    stats = {'replication': [], 'iteration': [], 'method': [], 'measure': [],
             'value': []}
    query_term_frequencies = {'baseline': {}, 'expansion': {}}
    replication = r['replication']
    iteration = r['iteration']

    if iteration == 0:
        print(replication)

    for method in ['baseline', 'expansion', 'random', 'active']:
        method_res = evaluate_selection(r[method]['selection'],
                                        ground_truth, df, dtm_hashtags,
                                        dtm_users)
        for measure in method_res:
            stats['replication'].append(replication)
            stats['iteration'].append(iteration)
            stats['method'].append(method)
            stats['measure'].append(measure)
            stats['value'].append(method_res[measure])

        if method in ['baseline', 'expansion']:
            for term in r[method]['query']:
                d = query_term_frequencies[method]
                d[term] = d.get(term, 0) + 1
    return {'stats': stats, 
            'query_term_frequencies': query_term_frequencies}

def replicate(replication):
    '''
    One replication of the main experiment. Wrapped in a function for
    paralleliztion purposes. A number of objects have to exist in the main
    namespace, see below

    replicatoin: int, Replication number
    '''

    # Set up data structures for this replication
    terms = copy.copy(kwords)
    baseline_query = []
    expansion_query = []

    trained_active = False     
    trained_random = False
    engine.reset()
    
    replication_results = []

    for iteration in range(0, N_ITERATIONS):
        
        if iteration == 0:
            print(f'Replication {replication}') 

        # Container for selections
        res = {'replication': replication, 'iteration': iteration}

        # --------------------------------------------------------------
        # Keyword Baseline
        # --------------------------------------------------------------

        ## Draw Keywords
        new_word, terms = draw_keywords(1, terms) 
        baseline_query.extend(new_word)

        ## Query Data
        res['baseline'] = {'selection': engine.query(baseline_query),
                           'query': copy.copy(baseline_query)}
       
        # --------------------------------------------------------------
        # Expansion and 2 step methods
        # --------------------------------------------------------------

        # Expand Query
        # --------------------------------------------------------------
        if trained_active:
            expansion = engine.expand_query(expansion_selection, 
                                            EXPANSION_SCORE,
                                            EXPANSION_METHOD, 
                                            expansion_query)
            expansion_query.append(expansion)
        else:
            expansion_query.extend(new_word)

        expansion_selection = engine.query(expansion_query)
        res['expansion'] = {'selection': expansion_selection,
                            'query': copy.copy(expansion_query)}


        # Annotate data
        # --------------------------------------------------------------
        ## For random classifier
        random_to_annotate = engine.random_annotate_docs(
                'random', expansion_selection)
        engine.annotated['random'].update(random_to_annotate)

        ## For active learner
        if trained_active:
            active_to_annotate = engine.active_annotate_docs(
                    expansion_selection)
        else:
            active_to_annotate = engine.random_annotate_docs(
                    'active', expansion_selection)
        engine.annotated['active'].update(active_to_annotate)


        # Train classifiers 
        # --------------------------------------------------------------
        trained_random = engine.fit('random')
        trained_active = engine.fit('active')


        # Filter data
        # --------------------------------------------------------------
        ## with active classifier
        if trained_active:
            selection_active = engine.filter_query(expansion_selection,
                                                   'active')
        else:
            selection_active = expansion_selection

        ## with random classifier
        if trained_random:
            selection_random = engine.filter_query(expansion_selection,
                                                   'random')
        else:
            selection_random = expansion_selection
        
        res['active'] = {'selection': selection_active}
        res['random'] = {'selection': selection_random}

        replication_results.append(res)

    return replication_results


if __name__ == "__main__":
    '''
    This script runs the main experiment
    '''

    # =========================================================================
    # Settings
    # =========================================================================
    DATA_DIR = '../data/dtms'
    N_REPLICATIONS = 100
    N_ITERATIONS = 100
    N_CORES = 10
    N_ANNOTATE_PER_ITERATION = 5
    EXPANSION_SCORE = sys.argv[1]
    EXPANSION_METHOD = 'automatic'
    STORE_FILE_NAME_SELEC = f'selections_{EXPANSION_SCORE}_{EXPANSION_METHOD}.p'
    STORE_FILE_NAME_QUERY = f'queries{EXPANSION_SCORE}_{EXPANSION_METHOD}.p'
    OUTPUT_FILE_NAME = (f'../data/results/experiment_results_{EXPANSION_SCORE}_'
                        f'{EXPANSION_METHOD}.csv')

    # =========================================================================
    # Data Import 
    # =========================================================================
    load = lambda x: pickle.load(open(os.path.join(DATA_DIR, x), 'rb'))
    df = load('df.p')
    dtm_normalized = pd.DataFrame(load('dtm_normalized.p').todense(), 
                                  columns=load('norm_terms.p'))
    dtm_non_normalized = pd.DataFrame(load('dtm_non_normalized.p').todense(), 
                                      columns=load('non_norm_terms.p'))
    ground_truth = load('ground_truth.p')
    dtm_hashtags = load('dtm_hashtags.p')
    dtm_users = load('dtm_users.p')
    kwords = load('kwords.p') 

    # =========================================================================
    # Experiment
    # =========================================================================
    # Set up data structures for the experiment
    engine = SearchEngine(dtm_non_normalized, dtm_normalized,
                          N_ANNOTATE_PER_ITERATION, df.annotation)
    process_pool = Pool(N_CORES)

    if not os.path.exists(STORE_FILE_NAME_SELEC):
        results = process_pool.map(replicate, list(range(0, N_REPLICATIONS)))
        results = list(itertools.chain.from_iterable(results))
        pickle.dump(results, open(STORE_FILE_NAME_SELEC, 'wb'))            
    else:
        print('Skipping experiment and loading cached results')
        results = pickle.load(open(STORE_FILE_NAME_SELEC, 'rb'))


    # =========================================================================
    # Process the results
    # =========================================================================
    processed = process_pool.map(process_iteration, results)

    stats = {'replication': [], 'iteration': [], 'method': [], 'measure': [],
             'value': []}
    query_term_frequencies = {'baseline': {}, 'expansion': {}}

    for r in processed:
        for key in stats:
            stats[key].extend(r['stats'][key])
        for method in ['baseline', 'expansion']:
            d = query_term_frequencies[method]
            d_iter = r['query_term_frequencies'][method]
            for term in d_iter:
                d[term] = d.get(term, 0) + d_iter[term]

    output = pd.DataFrame(stats) 
    output.to_csv(OUTPUT_FILE_NAME, index=False)
    pickle.dump(query_term_frequencies,
                open(STORE_FILE_NAME_QUERY, 'wb'))



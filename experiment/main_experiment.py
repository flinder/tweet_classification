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
import warnings

from scipy.special import gammaln as G
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from stop_words import get_stop_words
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from SearchEngine import SearchEngine

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
    query_term_frequencies = {'baseline': {}, 'expansion': {}, 'klr': {}}
    replication = r['replication']
    iteration = r['iteration']

    if iteration == 0:
        print(replication)

    #for method in ['baseline', 'expansion', 'random', 'active', 'klr']:
    for method in ['baseline', 'expansion', 'active', 'klr']:
        method_res = evaluate_selection(r[method]['selection'],
                                        ground_truth, df, dtm_hashtags,
                                        dtm_users)
        for measure in method_res:
            stats['replication'].append(replication)
            stats['iteration'].append(iteration)
            stats['method'].append(method)
            stats['measure'].append(measure)
            stats['value'].append(method_res[measure])

    return stats

def replicate(replication):
    '''
    One replication of the main experiment. Wrapped in a function for
    paralleliztion purposes. A number of objects have to exist in the main
    namespace, see below

    replicatoin: int, Replication number
    '''

    # Set up data structures for this replication
    start = time.time()
    terms = copy.copy(kwords)
    baseline_query = []
    expansion_query = []
    klr_query = []

    trained_active = False     
    trained_random = False
    trained_klr = False
    engine.reset()
    np.random.seed(replication)
    
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
            s = time.time()
            try:
                expansion = engine.expand_query(expansion_selection, 
                                                EXPANSION_SCORE,
                                                EXPANSION_METHOD, 
                                                expansion_query)
                expansion_query.append(expansion)
            except Exception as e:
                print(e)
                # If the query can't be extended, use a survey keyword
                expansion_query.extend(new_word)
            t = time.time() - s
            #print(f'active expansion took: {t}s')
        else:
            expansion_query.extend(new_word)

        expansion_selection = engine.query(expansion_query)
        res['expansion'] = {'selection': expansion_selection,
                            'query': copy.copy(expansion_query)}

        if trained_klr:
            s = time.time()
            try:
                expansion = engine.expand_query(klr_selection,
                                                EXPANSION_SCORE,
                                                EXPANSION_METHOD,
                                                klr_query, True)
                klr_query.append(expansion)
            except Exception as e:
                print(e)
                # If the query can't be extended, use a survey keyword
                klr_query.extend(new_word)

            t = time.time() - s
            #print(f'klr expansion took: {t}s')
        else:
            klr_query.extend(new_word)

        klr_selection = engine.query(klr_query)
        res['klr'] = {'selection': klr_selection,
                      'query': copy.copy(klr_query)}


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
        #trained_random = engine.fit('random')
        trained_active = engine.fit('active')
        trained_klr = engine.fit('klr')


        # Filter data
        # --------------------------------------------------------------
        ## with active classifier
        if trained_active:
            selection_active = engine.filter_query(expansion_selection,
                                                   'active')
        else:
            selection_active = expansion_selection

        ## with random classifier
        #if trained_random:
        #    selection_random = engine.filter_query(expansion_selection,
        #                                           'random')
        #else:
        #    selection_random = expansion_selection

        
        res['active'] = {'selection': selection_active}
        #res['random'] = {'selection': selection_random}
        res['tuning_parameters'] = engine.get_tuning_parameters()

        replication_results.append(res)
    
    t = time.time() - start
    print(f'Replication {replication} took {t} seconds')
    return replication_results


if __name__ == "__main__":
    '''
    This script runs the main experiment
    '''

    # =========================================================================
    # Settings
    # =========================================================================
    DATA_DIR = '../data/dtms'
    N_CORES = 4
    N_REPLICATIONS = 50
    N_ITERATIONS = 200
    N_ANNOTATE_PER_ITERATION = 5
    EXPANSION_SCORE = 'lasso'
    EXPANSION_METHOD = 'automatic'
    #RUN_ID = 'new_version_'
    RUN_ID = 'v3_aci'
    STORE_FILE_NAME_SELEC = f'{RUN_ID}_{EXPANSION_SCORE}_{EXPANSION_METHOD}.p'
    STORE_FILE_NAME_QUERY = f'{RUN_ID}_queries_{EXPANSION_SCORE}_{EXPANSION_METHOD}.p'
    OUTPUT_FILE_NAME = (f'../data/results/{RUN_ID}_experiment_results_{EXPANSION_SCORE}_'
                        f'{EXPANSION_METHOD}.csv')
    TUNING_FILE_NAME = (f'../data/results/{RUN_ID}_tuning_parameters_{EXPANSION_SCORE}_'
                        f'{EXPANSION_METHOD}.csv')


    # =========================================================================
    # Data Import 
    # =========================================================================
    load = lambda x: pickle.load(open(os.path.join(DATA_DIR, x), 'rb'))
    load_pd = lambda x: pd.read_pickle(os.path.join(DATA_DIR, x))
    df = load_pd('df.p')
    dtm_normalized = pd.DataFrame(load('dtm_normalized.p').todense(), 
                                  columns=load('norm_terms.p'))
    dtm_non_normalized_sparse = load('dtm_non_normalized.p')
    dtm_non_normalized = pd.DataFrame(load('dtm_non_normalized.p').todense(), 
                                      columns=load('non_norm_terms.p'))

    ground_truth = load_pd('ground_truth.p')
    dtm_hashtags = load_pd('dtm_hashtags.p')
    dtm_users = load_pd('dtm_users.p')
    kwords = load_pd('kwords.p') 

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

    for r in processed:
        for key in stats:
            stats[key].extend(r[key])

    output = pd.DataFrame(stats) 
    output.to_csv(OUTPUT_FILE_NAME, index=False)

    # Process the query terms: For each final query (iteration N_ITERATIONS) get
    # count the terms
    # Process tunign parameters: For each iteration extract the tunign
    # parameters if a model has been fit
   
    tuning_params = {'replication': [], 'iteration': [], 'method': [], 
                     'value': []}
    query_terms = {'baseline': {}, 'expansion': {}, 'klr': {}}

    for r in results:
        # Get the tuning parameters
        tps = r['tuning_parameters']
        #for i,method in enumerate(['active', 'random', 'klr']):
        for i,method in enumerate(['active', 'klr']):
            tuning_params['replication'].append(r['replication'])
            tuning_params['iteration'].append(r['iteration'])
            tuning_params['method'].append(method)
            try:
                tuning_params['value'].append(tps[i]['alpha'])
            except TypeError:
                tuning_params['value'].append(np.nan)

        # Process the query
        if r['iteration'] == N_ITERATIONS - 1:
            for method in ['baseline', 'expansion', 'klr']:
                terms = r[method]['query']
                for t in terms:
                    query_terms[method][t] = query_terms[method].get(t, 0) + 1
        
    output = pd.DataFrame(tuning_params) 
    output.to_csv(TUNING_FILE_NAME)
    pickle.dump(query_terms, open(STORE_FILE_NAME_QUERY, 'wb'))


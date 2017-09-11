#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
import pandas as pd
import copy
import logging
import numpy as np
import string

from scipy.special import gammaln as G
from sklearn.linear_model import SGDClassifier
from stop_words import get_stop_words



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
        self.y = y
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

    def expand_query(self):
        '''
        Given a list of indices of selected documents, expand the query using
        provided method
        '''
        pass

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
        available = set(selection).difference(self.annotated['active'])
        if len(available) < self.n_annotate:
            return []
        X = self.clf_data.iloc[selection]
        pps = self.clfs['active'].predict_proba(X)[:, 1]
        return np.argsort((0.5 - pps)**2)[:self.n_annotate]

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
            scores = self.king_score(selection, classification)
        elif scoring_method == 'monroe':
            scores = self.monroe_score(selection, classification)
        else:
            scores = self.lasso_score(selection)

        if method == 'automatic':
            for word in scores.index:
                if word not in self.expansion_stopwords\
                and word not in expansion_query\
                and len(word) > 4:
                    return word
        else: 
            raise ValueError()

    def king_score(self, selection, classification):
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

    def monroe_score(self, selection, classification):
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
       
        
    def lasso_score(self, selection):
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



if __name__ == "__main__":
    '''
    This script runs the main experiment
    '''

    # =========================================================================
    # Settings
    # =========================================================================
    DATA_DIR = '../data/dtms'
    N_REPLICATIONS = 200
    N_ITERATIONS = 100
    N_ANNOTATE_PER_ITERATION = 5
    EXPANSION_SCORE = 'lasso'
    EXPANSION_METHOD = 'automatic'

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
    kwords = load('kwords.p') 

    # =========================================================================
    # Experiment
    # =========================================================================
    # Set up data structures for the experiment
    engine = SearchEngine(dtm_non_normalized, dtm_normalized,
                          N_ANNOTATE_PER_ITERATION, df.annotation)

    results = [] 
    for replication in range(0, N_REPLICATIONS):

        # Set up data structures for this replication
        terms = copy.copy(kwords)
        baseline_query = []
        expansion_query = []

        trained_active = False     
        trained_random = False
        engine.reset()

        for iteration in range(0, N_ITERATIONS):
    
            print(f'Replication {replication}, iteration {iteration}') 

            # Container for selections
            res = {'replication': replication, 'iteration': iteration}

            # ------------------------------------------------------------------
            # Keyword Baseline
            # ------------------------------------------------------------------

            ## Draw Keywords
            new_word, terms = draw_keywords(1, terms) 
            baseline_query.extend(new_word)

            ## Query Data
            res['baseline'] = {'selection': engine.query(baseline_query),
                               'query': copy.copy(baseline_query)}
           
            # ------------------------------------------------------------------
            # Expansion and 2 step methods
            # ------------------------------------------------------------------

            # Expand Query
            # ------------------------------------------------------------------
            if trained_active:
                expansion = engine.expand_query(expansion_selection, 
                                                EXPANSION_SCORE,
                                                EXPANSION_METHOD, 
                                                expansion_query)
                expansion_query.append(expansion)
            else:
                expansion_query.extend(new_word)
            print(expansion_query)

            expansion_selection = engine.query(expansion_query)
            print(len(expansion_selection))
            res['expansion'] = {'selection': expansion_selection,
                                'query': copy.copy(expansion_query)}


            # Annotate data
            # ------------------------------------------------------------------
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
            # ------------------------------------------------------------------
            trained_random = engine.fit('random')
            trained_active = engine.fit('active')


            # Filter data
            # ------------------------------------------------------------------
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

            results.append(res)
            del res


    pickle.dump(results, open('selections_09122017.p', 'wb'))            


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
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from stop_words import get_stop_words
from multiprocessing import Pool


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
        self.data_bool = self.data != 0
        self.clf_data = clf_documents
        self.valid_terms = set(self.data.columns)
        self.all_doc_ids = set(self.data.index)
        self.annotated = {'active': set(), 'random': set()}
        self.clf_template = {
                'active': GridSearchCV(
                    estimator=SGDClassifier(loss='log', penalty='l1'), 
                    param_grid={'alpha': np.linspace(0.0001, 0.01, 5)}),
                'random': GridSearchCV(
                    estimator=SGDClassifier(loss='log', penalty='l1'), 
                    param_grid={'alpha': np.linspace(0.0001, 0.01, 5)}),
                'klr': GridSearchCV(
                    estimator=SGDClassifier(loss='log', penalty='l1'), 
                    param_grid={'alpha': np.linspace(0.0001, 0.01, 5)})}
        self.clfs = copy.copy(self.clf_template)
        self.n_annotate = n_annotate
        self.y = pd.Series(y, index=self.data.index)
        self.pos_index = set(self.y.index[self.y == 1])
        self.neg_index = set(self.y.index[self.y == 0])
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
        if method in ['klr', 'random']:
            annot_method = 'random'
        else:
            annot_method = 'active'

        positives = self.annotated[annot_method].intersection(self.pos_index)


        if method == 'klr':
            if len(positives) < 3:
                return False
            # Get a random sample of 'negatives'
            negatives_pool = list(self.all_doc_ids)
            
            neg_sample = np.random.choice(a=negatives_pool, size=len(positives),
                                          replace=False)
            X = self.clf_data.iloc[list(positives) + list(neg_sample)]
            y = pd.Series([1]*len(positives) + [0]*len(neg_sample))
        else:
            if len(positives) < 3 or len(positives) > (len(self.annotated[method])-3):
                return False
            X = self.clf_data.iloc[list(self.annotated[annot_method])]
            y = self.y.iloc[list(self.annotated[annot_method])]

        self.clfs[method].fit(X.as_matrix(), y.as_matrix())
        return True

    def filter_query(self, selection, method):
        '''
        Use a classification model to remove potential false positives from a
        query result
        '''
        X = self.clf_data.iloc[selection]
        classification = self.clfs[method].predict(X)
        return selection[classification == 1]

    def expand_query(self, selection, scoring_method, method, expansion_query,
                     random_negative=False):
        '''
        Suggest new term for query
        selection: iterable, ids of docs returned by current query
        scoring_method: str, 'king', 'monroe', 'lasso'
        method: str, 'manual' or 'automatic': should terms be selected manually 
           or automatically
        random_negative: bool, should the klr method of query expansion be used
        '''
        
        if not random_negative:
            X = self.clf_data.loc[selection]
            classification = self.clfs['active'].predict(X)

            if scoring_method == 'king':
                scores = self._king_score(selection, classification)
            elif scoring_method == 'monroe':
                scores = self._monroe_score(selection, classification)
            else:
                scores = self._lasso_score(selection)

        else:
            positives = self.annotated['random'].intersection(self.pos_index)
            search_set = self.all_doc_ids.difference(positives)
            X = self.clf_data.loc[search_set]
            classification = self.clfs['klr'].predict(X)
            scores = self._king_score(search_set, classification)

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
        
        n = len(selection)
        n_1 = sum(classification)
        n_0 = n - n_1
        X = self.data_bool.loc[selection].groupby(classification == 1).sum()

        # Count how many docs match each word in each condition
        try:
            n_match_1 = X.loc[True] 
        except KeyError:
            n_match_1 = 0
        try:
            n_match_0 = X.loc[False] 
        except KeyError:
            n_match_0 = 0

        # Calculate score per word
        likelihood = ((G(n_match_1 + 1) * 
                       G(n_match_0 + 1) * 
                       G(n_1 - n_match_1 + 1) * 
                       G(n_0 - n_match_0 + 1)) /
                      (G(n_match_1 + n_match_0 + 2) *
                       G(n_1 - n_match_1 + n_0 - n_match_0 + 2)))
        
        return likelihood.sort_values(ascending=False)

    def _monroe_score(self, selection, classification):
        '''
        Assign a score to each word in vocabulary base on the method proposed
        by King et al (2017)
        selection: iterable, ids of docs returned by current query
        classification: array of classification for each element in query
           selection
        '''
        X = self.data.loc[selection].groupby(classification == 1).sum()
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
        clf = SGDClassifier(loss='log').fit(X, y)        
        scores = pd.Series(clf.coef_[0])
        scores.index = self.data.columns

        return scores.sort_values(ascending=False)

    def reset(self):
        self.annotated = {'active': set(), 'random': set()}
        self.clfs = copy.copy(self.clf_template)


    def get_tuning_parameters(self):
        try:
            a = self.clfs['active'].best_params_
        except (NotFittedError, AttributeError):
            a = None
        try:
            b = self.clfs['random'].best_params_
        except (NotFittedError, AttributeError):
            b = None
        try:
            c = self.clfs['klr'].best_params_
        except (NotFittedError, AttributeError):
            c = None
        return a, b, c



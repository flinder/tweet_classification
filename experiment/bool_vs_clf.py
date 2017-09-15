#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, make_scorer)
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from main_experiment import draw_keywords


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=10, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]

if __name__ == "__main__":
    '''
    This script runs the comparison of a boolean keyword classifier and a 
    machine learning classifier
    '''

    ## Load the data
    dtm_normalized = pickle.load(open('../data/dtms/dtm_normalized.p', 'rb'))
    df = pickle.load(open('../data/dtms/df.p', 'rb'))

    models = { 
	#'ExtraTreesClassifier': ExtraTreesClassifier(),
	#'RandomForestClassifier': RandomForestClassifier(),
	#'AdaBoostClassifier': AdaBoostClassifier(),
	#'GradientBoostingClassifier': GradientBoostingClassifier(),
	#'SVC': SVC(),
        'SGD': SGDClassifier()	
    }

    params = { 
	#'ExtraTreesClassifier': { 'n_estimators': [10, 50, 100] },
	#'RandomForestClassifier': { 'n_estimators': [10, 50, 100] },
	#'AdaBoostClassifier':  { 'n_estimators': [10, 50, 100] },
	#'GradientBoostingClassifier': { 'n_estimators': [10, 50, 100], 
        #                                'learning_rate': [0.8, 1.0] },
	#'SVC': [
	#    {'kernel': ['linear'], 'C': [1, 10]},
	#    {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	#],
        'SGD': {'loss': ['log', 'hinge', 'modified_huber'], 
                'penalty': ['elasticnet'], 
                'alpha': np.linspace(0.00001,0.001, 5),
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'eta0': [0.1,0.5,1],
                'l1_ratio': np.linspace(0,1,5)}

    }
    
    helper = EstimatorSelectionHelper(models, params)
    scorer = make_scorer(f1_score)
    X_train, X_test, y_train, y_test = train_test_split(
            dtm_normalized.todense(), df['annotation'].as_matrix()) 
    helper.fit(X_train, y_train, scoring=scorer, n_jobs=1)

    pred = clf.predict(X_test)
    print('Classifier metrics: {}'.format(get_metrics(y_test, pred)))

    coefs = clf.best_estimator_.coef_
    largest_coefs = np.argsort(coefs[0])[-200:]
    best_features = dtm_normalized.columns[largest_coefs]
    probs = coefs[0, largest_coefs] / coefs[0, largest_coefs].sum()
    words = pd.DataFrame({'word': best_features, 'weight': probs})
    
    output = {'replication': [], 'iteration': [], 'measure': [], 'value': []}
    for replication in range(50):
        print(replication)
        keywords, seed_keywords = draw_keywords(1, words[['word', 'weight']])
        for iteration in range(1, 201):
            if iteration > 1:
                next_keyword, seed_keywords = draw_keywords(
                        1, seed_keywords[['word', 'weight']]
                        )
            keywords.extend(next_keyword)
            prediction = dtm_normalized[keywords].sum(axis=1) > 0
            scores = get_metrics(df['annotation'], prediction)
            for i, measure in enumerate(['precision', 'recall', 'f1']):
                output['replication'].append(replication)
                output['iteration'].append(iteration)
                output['measure'].append(measure)
                output['value'].append(scores[i])

    pd.DataFrame(output).to_csv('../data/boolean_vs_clf.csv', index=False)

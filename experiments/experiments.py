import copy
import csv
import spacy 
import re
import logging
import sys
import itertools
import Stemmer
import pickle
import os
import glob 
import operator
import time
import datetime

import pandas as pd
import numpy as np

from gensim import corpora, matutils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, make_scorer)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from multiprocessing import Pool
from scipy.special import gamma as G
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


sys.path.append('../../dissdat/database/')
from db import make_session


class TextParser(object):

    def __init__(self, language):
        self.parser = spacy.load(language)
        self.stemmer = Stemmer.Stemmer(language).stemWord 
        self.rm_chars = re.compile('[@/\\\\]')

    def tokenize(self, text, stem=False):
        if stem:
            tokens = [self.stemmer(self.rm_chars.sub('', t.orth_).lower())
                      for t in self.parser(text)]
        else:
            tokens = [self.rm_chars.sub('', t.orth_).lower()
                      for t in self.parser(text)]

        return tokens
    
    def lemmatize(self, text):
        return [t.lemma_ for t in self.parser(text)]

def get_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return [prec, rec, f1]

def make_dtm(documents, parser, normalize=True, entities=False):
    '''
    Tokenize and stem documents and store in sparse matrix. Note this is in
    memory, so don't use on large text collections
    '''
    unigrams = []
    vocabulary = corpora.Dictionary()
    bow = []
    for d in documents:
        unis = [t for t in parser.tokenize(d, stem=normalize)]
        unigrams.append(unis)
        bow.append(vocabulary.doc2bow(unis, allow_update=True))
    if normalize:
        vocabulary.filter_extremes(no_below=2, no_above=0.3)
        bow = [vocabulary.doc2bow(x) for x in unigrams]

    dtm = pd.DataFrame(matutils.corpus2dense(bow, num_terms=len(vocabulary),
                                             num_docs=len(bow)).transpose(),
                       columns=[x[1] for x in vocabulary.items()])

    return dtm

def make_index(documents, users, parser, mode):
    unigrams = []
    vocabulary = corpora.Dictionary()
    bow = []
    for doc, user in zip(documents, users):
        out = []
        if mode == 'users':
            parser_ = copy.copy(parser)
            parser_.rm_chars = re.compile('[/\\\\]')
            unis = [t for t in parser_.tokenize(doc, stem=False)]
            out.append('@' + user)
            for u in unis:
                if re.match('@[A-Za-z0-9_]+', u) is not None:
                   out.append(u) 
        if mode == 'hashtags':
            unis = [t for t in parser.tokenize(doc, stem=False)]
            skip = False
            for i,u in enumerate(unis):
                if skip:
                    skip = False
                    continue
                if u == '#':
                    try:
                        ht = u + unis[i+1]
                        if re.match('#[A-Za-z0-9_]+', ht) is not None:
                            out.append(ht)
                        skip = True
                    except IndexError: 
                        break

        unigrams.append(out)
        bow.append(vocabulary.doc2bow(out, allow_update=True))

    dtm = pd.DataFrame(matutils.corpus2dense(bow, num_terms=len(vocabulary),
                                             num_docs=len(bow)).transpose(),
                        columns=[x[1] for x in vocabulary.items()])

    return dtm

def clean(word, parser):
    w = word.lstrip()
    w = w.strip()
    w = w.lower()
    w = parser.tokenize(w)
    if len(w) != 1:
        return None
    w = w[0]
    if w == '':
        return None
    if len(w) > 60:
        return None
    return w

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


def keyword_likelihood(df, dtm):
    '''
    Calculate the likelihood score for each word as described in King et al 2017
    '''
    kw_rel = df['keyword_relevant']
    clf_rel = df['clf_relevant']
    n_kw_rel = kw_rel.sum()
    rel = dtm.loc[kw_rel].loc[clf_rel]
    n_match_rel = (rel != 0).sum(axis=0)
    n_not_match_rel = n_kw_rel - n_match_rel

    irrel = dtm.loc[kw_rel].loc[~clf_rel]
    n_match_irrel = (irrel != 0).sum(axis=0)
    n_not_match_irrel = n_kw_rel - n_match_irrel

    likelihood = ((G(n_match_rel + 1) * 
                   G(n_match_irrel + 1) * 
                   G(n_not_match_rel + 1) * 
                   G(n_not_match_irrel + 1)) /
                  (G(n_match_rel + n_match_irrel + 2) *
                   G(n_not_match_rel + n_not_match_irrel + 2)))

    return likelihood
   

def fighting_words(df, dtm):
    kw_rel = df['keyword_relevant']
    clf_rel = df['clf_active_relevant']
    X = dtm.loc[kw_rel].groupby(clf_rel.loc[kw_rel]).sum()
    n = X.sum(axis=1)
    N = n.sum()
    y = X.sum(axis=0)
    a0 = 1000
    a = (a0 * y) / N
    a0i = a0 * n / N
    # Odds ratios for the irrelevant documents
    l = X.index
    print(f'Index labels X: {l}')
    row = X.loc[False]
    irel_ratios = (np.log((row+a)/(row.sum()+a0i.iloc[0]-row-a))
                   -np.log((y+a)/(N+a0-y-a)))
    # Odds ratios for the relevant documents
    row = X.loc[True]
    rel_ratios = (np.log((row+a)/(row.sum()+a0i.iloc[1]-row-a))
                  -np.log((y+a)/(N+a0-y-a)))


    return rel_ratios, irel_ratios

def replication_2_step(df, n_iterations, seed_keywords, 
                       dtm_search, dtm_clf, n_annotate_step):
    '''
    Execute one replication of the 2 stage data retrieval experiment

    Arguments:
    ---------- 
    df: the full dataset
    n_iterations: How many keywords to add
    accepted_keywords: list of words a human would accept as search terms
    seed_keywords: pandas df with list of seed search terms and weights
    dtm_search: A document term matrix containng the full non-normalized 
        vocabulary.
    dtm_clf: A document term matrix with normalized vocabulary
    n_annotate_step: Number of tweets to annotate for 2.step (per iteration)

    Returns:
    ---------- 
    A list with a dictionary for the results of each iteration. The dict
    had the following keys:
        * iteration: int, iteration #
        * selected_search: pandas index with selected tweets from search terms
        * selected_clf_active: pandas index with selected tweets after 2. step
            with active learning annotation
        * selected_clf_random: pandas index with selected tweets after 2. 
            step with random annotation
        * clf_success: Could 2. step be executed
        * search_terms: the terms used in this iteration
    '''

    global accepted_keywords
    out = [] 
    keywords = []
    next_keyword = None
    to_annotate = None
    random_for_active = True
    valid_search_terms = set(dtm_search.columns)
    params = {'loss': ['log'],
                  'penalty': ['elasticnet'],
                  'alpha': [0.00001, 0.001], #regulariz. weight
                  'l1_ratio': [0,1]}          #balance l1 l2 norm
 
    mod = SGDClassifier()
    clf_active = GridSearchCV(estimator=mod, param_grid=params, n_jobs=10)
    clf_random = GridSearchCV(estimator=mod, param_grid=params, n_jobs=10)
    #clf_active = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5)
    #clf_random = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5)

    for iteration in range(iterations):
        print(f'iteration {iteration}')
        # Create default output object
        iter_out = {'iteration': iteration, 
                    'selected_search': None,
                    'selected_clf_random': None,
                    'selected_clf_active': None,
                    'clf_random_success': False,
                    'clf_active_success': False}
        
        
        # Draw next keyword if necessary
        if next_keyword is None:
            print('Next keyword is none')
            next_keyword, seed_keywords = draw_keywords(
                    1, seed_keywords[['word', 'weight']]
                    )
            print(f'Drew new keyword: {next_keyword}')

        # Add the next keyword to search term list and store in output
        keywords.extend(next_keyword)
        print(f'Keywords: {keywords}')
        iter_out['search_terms'] = keywords
       
        # Query the data with the keywords
        keywords = [w for w in keywords if w in valid_search_terms]
        if len(keywords) == 0: 
            out.append(iter_out)
            print(f'No valid keywords')
            next_keyword = None
            print('CONTINUE')
            #continue

        df['keyword_relevant'] = dtm_search[keywords].sum(axis=1) > 0

        # Store the search selection
        iter_out['selected_search'] = df[df['keyword_relevant']].index
        
        n = len(iter_out['selected_search'])
        print(f'# Query results: {n}')
        kw_rel = df[df['keyword_relevant']].index

        # Check if there are samples left for annotation
        selected_not_annotated_random = df[df.keyword_relevant & 
                                           ~df.annotated_random].index
        selected_not_annotated_active = df[df.keyword_relevant & 
                                           ~df.annotated_active].index

        train_random = (len(selected_not_annotated_random) > 0)
        train_active = (len(selected_not_annotated_active) > 0)
        print(f'train_random: {train_random}')
        print(f'train_active: {train_active}')

        
        # Random classifier
        if train_random:
            # Annotate a random selection
            n_selected_not_annotated = len(selected_not_annotated_random)
            n_to_annotate = min(n_selected_not_annotated, n_annotate_step)
            to_annotate_random = np.random.choice(selected_not_annotated_random, 
                                                  n_to_annotate, 
                                                  replace=False)
            df.loc[to_annotate_random, 'annotated_random'] = True
            nar = df['annotated_random'].sum()
            print(f'# annotated random: {nar}')
            
            # Check if the data allows to train the clf
            train_random, _ = check_sample(df, n_annotate_step)            

            if train_random:
                # Train the model
                annotated = df[df.annotated_random].index
                clf_random.fit(dtm_clf.loc[annotated].as_matrix(), 
                        df.annotation.loc[df.annotated_random])
                iter_out['clf_random_success'] = True
                print('Random learner trained')
                # Classify all query results
                pred = clf_random.predict(dtm_clf.loc[kw_rel]).astype(bool)
                df.loc[kw_rel, 'clf_random_relevant'] = pred
                iter_out['selected_clf_random'] = \
                        df[df['clf_random_relevant']].index
                ns = len(iter_out['selected_clf_random'])
                print(f'# selected by random learner: {ns}')


            else:
                print('Class conditions not fulfilled for random')
                iter_out['selected_clf_random'] = iter_out['selected_search']

        # Active learning classifier
        if train_active:
            # Annotate the sample (if first iteration or if no probabilities
            # from last iteration use a random sample
            if random_for_active:
                print('annotating randomly for active learner')
                n_selected_not_annotated = len(selected_not_annotated_active)
                n_to_annotate = min(n_selected_not_annotated, n_annotate_step)
                to_annotate_active = np.random.choice(selected_not_annotated_active, 
                                                      n_to_annotate, 
                                                      replace=False)
                df.loc[to_annotate_active, 'annotated_active'] = True
                naa = df['annotated_active'].sum()
                print(f'# annotated active: {naa}')
     
                random_for_active = False

            # Check if the data allows to train the clf
            _, train_active = check_sample(df, n_annotate_step)            
            
            if train_active:
                # Train the model
                annotated = df[df.annotated_active].index
                clf_active.fit(dtm_clf.loc[annotated].as_matrix(), 
                        df.annotation.loc[df.annotated_active])
                iter_out['clf_active_success'] = True
                print('Active learner trained')

                # Predict for all query results
                pred = clf_active.predict_proba(dtm_clf.loc[kw_rel])[:, 1]
                df.loc[kw_rel, 'clf_active_relevant'] = pred >= 0.5
                iter_out['selected_clf_active'] = \
                        df[df['clf_active_relevant']].index
                ns = len(iter_out['selected_clf_active'])
                print(f'# selected by active learner: {ns}')

                # Find tweets to annotate next round (for active learner)
                kw_rel_not_annotated = df[df['keyword_relevant'] & 
                                          ~df['annotated_active']].index
                n_krna = len(kw_rel_not_annotated)
                if n_krna == 0:
                    random_for_active = True
                else:
                    all_probs = pd.Series(pred)
                    all_probs.index = kw_rel
                    probs = all_probs.loc[kw_rel_not_annotated]

                    n_to_annotate = min(n_krna, n_annotate_step)
                    to_annotate_active = np.argsort((0.5 - probs)**2)[:n_to_annotate]
                    df.loc[kw_rel_not_annotated[to_annotate_active], 
                           'annotated_active'] = True
                    naa = df['annotated_active'].sum()
                    print(f'# annotated for active learning: {naa}') 

                # Expand query for next iteration

                ## Check if there are predictions for both classes
                ncp = df['clf_active_relevant'].sum()

                if ncp > 0 and ncp < n:
                    print('Getting new keywords')
                    word_scores, _ = fighting_words(df, dtm_search)
                    word_scores.sort_values(ascending=False, inplace=True)
                    prop = word_scores[~word_scores.index.isin(keywords)].iloc[:50]
                    next_keyword = None
                    # First check if there is a word that has been chosen in
                    # previous iterations. If so choose that:
                    for w in prop.index:
                        if w in accepted_keywords:
                            next_keyword = [w]
                    #if next_keyword is None:
                    #    #for w in prop.index:
                    #    #    if w not in accepted_keywords:
                    #    #        print(w)
                    #    print(prop)
                    #    while True:
                    #        inp = input(f"Type new word: ")
                    #        if inp == '':
                    #            next_keyword = None
                    #        else:
                    #            accepted_keywords.append(inp)
                    #            next_keyword = [inp]
                    #        break
                else:
                    next_keyword = None

            else:
                print('Class conditions not fulfilled for active')
                iter_out['selected_clf_active'] = iter_out['selected_search']
                random_for_active = True
                next_keyword = None

        else:
            iter_out['selected_clf_active'] = iter_out['selected_search']
            random_for_active = True
            next_keyword = None
        
        out.append(iter_out)

    return out
    


def check_sample(df, n_annotate_step):
    '''
    Check if the current datastructure allows training of clfs
    '''
    random_good = True
    active_good = True
    n_annotated = df['annotated_random'].sum()
    n_positive_random = df[df['annotated_random']].annotation.sum()
    n_positive_active = df[df['annotated_active']].annotation.sum()

    if n_positive_random < 2 or (n_annotated - n_positive_random) < 2:
        random_good = False
    if n_positive_active < 2 or (n_annotated - n_positive_active) < 2:
        active_good = False

    return random_good, active_good

def evaluate_selection(selection):
    if selection is None:
        selection = []
    # Precision and recall
    if len(selection) > 0:
        y_true = df['annotation']
        y_pred = expand_vector(selection, len(y_true))
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
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

def expand_vector(sparse_vector, length):
    out = np.full(length, False, bool)
    out[sparse_vector] = True
    return out


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

    # Create sql connection
    print('Connecting to DB...')
    _, engine = make_session()

    # Get the data  
    query = ('SELECT cr_results.tweet_id,' 
             '       cr_results.annotation,' 
             '       cr_results.trust,'
             '       tweets.text,'
             '       tweets.created_at,'
             '       users.screen_name '
             'FROM cr_results '
             'INNER JOIN tweets ON tweets.id=cr_results.tweet_id '
             'INNER JOIN users ON users.id=tweets.user_id ')

    print('Getting the data...')
    df = pd.read_sql(sql=query, con=engine)
    
   
    df.replace(to_replace=['relevant', 'irrelevant', 'None'], 
               value=[1,0,np.nan], inplace=True)
    
    # Select only judgements from trusted contributors
    df = df[['tweet_id', 'annotation', 'text', 
             'screen_name', 'created_at']].loc[df['trust'] > 0.8]

    # Aggregate to one judgement per tweet
    def f(x):
         return pd.Series({'annotation': x['annotation'].mean(),
                           'text': x['text'].iloc[0],
                           'tweet_id': x['tweet_id'].iloc[0],
                           'screen_name': x['screen_name'].iloc[0],
                           'created_at': x['created_at'].iloc[0]})

    print('Aggregating...')
    # Convert tweet timestamp to utc (is in utc but has tz aware format)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df = df[['annotation', 'text', 'tweet_id', 
             'screen_name', 'created_at']].groupby('tweet_id').apply(f)


    df = df[['annotation', 'text', 'screen_name', 'created_at']]
    
    # Make annotations binary
    df.loc[df['annotation'] >= 0.5, 'annotation'] = 1
    df.loc[df['annotation'] < 0.5, 'annotation'] = 0

    # For some reason a few tweets outside of the 2015 snuck in, remove them
    df = df[df.created_at.dt.year == 2015]
 
    df.reset_index(inplace=True)

    print('Loading nlp pipeline...')
    parser = TextParser('de')
    print('Generating document term matrices...')
    dtm_normalized = make_dtm(df.text, parser, normalize=True)
    dtm_non_normalized = make_dtm(df.text, parser, normalize=False)
    dtm_hashtags = make_index(df.text, df.screen_name, parser, 'hashtags')
    dtm_users = make_index(df.text, df.screen_name, parser, 'users')

    # ground truth distributions for users and hastags
    gt_hashtags = dtm_hashtags[df['annotation'] == 1].sum(axis=0)
    gt_hashtags /= gt_hashtags.sum()
    gt_hashtags = gt_hashtags.values.reshape(1, -1)
    gt_users = dtm_users[df['annotation'] == 1].sum(axis=0)
    gt_users /= gt_users.sum()
    gt_users = gt_users.values.reshape(1, -1)

    gt_timeline = df['tweet_id'].groupby(df.created_at.dt.dayofyear).count()
    gt_timeline /= gt_timeline.sum()
    gt_timeline = gt_timeline.values.reshape(1, -1)

    # get the crowdflower words for the keywords
    reports = glob.glob('../data/cf_report*')
    words = []
    for r in reports:
        with open(r, 'r') as infile:
            for line in infile:
                try:
                    w = line.strip('"\n').split(',"')[1]
                    ws = w.split(',')
                    print(ws)
                    words.extend(ws)
                except IndexError:
                    pass
    
    clean_words = [clean(w, parser) for w in words if clean(w, parser) is not None]
    survey_keywords = {}
    for w in clean_words:
        survey_keywords[w] = survey_keywords.get(w, 0) + 1
    
    # calculate weights
    kwords = pd.DataFrame([[k,survey_keywords[k]] for k in survey_keywords],
                       columns=['word', 'count'])
    kwords['weight'] = kwords['count'] / kwords['count'].sum()

    # write to file for the table
    kwords.sort_values(by='count', ascending=False).to_csv(
            '../data/crowdflower_keywords.csv', index=False
            )

    # get the baseline scores (keywords additive in order)
    # ==========================================================================
    
    ### prepare data with keyword indicator variables
    print('searching all keywords in tweets')
    for k in kwords.word:
        df[k] = false

    for index, row in df.iterrows():
        text = row['text']
        tokens = set([x.lower() for x in parser.tokenize(text)])
        for k in kwords.word:
            if k in tokens:
                df.loc[index, k] = True


    replications = 100
    iterations = 100
    # =========================================================================
    # get the baseline scores (random draws of increasing number of keywords)
    # =========================================================================
    print('getting baseline scores...')

    n_words = iterations
    stats = {'replication': [], 'iteration': [], 'method': [], 'measure': [],
             'value': []}

    for replication in range(replications):
        print(f'replication {replication}')

        # draw initial keywords
        keywords, terms = draw_keywords(1, kwords[['word', 'weight']])

        for i in range(0, n_words):
            if i > 0:
                new_word, terms = draw_keywords(1, terms)
                keywords.extend(new_word)
            n = len(keywords)

            df['keyword_relevant'] = False
               
            column_idx = [list(df.columns).index(w) for w in keywords]
            df['keyword_relevant'] = df.iloc[:, column_idx].sum(axis=1) != 0
            selection = df[df['keyword_relevant']].index
            res = evaluate_selection(selection)
            
            for measure in res:
                stats['replication'].append(replication)
                stats['iteration'].append(i)
                stats['method'].append('keyword')
                stats['measure'].append(measure)
                stats['value'].append(res[measure])

    #pickle.dump(stats, open('stats_temp.p', 'wb'))
    stats = pickle.load(open('stats_temp.p', 'rb'))
            
    # =========================================================================
    # get the active learning scores
    # =========================================================================

    n_annotate_step = 5
    #accepted_keywords = []
    if os.path.exists('used_keywords.p'):
        accepted_keywords = pickle.load(open('used_keywords.p', 'rb'))
    if os.path.exists('results_backup.p'):
        results = pickle.load(open('results_backup.p', 'rb'))
    #
    ### Replications
    #results = []
    #for replication in range(len(results), replications):
    #    print(f'Replication: {replication}')

    #    # Reset everyting
    #    df['keyword_relevant'] = False
    #    df['clf_random_relevant'] = False
    #    df['clf_active_relevant'] = False
    #    df['annotated_active'] = False
    #    df['annotated_random'] = False

    #    results.append(replication_2_step(df, iterations,
    #                                      kwords, dtm_non_normalized, 
    #                                      dtm_normalized, n_annotate_step))
    ## Back up results
    #pickle.dump(results, open('results_backup.p', 'wb'))
    #pickle.dump(accepted_keywords, open('used_keywords.p', 'wb'))

    # Analyze the results
    for r, iterations in enumerate(results):
        print(r)
        for i, iteration in enumerate(iterations):
            for method in ['search', 'clf_random', 'clf_active']:
                selection = iteration[f'selected_{method}']
                this_stats = evaluate_selection(selection)
                for measure in ['precision', 'recall', 'f1', 'user_similarity',
                                'timeline_similarity', 'hashtag_similarity']:
                    stats['replication'].append(r)
                    stats['iteration'].append(i)
                    stats['method'].append(method)
                    stats['measure'].append(measure)
                    stats['value'].append(this_stats[measure])

    output = pd.DataFrame(stats) 
    output.to_csv('../data/experiment_results.csv', index=False)
    sys.exit()
    # Analyze the queries

    ## Count them
    qe_terms = {}
    total_expansion = 0
    queries = [r[99]['search_terms'] for r in results]
    for query in queries:
        for word in query:
            qe_terms[word] = qe_terms.get(word, 0) + 1
            total_expansion += 1

    # Normalize expansion and query terms
    for term in qe_terms:
        qe_terms[term] = round(qe_terms[term] / total_expansion, 4)

    for term in survey_keywords:
        survey_keywords[term] = round(survey_keywords[term] / kwords['count'].sum(), 4)

    expansion_terms = sorted(qe_terms.items(), key=lambda x: x[1],
                             reverse=True)
    survey_terms = sorted(survey_keywords.items(), key=lambda x: x[1],
                          reverse=True)
    
    # Write top n to table
    n = 293
    ts = pd.DataFrame({'expansion_term': [x[0] for x in expansion_terms[:n]],
                       'expansion_proportion': [x[1] for x in expansion_terms[:n]],
                       'survey_term': [x[0] for x in survey_terms[:n]],
                       'survey_proportion_': [x[1] for x in survey_terms[:n]]})
    ts.to_latex('../paper/tables/top_terms.tex')


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Boolean vs clf experiment
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
            dtm_normalized.as_matrix(), df['annotation'].as_matrix()) 
    helper.fit(X_train, y_train, scoring=scorer, n_jobs=12)



    pred = clf.predict(X_test)
    get_metrics(y_test, pred)

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

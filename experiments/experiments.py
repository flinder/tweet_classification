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
            return True
    return False


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

def clean(word):
    w = word.lstrip()
    w = w.strip()
    w = w.lower()
    w = process_text(w, stemmer)
    if len(w) != 1:
        return None
    w = w[0]
    w = transform_umlaute(w)
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
    
  
    # Get the crowdflower words for the keywords
    reports = glob.glob('../data/cf_report*')
    words = []
    for r in reports:
        with open(r, 'r') as infile:
            for line in infile:
                try:
                    w = line.strip('"\n').split(',"')[1]
                    ws = w.split(',')
                    words.extend(ws)
                except IndexError:
                    pass
    
    clean_words = [clean(w) for w in words if clean(w) is not None]
    survey_keywords = {}
    for w in clean_words:
        survey_keywords[w] = survey_keywords.get(w, 0) + 1
    
    # Calculate weights
    kwords = pd.DataFrame([[k,survey_keywords[k]] for k in survey_keywords],
                       columns=['word', 'count'])
    kwords['weight'] = kwords['count'] / kwords['count'].sum()

    # Write to file for the table
    kwords.sort_values(by='count', ascending=False).to_csv(
            '../data/crowdflower_keywords.csv', index=False
            )

    # Get the baseline scores (keywords additive in order)
    # ==========================================================================
    
    ### Prepare data with keyword indicator variables
    for i,k in enumerate(kwords.word):
        df[k] = [search_keywords(x, [k]) for x in df.text]

    #logging.info('Getting baseline scores...')

    # Get the baseline scores (random draws of increasing number of keywords)


    ## Everything we need for the classifiers
    params = {'loss': ['log'],
                  'penalty': ['elasticnet'],
                  'alpha': np.linspace(0.00001, 0.001, 5), #regulariz. weight
                  'l1_ratio': np.linspace(0,1,5)}          #balance l1 l2 norm
 
    mod = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5)
    clf = GridSearchCV(estimator=mod, param_grid=params, n_jobs=3)

    
    ## Generate document term matrix for all tweets for classifier
    unigrams, bigrams = make_dictionaries(df.text, process_text, stemmer)    
    X_full = make_doc_term_mat(df, process_text, stemmer, unigrams, bigrams)


    n_words = 29
    replications = 21
    scores = []
    
    for replication in range(replications):
        logging.info(f'Replication: {replication}')
        
        # Draw initial keywords
        keywords, terms = draw_keywords(4, kwords[['word', 'weight']])

        for i in itertools.repeat(None, n_words - 4):
            # add a keyword
            new_word, terms = draw_keywords(1, terms)
            keywords.extend(new_word)
            n = len(keywords)
            logging.info(f'Number of keywords: {n}')

            df['keyword_relevant'] = False
            df['clf_relevant'] = False
            df['annotated'] = False
            metrics_kw = [np.nan] * 3
            metrics_clf = [np.nan] * 3
               
            column_idx = [list(df.columns).index(w) for w in keywords]
            df['keyword_relevant'] = df.iloc[:, column_idx].sum(axis=1) != 0
            n_selected_kw = df.keyword_relevant.sum()

            # Check if there are tweets from both classes
            n_pos = df[df.keyword_relevant].annotation.sum()
            if n_pos == 0:
                logging.info('No positive samples')
                metrics_kw = [0] * 3
                metrics_clf = [0] * 3
                continue
            
            else:
                # Assess performance of keywords alone
                metrics_kw = get_metrics(df.annotation, df.keyword_relevant)

                # Assess performance of keywords + clf
                
                ## Annotate some of the data randomly
                n_to_annotate = n_selected_kw // 10
                to_annotate = np.random.choice(df[df.keyword_relevant].index, 
                                               n_to_annotate, replace=False)
                df.loc[to_annotate, 'annotated'] = True
                
                # Check if there a0re samples from both classes
                annot_sum = df.annotation.loc[df.annotated].sum()
                if annot_sum == n_to_annotate or annot_sum == 0:
                    logging.info('Only one class of sample in annotated data')
                    metrics_clf = [0] * 3
                else:
                    # Train the clf on annotated data
                    try:
                        clf.fit(X_full[to_annotate, :], 
                                df.annotation.loc[df.annotated])
                    except:
                        logging.info('clf_error')
                        break
                    kw_rel_idx = df[df.keyword_relevant].index

                    # Make prediciton for all selected data
                    pred = clf.predict(X_full[kw_rel_idx, :]).astype(bool)
                    df.loc[kw_rel_idx, 'clf_relevant'] = pred
                    
                    # Evaluate
                    metrics_clf = get_metrics(df.annotation, df.clf_relevant)
                        
            
            out = metrics_kw + metrics_clf + [n_selected_kw, n, replication]
            print(out)
            scores.append(out)

    with open('../data/experiment_results.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['kw_precision','kw_recall','kw_f1', 
                         'clf_precision','clf_recall','clf_f1',
                         'n_selected', 'n_keywords', 
                         'replication'])
        for s in scores:
            writer.writerow(s)




    # Get the active learning scores
    iterations = 50
    replications = 50
    n_annotate_step = 20
    scores = []
    queries = []
    negatives = []

    ## Everything we need for the classifiers
    params = {'loss': ['log'],
                  'penalty': ['elasticnet'],
                  'alpha': np.linspace(0.00001, 0.001, 5), #regulariz. weight
                  'l1_ratio': np.linspace(0,1,5)}          #balance l1 l2 norm
 
    mod = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5)
    clf = GridSearchCV(estimator=mod, param_grid=params, n_jobs=3)



    ## Replications
    for replication in range(replications):

        logging.info(f'Replication: {replication}')

        # Reset everyting
        df['keyword_relevant'] = False
        df['clf_relevant'] = False
        df['annotated'] = False
        metrics_clf = [np.nan] * 3
        metrics_kw = [np.nan] * 3

        # Choose seed keyword
        keywords = list(np.random.choice(kwords['word'], 5, 
                                    replace=False, p=kwords['weight']))
        logging.info(f'Seed keyword: {keywords}')

        # Run first query expansion than active learning clf up to 100 words
        for iteration in range(iterations):
            logging.info(f'Iteration {iteration}')
            if iteration > 0:
                # Expand query
                keywords = keywords + [new_keyword]
                logging.info(f'Querying with {keywords}')

            # Query with the current keywords
            df['keyword_relevant'] = [search_keywords(t, keywords) for t in df.text]
            n_selected_kw = df.keyword_relevant.sum()
            logging.info(f"Query returned {n_selected_kw} tweets")

            # Check if there are tweets from both classes
            n_pos = df[df.keyword_relevant].annotation.sum()
            if n_pos == 0:
                logging.info('No positive samples')
                metrics_kw = [0] * 3
                metrics_clf = [0] * 3
                continue
            else:
                metrics_kw = get_metrics(df.annotation, df.keyword_relevant)
 

            # Annotate data (random draw) in first iteration
            if iteration == 0:
                n_to_annotate = min(n_selected_kw, n_annotate_step)
                to_annotate = np.random.choice(df[df.keyword_relevant].index, 
                                               n_to_annotate, replace=False)
                df.loc[to_annotate, 'annotated'] = True
            
            # Check if there are samples from both classes in annotated data
            annot_sum = df.annotation.loc[df.annotated].sum()
            n_samples = df.annotated.sum()
            if annot_sum == n_samples or annot_sum == 0:
                logging.info('Only one class of sample in annotated data')
                metrics_clf = [0] * 3
                out = metrics_kw + metrics_clf + [replication, iteration] 
                scores.append(out)
                break
            else:
                # Train the clf on annotated data

                logging.info("Training clf...")
                logging.info(f"Number of annotated samples: {n_samples}")
                logging.info(f"Number of annotated positives: {annot_sum}")

                annotated = df[df.annotated].index
                clf.fit(X_full[annotated, :], 
                        df.annotation.loc[df.annotated])

                kw_rel_not_annot = df[df.keyword_relevant & ~df.annotated].index
                kw_rel = df[df.keyword_relevant].index

                # Make prediciton from clf for all data selected by query
                n = len(kw_rel)
                logging.info(f"Classifying all {n} queried samples")
                pred = clf.predict(X_full[kw_rel, :]).astype(bool)

                df.loc[kw_rel, 'clf_relevant'] = pred
                n_selected_clf = df.clf_relevant.sum()
                
                # Evaluate
                metrics_clf = get_metrics(df.annotation, df.clf_relevant)
                out = metrics_kw + metrics_clf + [replication, iteration, 
                                                  n_selected_kw, n_selected_clf] 
                scores.append(out)
                logging.info(out[:6])

                # Choose tweets for annotation in next round

                ## Get predicted probabilities for queried but not annotated
                # tweets
                n = len(kw_rel_not_annot)
                logging.info(f"Predicting probability for remaining {n} samples")
                try:
                    probs = clf.predict_proba(X_full[kw_rel_not_annot, :])[:, 1]

                    n_to_annotate = min(n_selected_kw, n_annotate_step)
                    logging.info(f"Annotating {n_to_annotate} new samples")
                    to_annotate = np.argsort((0.5 - probs)**2)[:n_to_annotate]
                    df.loc[kw_rel_not_annot[to_annotate], 'annotated'] = True
                except ValueError as e:
                    logging.error(f"Error in prediction: {e}")
               

                # Select new query terms
                ## Train new clf
                pos_annotated = df[(df.annotated) & (df.annotation == 1)].index
                nr = np.random.choice(df[df.keyword_relevant == False].index, 
                                      len(pos_annotated), replace=False)
                sel = list(pos_annotated) + list(nr)
                clf.fit(X_full[sel, :], np.array([1]*len(nr)+[0]*len(nr)))
                
                i = 1
                while True:
                    prop = unigrams[np.argsort(clf.best_estimator_.coef_)[0][-i]]
                    if prop in keywords or prop in negatives:
                        i += 1
                    else:
                        inp = input(f"Proposal: {i:}, {prop}")
                        if inp == 'y':
                            new_keyword = prop
                            break
                        elif inp == 'n':
                            i += 1
                            negatives.append(prop)
                            continue
                        else:
                            break

                logging.info(f'New keyword: {new_keyword}')

        queries.append(keywords)
    
    
    with open('../data/full_system_scores_1.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['precision_kw','recall_kw','f1_kw', 'precision_clf', 
                         'recall_clf','f1_clf','replication', 'iteration',
                         'n_selected_kw', 'n_selected_clf'])
        for s in scores:
            writer.writerow(s)


    # Analyze the queries

    ## Count them
    qe_terms = {}
    total_expansion = 0
    for query in queries:
        for word in query:
            qe_terms[word] = qe_terms.get(word, 0) + 1
            total_expansion += 1

    # Normalize expansion and query terms
    for term in qe_terms:
        qe_terms[term] = round(qe_terms[term] / total_expansion, 3)

    for term in survey_keywords:
        survey_keywords[term] = round(survey_keywords[term] / kwords['count'].sum(), 3)

    expansion_terms = sorted(qe_terms.items(), key=lambda x: x[1],
                             reverse=True)
    survey_terms = sorted(survey_keywords.items(), key=lambda x: x[1],
                          reverse=True)
    
    # Write top 10 to csv
    with open('../data/survey_expansion_terms.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['rank', 'survey_term', 'proportion', 'expansion_term',
                         'proportion'])
        for st, et, i in zip(survey_terms, expansion_terms, range(0, 10)):
            writer.writerow([i+1, st[0], st[1], et[0], et[1]])



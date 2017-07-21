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
from scipy.special import gamma as G


sys.path.append('../../dissdat/database/')
from db import make_session


class TextParser(object):

    def __init__(self, language):
        self.parser = spacy.load(language)
        self.stemmer = Stemmer.Stemmer('german').stemWord 
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

def make_dtm(documents, parser, normalize=True):
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
    clf_rel = df['clf_relevant']
    X = dtm.loc[kw_rel].groupby(clf_rel.loc[kw_rel]).sum()
    n = X.sum(axis=1)
    N = n.sum()
    y = X.sum(axis=0)
    a0 = 1000
    a = (a0 * y) / N
    a0i = a0 * n / N
    # Odds ratios for the irrelevant documents
    row = X.loc[False]
    irel_ratios = (np.log((row+a)/(row.sum()+a0i.iloc[0]-row-a))
                   -np.log((y+a)/(N+a0-y-a)))
    # Odds ratios for the relevant documents
    row = X.loc[True]
    rel_ratios = (np.log((row+a)/(row.sum()+a0i.iloc[1]-row-a))
                  -np.log((y+a)/(N+a0-y-a)))


    return rel_ratios, irel_ratios


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

    df.reset_index(inplace=True)

    logging.info('Loading nlp pipeline...')
    parser = TextParser('de')
    logging.info('Generating document term matrices...')
    dtm_normalized = make_dtm(df.text, parser, normalize=True)
    dtm_non_normalized = make_dtm(df.text, parser, normalize=False)

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
    
    clean_words = [clean(w, parser) for w in words if clean(w, parser) is not None]
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
    logging.info('Searching all keywords in tweets')
    for k in kwords.word:
        df[k] = False

    for index, row in df.iterrows():
        text = row['text']
        tokens = set([x.lower() for x in parser.tokenize(text)])
        for k in kwords.word:
            if k in tokens:
                df.loc[index, k] = True


    # =========================================================================
    # Get the baseline scores (random draws of increasing number of keywords)
    # =========================================================================
    logging.info('Getting baseline scores...')

    n_words = 50
    replications = 21
    scores = []
    
    for replication in range(replications):
        logging.info(f'Replication {replication}')

        # Draw initial keywords
        keywords, terms = draw_keywords(1, kwords[['word', 'weight']])

        for i in range(0, n_words):
            if i > 0:
                new_word, terms = draw_keywords(1, terms)
            keywords.extend(new_word)
            n = len(keywords)

            df['keyword_relevant'] = False
               
            column_idx = [list(df.columns).index(w) for w in keywords]
            df['keyword_relevant'] = df.iloc[:, column_idx].sum(axis=1) != 0
            n_selected_kw = df.keyword_relevant.sum()

            # Check if there are tweets from both classes
            n_pos = df[df.keyword_relevant].annotation.sum()
            if n_pos == 0:
                metrics_kw = [0] * 3
            else:
                # Assess performance of keywords alone
                metrics_kw = get_metrics(df.annotation, df.keyword_relevant)
            
            out = metrics_kw + [n_selected_kw, n, replication]
            scores.append(out)
            
    scores_df = pd.DataFrame(scores, columns=['kw_precision','kw_recall','kw_f1', 
                                              'n_selected', 'n_keywords', 
                                              'replication'])
    scores_df.to_csv('../data/keyword_only_results.csv', index=False)

    # =========================================================================
    # Get the active learning scores
    # =========================================================================

    iterations = 50
    replications = 20
    n_annotate_step = 5
    scores = []
    queries = []
    negatives = []
    selections = []
    stored_kw_entries = []

    ## Everything we need for the classifiers
    params = {'loss': ['log'],
                  'penalty': ['elasticnet'],
                  'alpha': np.linspace(0.00001, 0.001, 5), #regulariz. weight
                  'l1_ratio': np.linspace(0,1,5)}          #balance l1 l2 norm
 
    #mod = SGDClassifier()
    #clf = GridSearchCV(estimator=mod, param_grid=params, n_jobs=10)
    clf = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.5)

    valid_search_terms = set(dtm_non_normalized.columns)

    if os.path.exists('full_sys_scores_backup.p'):
        scores = pickle.load(open('full_sys_scores_backup.p', 'rb'))
    if os.path.exists('selections.p'):
        selections = pickle.load(open('selections.p', 'rb'))

    start_replication = 0 
    try:
        last_iteration = scores[-1]
        start_replication = last_iteration[6] + 1
    except IndexError:
        pass

    ## Replications
    for replication in range(start_replication, replications):

        logging.info(f'Replication: {replication}')

        # Reset everyting
        df['keyword_relevant'] = False
        df['clf_relevant'] = False
        df['annotated'] = False
        metrics_clf = [np.nan] * 3
        metrics_kw = [np.nan] * 3

        # Choose seed keyword
        keywords, terms = draw_keywords(1, kwords[['word', 'weight']])

        # Run first query expansion than active learning clf
        to_annotate = None
        iter_res = []
        iter_selection = []

        for iteration in range(iterations):
            logging.info(f'Iteration {iteration}')
            if iteration > 0:
                # Expand query
                keywords = keywords + new_keyword
                logging.info(f'Querying with {keywords}')

            # Query with the current keywords
            keywords = [w for w in keywords if w in valid_search_terms]
            if len(keywords) == 0:
                logging.info('No valid seed keywords, adding seed word')
                new_keyword, terms = draw_keywords(1, terms)
                metrics_kw = [0] * 3
                metrics_clf = [0] * 3
                out = metrics_kw + metrics_clf + [replication, iteration, 0, 0] 
                iter_res.append(out)
                continue
            
            # Search for keywords in dtm
            df['keyword_relevant'] = dtm_non_normalized[keywords].sum(axis=1) > 0

            n_selected_kw = df.keyword_relevant.sum()
            iter_selection.append([replication, iteration, df[df['keyword_relevant']].index])
            logging.info(f"Query returned {n_selected_kw} tweets")

            # Check if there are tweets from both classes
            n_pos = df[df.keyword_relevant].annotation.sum()
            #if n_pos == 0 or df[df.keyword_relevant].shape[0] < 30:
            if n_pos == 0:
                logging.info('No positive samples, adding seed word')
                metrics_kw = [0] * 3
                metrics_clf = [0] * 3
                new_keyword, terms = draw_keywords(1, terms)
                out = metrics_kw + metrics_clf + [replication, iteration, 
                                                  n_selected_kw, 0] 
                iter_res.append(out)
                pickle.dump(scores, open('full_sys_scores_backup.p', 'wb'))
                pickle.dump(selections, open('selections.p', 'wb'))
                continue
            else:
                metrics_kw = get_metrics(df.annotation, df.keyword_relevant)
         

            # Annotate data (random draw) in first iteration
            if to_annotate is None:
                n_to_annotate = min(n_selected_kw, n_annotate_step)
                to_annotate = np.random.choice(df[df.keyword_relevant].index, 
                                               n_to_annotate, replace=False)
                df.loc[to_annotate, 'annotated'] = True
            
            # Check if there are samples from both classes in annotated data
            annot_sum = df.annotation.loc[df.annotated].sum()
            n_samples = df.annotated.sum()
            if annot_sum == n_samples or annot_sum < 2:
                logging.info('Only one class of sample in annotated data, '
                             'adding seed keyword')
                logging.info(f'annot_sum: {annot_sum}')
                new_keyword, terms = draw_keywords(1, terms)
                to_annotate = None
                metrics_clf = [0] * 3
                out = metrics_kw + metrics_clf + [replication, iteration,
                                                  n_selected_kw, 0] 
                iter_res.append(out)
                continue
            else:
                # Train the clf on annotated data

                logging.info("Training clf...")
                logging.info(f"Number of annotated samples: {n_samples}")
                logging.info(f"Number of annotated positives: {annot_sum}")

                annotated = df[df.annotated].index
                try:
                    clf.fit(dtm_normalized.loc[annotated].as_matrix(), 
                            df.annotation.loc[df.annotated])
                except Exception as e:
                    logging.info("Error in training clf: {e}")
                    metrics_clf = [0] * 3
                    out = metrics_kw + metrics_clf + [replication, iteration,
                                                      n_selected_kw, 0] 
                    continue
 
                kw_rel_not_annot = df[df.keyword_relevant & ~df.annotated].index
                kw_rel = df[df.keyword_relevant].index

                # Make prediciton from clf for all data selected by query
                n = len(kw_rel)
                logging.info(f"Classifying all {n} queried samples")
                pred = clf.predict(dtm_normalized.loc[kw_rel]).astype(bool)

                df.loc[kw_rel, 'clf_relevant'] = pred
                n_selected_clf = df.clf_relevant.sum()
                
                # Evaluate
                metrics_clf = get_metrics(df.annotation, df.clf_relevant)
                out = metrics_kw + metrics_clf + [replication, iteration, 
                                                  n_selected_kw, n_selected_clf] 
                iter_res.append(out)
                logging.info(out[:6])

                # Choose tweets for annotation in next round

                ## Get predicted probabilities for queried but not annotated
                # tweets
                n = len(kw_rel_not_annot)
                logging.info(f"Predicting probability for remaining {n} samples")
                try:
                    probs = clf.predict_proba(dtm_normalized.loc[kw_rel_not_annot])[:, 1]

                    n_to_annotate = min(n_selected_kw, n_annotate_step)
                    logging.info(f"Annotating {n_to_annotate} new samples")
                    to_annotate = np.argsort((0.5 - probs)**2)[:n_to_annotate]
                    df.loc[kw_rel_not_annot[to_annotate], 'annotated'] = True
                except ValueError as e:
                    logging.error(f"Error in prediction: {e}")
               

                # Select new query terms
                ## Train new clf
                word_scores, _ = fighting_words(df, dtm_non_normalized)
                word_scores.sort_values(ascending=False, inplace=True)
                prop = word_scores[~word_scores.index.isin(keywords)].iloc[:50]
                # First check if there is a word that has been chosen in
                # previous iterations. If so choose that:
                new_keyword = None
                for w in prop.index:
                    if w in stored_kw_entries:
                        new_keyword = [w]
                if new_keyword is None:
                    print(prop)
                    while True:
                        inp = input(f"Type new word: ")
                        stored_kw_entries.append(inp)
                        if inp in keywords:
                            print('Already in keywords')
                            continue
                        else:
                            new_keyword = [inp]
                            break
                    continue

                logging.info(f'New keyword: {new_keyword}')

        scores.extend(iter_res)
        selections.extend(iter_selection)
        queries.append(keywords)

        # store backup
        pickle.dump(scores, open('full_sys_scores_backup.p', 'wb'))
        pickle.dump(selections, open('selections.p', 'wb'))
        pickle.dump(queries, open('queries.p', 'wb'))
        pickle.dump(stored_kw_entries, open('used_keywords.p', 'wb'))

    
    
    with open('../data/full_system_scores_2.csv', 'w') as outfile:
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



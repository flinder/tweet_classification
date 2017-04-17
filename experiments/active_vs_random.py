import csv
import spacy
import re
import logging
import sys

import pandas as pd
import numpy as np

from gensim import corpora, matutils
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from multiprocessing import Pool

sys.path.append('../../dissdat/database/')
from db import make_session

def bi_from_uni(unis):
        bis = [] 
        for i, _ in enumerate(unis):
            try:
                bigram = '{}_{}'.format(unis[i], unis[i+1])
                bis.append(bigram)
            except IndexError:
                break
        return bis

def make_dictionaries(documents, parser):
    
    unigrams = corpora.Dictionary()
    bigrams = corpora.Dictionary()

    for d in documents:
        # Get unigrams (lemmas)
        unis = [token.lemma_ for token in parser(d)] 
        # Get bigrams
        bis = bi_from_uni(unis)
        # update dictionaries
        unigrams.doc2bow(unis, allow_update = True)
        bigrams.doc2bow(bis, allow_update = True)

    return unigrams, bigrams


def label_iter(input_file_name):
    with open(input_file_name, 'r', encoding='utf-8') as infile:
        header = next(infile)
        reader = csv.reader(infile, delimiter=',', quotechar='"')
        for row in reader:
            yield row[3]


def make_ngrams(tweet_text, parser):
    # Tokenize and make bigrams
    unis = [token.lemma_ for token in parser(tweet_text)] 
    bis = bi_from_uni(unis) # Get bow representation for dictionary terms
    uni_bow = unigrams.doc2bow(unis)
    bi_bow = bigrams.doc2bow(bis)

    return uni_bow, bi_bow 
    
def get_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return [prec, rec, f1]

def iteration(i):
    print(i)
    out = []
    y_train_idx = pd.Series(shuffle(y_train.index))
    ## Random annotation for increasing sample sizes
    for s in range(10, X_train.shape[0], 10):
        annot_idx = y_train_idx.head(s)

        y = y_train.loc[annot_idx]
        X = X_train.loc[annot_idx]

        clf.fit(X, y)

        y_hat = clf.predict(X_test)
        res = [s] + get_metrics(y_test, y_hat) + ['random_learner'] + [i]
        out.append(res)

    annot_idx = y_train_idx.head(10)

    ## active learning annotation for increasing sample sizes
    while True:

        # Get annotated data (as chosen in last iteration)
        y = y_train.loc[annot_idx]
        X = X_train.loc[annot_idx]

        clf.fit(X, y)

        # Get the scores for this iteration
        y_hat = clf.predict(X_test)
        res = [len(annot_idx)] + get_metrics(y_test, y_hat) + ['active_learner'] + [i]
        out.append(res)

        # Get targets for next annotation step
        ## query least certain instances in the training set

        X_remain = X_train.drop(annot_idx, inplace=False, axis=0)
        if X_remain.shape[0] == 0:
            break

        y_hat_al = pd.Series([x[1] for x in clf.predict_proba(X_remain)])
        y_hat_al.index = X_remain.index
        dist_from_05 = (y_hat_al - 0.5)**2
        most_uncertain = dist_from_05.nsmallest(n=10)
        annot_idx = annot_idx.append(pd.Series(most_uncertain.index))
        
    return out

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    # Create sql connection
    logging.info('Connecting to DB...')
    _, engine = make_session()

    # Get the data from db
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
    nlp = spacy.load('en')
     

    # Generate bag of words representation
          
    ## First pass to generate uni and bigram dictionaries
    logging.info('Generating dictionaries...')
    unigrams, bigrams = make_dictionaries(df.text, nlp)

    ## Filter vocabularies
    #unigrams.filter_extremes(no_below=2, no_above=0.5, keep_n=None)
    #bigrams.filter_extremes(no_below=2, no_above=0.5, keep_n=None)

    ### Create document term matrices for uni and bigrams
    logging.info('Generating uni and bigram features...')
    grams = [make_ngrams(d, nlp) for d in df.text]
    unigram_features = matutils.corpus2dense([x[0] for x 
                                            in grams],
                                            num_terms=len(unigrams)).transpose()
    bigram_features = matutils.corpus2dense([x[1] for x 
                                           in grams],
                                           num_terms=len(bigrams)).transpose()

    #X_full = pd.concat([pd.DataFrame(unigram_features),                      
    #                    pd.DataFrame(bigram_features)], axis=1)
    X_full = pd.DataFrame(unigram_features)
    
    ### Get test data
    X_train, X_test, y_train, y_test = train_test_split(X_full, df.annotation, 
                                                        test_size=0.2, 
                                                        random_state=221)
    y_train.index = X_train.index
    y_test.index = X_test.index

    ### Initialize classifier
    clf = SGDClassifier(penalty='elasticnet', loss='log', n_jobs=1)
    #clf = RandomForestClassifier(n_estimators=100, n_jobs=10) 

           
    pool = Pool(10)
    outputs = pool.map(iteration, np.arange(1000))
    pool.close()
    output = []
    for o in outputs:
        output.extend(o)

    with open('scores.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['train_size','precision','recall','f1', 'clf', 'iteration'])
        for s in output:
            writer.writerow(s)

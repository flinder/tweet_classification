import sys
import spacy
import numpy as np
import re
import pickle
import time
import os

from operator import itemgetter
from sqlalchemy import and_, or_
from copy import copy
from gensim import corpora
from gensim.matutils import corpus2csc

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# Custom imports
sys.path.append('../database')
from db import make_session, Tweet
sys.path.append('/home/flinder/Dropbox/current_projects/text_utils')
from text_utils import Cleaner, n_grams, url_regex

def get_miw(dictionary, clf, n=10):
    ## get dictionary items
    d = dictionary.token2id
    for e in d:
        dictionary.id2token[d[e]] = e

    print('*' * 70)
    print('Most predictive of positive:')
    print('*' * 70)
    coefs = clf.coef_.tolist()[0]

    sorted_coefs = sorted(enumerate(coefs),
                          key=itemgetter(1),
                          reverse=True)
    for c in sorted_coefs[:n]:
        print(dictionary.id2token[c[0]])

    print('*' * 70)
    print('Most predictive of negative:')
    print('*' * 70)
    coefs = clf.coef_.tolist()[0]
    sorted_coefs = sorted(enumerate(coefs),
                          key=itemgetter(1),
                          reverse=False)
    for c in sorted_coefs[:n]:
        print(dictionary.id2token[c[0]])


if __name__ == '__main__':

    ## Init text processing tools
    nlp = spacy.load('de')

    # Make sql session
    session = make_session()

    q = session.query(Tweet)

    # Get database iterators
    ## Keyword tweets
    kw_query = or_(Tweet.text.ilike('%flüchtling%'), 
                   Tweet.text.ilike('%fluechtling%'),
                   Tweet.text.ilike('%refugee%'),
                   Tweet.text.ilike('%asyl%'))

    full_query = and_(Tweet.data_group == 'pb_keyword', 
                      Tweet.lang == 'de', kw_query)

    keyword_tweets = q.filter(full_query)

    ## random tweets
    full_query = and_(or_(Tweet.data_group == 'de_stream', 
                          Tweet.data_group == 'de_random'), 
                      Tweet.lang == 'de')

    random_tweets = q.filter(full_query)
    
    # Data Preprocessing
    if os.path.isfile('full_dtm.p'):
        dat = pickle.load(open('full_dtm.p', 'rb'))
        uni_dict = dat['uni_dict']
        X = dat['X']
        y = dat['y']
    else:
        uni_dict = corpora.Dictionary()

        print('Generating dictionary from keyword tweets...')
        for i, tweet in enumerate(keyword_tweets):
            text = tweet.text
            grams = n_grams(text, parser=nlp, n=1)
            uni_dict.add_documents([grams])
            if i % 10000 == 0:
                print(i)

        print('Generating dictinoary from random tweets...')
        for i, tweet in enumerate(random_tweets):
            text = tweet.text
            grams = n_grams(text, parser=nlp, n=1)
            uni_dict.add_documents([grams])
            if i % 10000 == 0:
                print(i)

        print('Raw dictinoary size: {}'.format(len(uni_dict)))

        uni_dict.filter_extremes(no_below=5, no_above=0.8, keep_n=None)

        print('Reduced dictinoary size: {}'.format(len(uni_dict)))

        # Generate document term matrix removing the keywords from the
        # positive samples
        print('Generating corpus...')
        #re_remove_str = ('([a-z0-9A-Z#@\|]*fluechtling[a-z0-9A-Z\.:\|]*|'
        #                 '[a-z0-9A-Z#@\|]*flüchtling[a-z0-9A-Z\.:\|]*|'
        #                 '[a-z0-9A-Z#@\|]*refugee[a-z0-9A-Z\.:\|]*|'
        #                 '[a-z0-9A-Z#@\|]*asyl[a-z0-9A-Z\.:\|]*)')

        re_remove_str = ('([^\s]*fluechtling[^\s]*|'
                         '[^\s]*flüchtling[^\s]*|'
                         '[^\s]*refugee[^\s]*|'
                         '[^\s]*asyl[^\s]*)')



        re_remove = re.compile(re_remove_str, re.IGNORECASE)
        re_ex_space = re.compile('\s\s+')

        def corpus_elements(query_results):
            for q in query_results:
                for tweet in q:
                    text = tweet.text
                    text = re_remove.sub('', text)
                    text = re_ex_space.sub(' ', text)
         
                    grams = n_grams(text, parser=nlp, n=1)
                    if grams is None:
                        grams = ['']
                    
                    yield uni_dict.doc2bow(grams)

        corpus = list(corpus_elements([keyword_tweets, random_tweets]))

        print('Generating dtm...')
        X = corpus2csc(corpus).transpose()
        y = np.array([1] * keyword_tweets.count() + [0] * random_tweets.count())
        print('Shape y: {}'.format(y.shape))
        print('SHape X: {}'.format(X.shape))
        pickle.dump({'X': X, 'uni_dict': uni_dict, 'y': y}, 
                    open('full_dtm.p', 'wb'))

    if not os.isfile('bclf.p'):
        print('Train clf...')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=3432)

        # Train classifier

        ## Define parameter grid for hyperpar optimization
        params = {'loss': ['hinge', 'log', 'squared_hinge', 'modified_huber'],
                  'penalty': ['elasticnet'], 'alpha': 10.0**-np.arange(1,7), 'l1_ratio': np.arange(0, 1.2, 0.2)}
        ## Fit
        clf = GridSearchCV(estimator=SGDClassifier(), param_grid=params,
                               n_jobs=12, cv=10)
        clf.fit(X_train, y_train)

        # Assess best Classifier
        bclf = clf.best_estimator_
        print(bclf)
        pred = bclf.predict(X_test)
        print(classification_report(y_test, pred))
        pickle.dump(bclf, open('bclf.p', 'wb'))

        get_miw(dictionary=uni_dict, clf=bclf, n=20)

    # Classifiy the panel tweets 
    


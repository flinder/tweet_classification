#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import spacy
import pandas as pd
import numpy as np
import Stemmer
import pickle
import re
import datetime
import glob
import string

from gensim.corpora import Dictionary
from gensim import matutils

sys.path.append('../../dissdat/database/')
from db import make_session

class TextProcessor(object):
    '''
    Project specific text processor class

    Arguments:
    Language: string, Language of the text that is being processed
    '''

    def __init__(self, language):
        self.parser = spacy.load(language)
        # Workaround to make stopwords work in German spacy 
        # see (https://github.com/explosion/spaCy/issues/729)
        for word in self.parser.Defaults.stop_words:
            lex = self.parser.vocab[word]
            lex.is_stop = True
        self.stem = Stemmer.Stemmer(language).stemWord 
        self.rm_chars = re.compile('[@/\\\\]')
        self.re_hashtag = re.compile(r'#[A-Za-zÄÖÜäöü0-9_]+')
        self.re_user = re.compile(r'@[A-Za-zÄÖÜäöü0-9_]+')

    def tokenize(self, text):
        '''
        Tokenize original string, keep hashtags and handles intact and tag them
        as such entities

        Arguments:
        text: string, text to be parsed
        stem: boolean, should tokens be stemmed?
        '''
        doc = self.parser(text)

        for el in re.finditer(self.re_hashtag, text):
            doc.merge(start_idx=el.start(0), end_idx=el.end(0))

        return doc
    
    def document_stream(self, documents, stem=False, word_type=None):
        '''
        Generator that yields stream of tokenized documents

        Arguments:
        ---------
        documents: Iterable of strings
        stem: should tokens be stemmed?
        word_type: Filter by this word type ('hashtag' or 'user')
        '''
        for d in documents:
            tokens = self.tokenize(d)
            if word_type == 'hashtag':
                tokens = [t for t in tokens 
                          if self.re_hashtag.match(t.text) is not None]
            elif word_type == 'user':
                tokens = [t for t in tokens 
                          if self.re_user.match(t.text) is not None]

            if stem:
                tokens = [self.stem(t.orth_).lower() for t in tokens]
            else:
                tokens = [t.orth_.lower() for t in tokens]
            yield tokens

    def make_dtm(self, documents, normalize=True, word_type=None, sparse=False):
        '''
        Tokenize and stem documents and store in sparse matrix. Note this is in
        memory, so don't use on large text collections

        Arguments:
        ----------
        documents: An iterator over documents, each document is a string
        normalize: Should tokens be stemmed (lemmatization not implemented) 
        word_type: None for all words 'hashtag' or 'user'
        '''
    
        doc_stream = self.document_stream(documents, normalize, word_type)
        vocab = Dictionary(doc_stream)

        if normalize:
            vocab.filter_extremes(no_below=5, no_above=0.2)

        doc_stream = self.document_stream(documents, normalize, word_type)
    
        corpus = [vocab.doc2bow(tokens) for tokens in doc_stream]
        vocab_terms = [x[1] for x in vocab.items()]

        if not sparse:
            dtm = matutils.corpus2dense(corpus, num_terms=len(vocab),
                                        num_docs=len(corpus)).transpose()
            dtm = pd.DataFrame(dtm,
                               columns=vocab_terms)
            return dtm
        else:
            dtm = matutils.corpus2csc(corpus, num_terms=len(vocab),
                                      num_docs=len(corpus)).transpose()
            return dtm, vocab_terms
            



def get_data():
    '''
    Get the dataset from the sql database and aggregate annotations to binary
    '''
    _, engine = make_session()

    query = ('SELECT cr_results.tweet_id,' 
             '       cr_results.annotation,' 
             '       cr_results.trust,'
             '       tweets.text,'
             '       tweets.created_at,'
             '       users.screen_name '
             'FROM cr_results '
             'INNER JOIN tweets ON tweets.id=cr_results.tweet_id '
             'INNER JOIN users ON users.id=tweets.user_id ')

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

    # Convert tweet timestamp to utc (is in utc but has tz aware format)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df = df[['annotation', 'text', 'tweet_id', 
             'screen_name', 'created_at']].groupby('tweet_id').apply(f)
    
    # Select only what we need
    df = df[['annotation', 'text', 'screen_name', 'created_at']]
    
    # Make annotations binary
    df.loc[df['annotation'] >= 0.5, 'annotation'] = 1
    df.loc[df['annotation'] < 0.5, 'annotation'] = 0

    # For some reason a few tweets outside of the 2015 snuck in, remove them
    df = df[df.created_at.dt.year == 2015]
 
    df.reset_index(inplace=True)

    return df

def clean(word, tokenize):
    '''
    Process survey responses to produces separate keywords. tokenize, lowercase, 
    stem.
    '''
    tokens = tokenize(word)
    pattern = '[' + re.escape(string.punctuation) + '\s+' + ']'
    out = []
    for i,w in enumerate(tokens):
        if re.match(pattern, w.orth_) is not None or w.is_stop:
            continue
        out.append(w.orth_.lower())
    return out


if __name__ == "__main__":
    '''
    This script extracts the tweets and user information form the sql database,
    generates the following data objects:
    
    - Document-term-matrix w/o preprocessing (query matrix):
      dtm_non_normalized.p
    - Document-term-matrix stemmed (feature matrix): dtm_normalized.p
    - Ground truth for hashtag frequency distribution (gt_user.p)
    - Ground truth for user frequency distribution (gt_hashtags.p)
    - Ground truth for time frequency distribution (gt_timeline.p)

    And stores them (pickled) in the data directory.
    '''
   
    parser = TextProcessor('de')
    df = get_data()
    dtm_normalized, norm_terms = parser.make_dtm(df.text, normalize=True, 
                                                 sparse=True)
    dtm_non_normalized, non_norm_terms = parser.make_dtm(df.text, 
                                                         normalize=False, 
                                                         sparse=True)
    dtm_hashtags = parser.make_dtm(df.text, normalize=False, word_type='hashtag')
    dtm_users = parser.make_dtm(df.text, normalize=False, word_type='user')


    # ground truth distributions for users and hastags
    gt_hashtags = dtm_hashtags[df['annotation'] == 1].sum(axis=0)
    gt_hashtags = gt_hashtags.values.reshape(1, -1)
    gt_users = dtm_users[df['annotation'] == 1].sum(axis=0)
    gt_users = gt_users.values.reshape(1, -1)

    times = pd.DataFrame(df['created_at'][df['annotation'] == 1])
    per_day = (times.groupby(times.created_at.dt.dayofyear).count())
    per_day['count'] = per_day['created_at']
    per_day['date'] = per_day.index.astype(int)
    per_day['date'] = pd.DatetimeIndex([datetime.datetime(2015, 1, 1) 
                                        + datetime.timedelta(days=int(x-1)) for
                                        x in per_day['date']])
    every_day = pd.DataFrame({'date': pd.date_range(start='1-1-2015', 
                                                    end='12-31-2015')})
    tl = (every_day.merge(per_day[['count', 'date']], how='left')
                       .fillna(0))
    gt_timeline = np.array(tl['count']).reshape(1, -1)

    # Crowdflower survey
    reports = glob.glob('../data/survey_data/cf_report*')
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
    
    clean_words = []
    for w in words:
        clean_words.extend(clean(w, parser.tokenize))

    survey_keywords = {}
    for w in clean_words:
        survey_keywords[w] = survey_keywords.get(w, 0) + 1
    
    kwords = pd.DataFrame([[k,survey_keywords[k]] for k in survey_keywords],
                          columns=['word', 'count'])
    kwords['weight'] = kwords['count'] / kwords['count'].sum()


    # Serialize everyting
    pickle.dump(kwords, open('../data/dtms/kwords.p', 'wb'))
    pickle.dump(dtm_normalized, open('../data/dtms/dtm_normalized.p', 'wb'))
    pickle.dump(norm_terms, open('../data/dtms/norm_terms.p', 'wb'))
    pickle.dump(dtm_non_normalized, 
                open('../data/dtms/dtm_non_normalized.p', 'wb'))
    pickle.dump(non_norm_terms, open('../data/dtms/non_norm_terms.p', 'wb'))
    pickle.dump(dtm_hashtags, open('../data/dtms/dtm_hashtags.p', 'wb'))
    pickle.dump(dtm_users, open('../data/dtms/dtm_users.p', 'wb'))
    pickle.dump(df, open('../data/dtms/df.p', 'wb'))
    ground_truth = {'timeline': gt_timeline, 'hashtags': gt_hashtags, 
                    'users': gt_users}
    pickle.dump(ground_truth, open('../data/dtms/ground_truth.p', 'wb'))

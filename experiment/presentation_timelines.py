import pickle
import datetime
import os

import numpy as np
import pandas as pd

from ggplot import *

def timeline(selection, df, daily_counts):
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
    tl['proportion'] = tl['count'] / daily_counts['count']
    return tl[['proportion', 'date']]

def get_index(x, element):
    try:
        return x.index(element)
    except ValueError:
        return None
    
def query_dtm(query, dtm, terms):
    '''
    Query the doc-term matrix for documents containing at leas on term of query
    '''
    query_idx = [get_index(terms, w) for w in query if get_index(terms, w) is not None]
    return np.squeeze(np.asarray(dtm[:, query_idx].sum(axis=1) != 0))

if __name__ == "__main__":
    
    # Settings
    N_ITERATIONS = 500
    N_WORDS = 5
    SEED = 55771
    DATA_DIR = '../data/'
    
    # Load preprocessed data
    load = lambda x: pickle.load(open(os.path.join(DATA_DIR, x), 'rb'))
    dtm_full = load('dtms/dtm_non_normalized_full.p')
    terms = load('dtms/full_terms.p')
    df_full = load('dtms/df_full.p')
    kwords = load('dtms/kwords.p')
   
    # Calculate daily counts for full dataset
    daily_counts = df_full.groupby(df_full.created_at.dt.dayofyear).count()
    daily_counts['count'] = daily_counts['created_at']
    daily_counts['date'] = daily_counts.index.astype(int)
    daily_counts['date'] = pd.DatetimeIndex([datetime.datetime(2015, 1, 1) 
                                             + datetime.timedelta(days=int(x-1)) for
                                             x in daily_counts['date']])
    daily_counts.index = range(0,365)

    # Select the keywords words to sample from for each timeline
    kwords.sort_values('weight', inplace=True, ascending=False)
    words = kwords.iloc[:100]
    
    # Generate the timelines
    proportion = []
    date = []
    iteration = []
    keywords = []
    np.random.seed(SEED)
    for i in range(0, N_ITERATIONS):
        query = list(words.sample(N_WORDS, weights=words['weight']).word)
        keywords.append(query)
        selection = df_full[query_dtm(query, dtm_full, terms)].index
        tl = timeline(selection, df_full, daily_counts)
        proportion.extend(list(tl.proportion))
        date.extend(list(tl.date))
        
    
    # Create df for plotting
    iteration = [i for i in range(N_ITERATIONS) for _ in range(365)]
    pdat = pd.DataFrame({'proportion': proportion, 'iteration': iteration, 
                     'date': date})
    pdat.to_csv(os.path.join(DATA_DIR, 'results/timelines.csv', index=False)    

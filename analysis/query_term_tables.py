'''
This script produces the tables of query terms produced by the different query
expansion methods
'''

import pickle
import pandas as pd
import numpy as np

def make_df(method, queries):
    df = pd.DataFrame(
        {'word': [w for w in queries['expansion'].keys()],
         'count': [c for c in queries['expansion'].values()]}
    )
    df['proportion'] = round(df['count'] / df['count'].sum(), 3)
    del df['count']
    df['translation'] = np.nan
    return df.sort_values('proportion', ascending=False)

def make_tables(method):
    fname = f'../experiment/queries_{method}_automatic.p'
    queries = pickle.load(open(fname, 'rb'))
    df = make_df('expansion', queries)
    
    for version in ['long', 'short']:
        tab_fname = f'../paper/tables/queries_{method}_{version}.tex'
        if version == 'short':
            out = df.head(20)
        else:
            out = df
        out.to_latex(buf=tab_fname, index=False, longtable=True)
        
for m in ['monroe', 'king', 'lasso']:
    make_tables(m)

# Terms from survey
df = pd.read_csv('../data/survey_data/crowdflower_keywords.csv')
del df['count']
df['proportion'] = round(df['weight'], 3)
del df['weight']
df['translation'] = np.nan

for version in ['long', 'short']:
    tab_fname = f'../paper/tables/queries_baseline_{version}.tex'
    if version == 'short':
        out = df.head(20)
        lt = False
    else:
        out = df
        lt = True
    out.to_latex(buf=tab_fname, index=False, longtable=lt)

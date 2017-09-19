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
    df['proportion'] = df['count'] / df['count'].sum()
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
        out.to_latex(buf=tab_fname, index=False)
        
for m in ['monroe', 'king', 'lasso']:
    make_tables(m)

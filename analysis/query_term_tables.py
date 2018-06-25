'''
This script produces the tables of query terms produced by the different query
expansion methods
'''

import pickle
import pandas as pd
import numpy as np
import string

def remove_non_ascii(s, valid_chars):
    return ''.join(filter(lambda x: x in valid_chars, s))

def make_df(method, queries):
    df = pd.DataFrame(
        {'word': [w for w in queries[method].keys()],
         'count': [c for c in queries[method].values()]}
    )
    df['word'] = [remove_non_ascii(x, printable) for x in df['word']]
    df['proportion'] = round(df['count'] / df['count'].sum(), 3)
    del df['count']
    return df.sort_values('proportion', ascending=False)

def make_tables(method, queries):
    df = make_df(method, queries)
    
    for version in ['long', 'short']:
        tab_fname = f'../paper/tables/queries_{method}_{version}.tex'
        if version == 'short':
            out = df.head(20)
        else:
            out = df
            ## Customize the table
            tab_code = df.to_latex(longtable=True, index=False)
            # Add caption and label
            if method == 'expansion':
                m = 'lasso expansion method.'
            elif method == 'baseline':
                m = 'baseline survey keywords.'
            else: 
                m = 'klr expansion method.'
            caption = ('Proportion of times a term was included in the final '
                       'query (iteration 100) of the experiment using the '
                       f'{m}')
            label = f'tab:query_terms_{method}_long'
            out = tab_code.replace('\n',
                    f'\n\\caption{{{caption}}}\\\\\n\\label{{{label}}}\\\\\n', 1)
            out = out.replace('\\multicolumn{3}', '\\multicolumn{2}', 1)
            out = out.replace('\\endhead', '\\endfirsthead\n\\toprule\n word & proportion \\\\\\midrule\n\\endhead', 1)
            with open(tab_fname, 'w') as tf:
                tf.write(out)
            #df.to_latex(buf=tab_fname, index=False, longtable=True)
        
fname = f'../experiment/v3_aci_queries_lasso_automatic.p'
queries = pickle.load(open(fname, 'rb'))
printable = set(string.printable)
       
for m in ['klr', 'expansion', 'baseline']:
    make_tables(m, queries)

## Terms from survey
#df = pd.read_csv('../data/survey_data/crowdflower_keywords.csv')
#del df['count']
#df['proportion'] = round(df['weight'], 3)
#del df['weight']
#df['translation'] = np.nan
#
#for version in ['long', 'short']:
#    tab_fname = f'../paper/tables/queries_baseline_{version}.tex'
#    if version == 'short':
#        out = df.head(20)
#        lt = False
#    else:
#        out = df
#        lt = True
#    out.to_latex(buf=tab_fname, index=False, longtable=lt)

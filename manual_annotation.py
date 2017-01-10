# Program that presents tweets. Takes annotation and measures time

import json
import io
import time
import sys
import numpy as np
sys.path.append('../../dissdat/database/')
from db import make_session, Tweet
from sqlalchemy import and_
from  sqlalchemy.sql.expression import func


def save_times():
    global total_start
    total = time.time() - total_start
    print(total)
    print(np.array(times).mean())
    with open('times.txt', 'a+') as outfile:
        for t in times:
            outfile.write(str(t) + '\n')

# Start up sql engine
session, engine = make_session('../../dissdat/database/db_credentials')

cursor = session.query(Tweet).filter(and_(Tweet.data_group == 'de_panel', 
                                          Tweet.annotation_me == None)).order_by(func.random()).yield_per(1)
times = []

total_start = time.time()
try:
    for t in cursor:
        print(t.text)
        s = time.time()
        resp = input('Relevant?(y/n/m)')
        if resp == 'y':
            t.annotation_me = 'relevant'
        elif resp == 'n':
            t.annotation_me = 'irrelevant'
        elif resp == 'm':
            t.annotation_me = 'unclear'
        else:
            continue
        times.append(time.time() - s)
        
except KeyboardInterrupt as e:
    save_times()
    session.commit()
    session.close()
    raise e

save_times()

session.commit()
session.close()

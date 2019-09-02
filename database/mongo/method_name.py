from pymongo import MongoClient
from data.loader import Loader

client = MongoClient()

for d, e in Loader.yield_episodes('cylinder-1'):
    method = e['actions'][0]['method']

    print(e['id'], method)

    if method == 'RANDOM':
        method = 'Random'
    elif method == 'TOP_5':
        method = 'Top5'
    elif method == 'BOTTOM_5':
        method = 'Bottom5'
    elif method == 'MAX':
        method = 'Max'
    elif method == 'UNCERTAIN':
        method = 'Uncertain'
    elif method == 'PROB':
        method = 'Prob'
    elif method == 'BAYES':
        method = 'Bayes'
    elif method == 'BAYES_TOP':
        method = 'BayesTop'
    elif method == 'BAYES_PROB':
        method = 'BayesProb'
    elif method == 'NOT_ZERO':
        method = 'NotZero'
    elif method == 'RANDOM_INFERENCE':
        method = 'RandomInference'

    if method != e['actions'][0]['method']:
        client[d].episodes.update_one(
            {'id': e['id']},
            {'$set': {'actions.0.method': method}}
        )

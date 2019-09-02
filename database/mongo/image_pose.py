
from pymongo import MongoClient
from data.loader import Loader

client = MongoClient()

for d, e in Loader.yield_episodes('cylinder-cube-1'):
    data = e['actions'][0]['images']
    print(e['id'])

    new_data = {}
    for s in data.keys():
        if '.' in s:
            new_data[s.replace('.', '_')] = data[s]
        else:
            new_data[s] = data[s]

    client[d].episodes.update_one(
        {'id': e['id']},
        {'$set': {'actions.0.images': new_data}}
    )

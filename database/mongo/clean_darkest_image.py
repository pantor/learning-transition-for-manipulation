import argparse

import numpy as np

from data.loader import Loader


def get_mean(episode):
    _, image = Loader.get_action(episode[0], episode[1]['id'], 'ed-after')

    if image is None:
        return {'id': episode[1]['id'], 'mean': 1e6}

    return {'id': episode[1]['id'], 'mean': np.mean(image.mat)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean the robot learning database.')
    parser.add_argument('database', type=str, help='database name')
    parser.add_argument('--N', type=int, default=20, help='number results')
    args = parser.parse_args()

    dataset = map(get_mean, Loader.yield_episodes(args.database))
    sorted_dataset = sorted(dataset, key=lambda k: k['mean'])

    for i, result in enumerate(sorted_dataset[:args.N]):
        print(f'{i + 1}: {result}')

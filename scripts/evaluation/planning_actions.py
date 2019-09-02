import numpy as np


class Evaluation:
    def __init__(self, steps):
        self.steps = np.array(steps)

    @classmethod
    def sem(cls, data):
        return data.std() / np.sqrt(data.size)

    def __repr__(self):
        print(f'{self.steps.mean():.2f} ({self.sem(self.steps):.2f})')
        return ''


if __name__ == '__main__':
    # No Planning: Depth 6, Leaves 1
    unplanned = Evaluation([4, 4, 5, 5, 4, 5, 5, 4, 5, 4, 6, 5, 6])
    print('Unplanned: {unplanned}')

    # Planning: Depth 6, Leaves 3
    planned = Evaluation([4, 4, 4, 5, 4, 4, 4])
    print('Unplanned: {planned}')

    print(f'Success rate: {1 - 7 / 32:0.3f}')
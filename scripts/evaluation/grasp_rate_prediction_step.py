from typing import Tuple

import numpy as np
from pathlib import Path
import pandas as pd


class Evaluation:
    def __init__(self, directory: Path, filename: str):
        self.directory = directory
        self.filename = filename

        self.data = pd.read_csv(self.directory / self.filename)

    def __repr__(self):
        print(f'Total {self.data.step.count()}')
        print()
        print(f'Step \t| Grasp Rate \t| Uncertainty \t| Estimated Reward \t| Number')

        steps = []
        grasp_rates = []

        for step in range(self.data.step.max() + 1):
            series = self.data[self.data.step == step]

            steps.append(step)
            grasp_rates.append(series.reward.mean())

            print(f'{step} \t| {series.reward.mean():0.3f}({1.2*series.reward.sem():0.3f}) \t| {series.estimated_reward_std.mean():0.3f}({1.2*series.estimated_reward_std.sem():0.3f}) \t| {series.estimated_reward.mean():0.3f}({1.2*series.estimated_reward.sem():0.3f}) \t| {series.reward.count()}')
        print()

        z = np.polyfit(steps[:8], grasp_rates[:8], 2)
        print(f'Quadratic regression: {z}')

        print()
        print(f'Estimated Reward \t| Grasp Rate \t| Number')
        for r_top in [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89]:
            a = self.data[(r_top - 0.01 < self.data.estimated_reward) & (self.data.estimated_reward < r_top)]
            print(f'{r_top - 0.01:0.2f} - {r_top:0.2f} \t\t| {a.reward.mean():0.3f}({a.reward.sem():0.3f}) \t| {a.reward.count()}')

        return ''


if __name__ == '__main__':
    evaluation = Evaluation(
        Path.home() / 'Documents' / 'data' / 'cylinder-cube-1' / 'evaluation',
        'grasp-rate-prediction-step.txt',
    )
    print(evaluation)

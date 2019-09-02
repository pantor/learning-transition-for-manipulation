from typing import Tuple

from pathlib import Path
import pandas as pd


class Evaluation:
    def __init__(self, directory: Path, filename: str):
        self.directory = directory
        self.filename = filename

        self.out, self.out_of = self.split_filename(self.filename)
        self.data = pd.read_csv(self.directory / self.filename)

    @classmethod
    def split_filename(cls, name: str) -> Tuple[int, int]:
        header = name.split('.')[0].split('-')
        return int(header[0]), int(header[3])

    def __repr__(self):
        grasp_rate = self.out / self.data.grasps.mean()
        grasp_rate_std = grasp_rate / self.data.grasps.mean() * self.data.grasps.sem()

        print(f'File {self.directory}{self.filename}')
        print(f'Evaluation ({self.out} out of {self.out_of})')
        print(f'Count episodes:\t{self.data.grasps.size}')
        print(f'Count tries:\t{self.data.grasps.sum()}')

        print(f'Grasps:\t\t{self.data.grasps.mean():.2f} ({self.data.grasps.sem():.2f})')
        print(f'Shifts:\t\t{self.data.shifts.mean() / self.out:.2f} ({ self.data.shifts.sem() / self.out:.2f})')

        time_mean = self.data.time.mean() - 1.0  # [s] for waiting
        pph_mean = self.out * 3600 / time_mean
        print(f'Time:\t\t{time_mean:.2f} ({self.data.time.sem():.2f}) [s]')
        print(f'PPH:\t\t{pph_mean:.1f} ({pph_mean / time_mean * self.data.time.sem():.1f})')

        print(f'Grasp rate:\t{grasp_rate:.3f} ({grasp_rate_std:.3}) [%]')
        return ''


if __name__ == '__main__':
    for f in [
            '20-out-of-30.txt',
        ]:
        evaluation = Evaluation(Path.home() / 'Documents' / 'data' / 'cylinder-cube-mc-1' / 'evaluation', f)
        print(evaluation)

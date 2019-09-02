import numpy as np


class Evaluation:
    def __init__(self, object_counts, failed_grasps=None):
        self.object_counts = np.array(object_counts)
        self.failed_grasps = np.array(failed_grasps) if failed_grasps else failed_grasps

        self.time_length = 120  # [s]

    @classmethod
    def sem(cls, data):
        return data.std() / np.sqrt(data.size)

    def __repr__(self):
        print(f'Mean: {self.object_counts.mean():.2f} ({self.sem(self.object_counts):.2f})')

        pph = self.object_counts.astype(np.float32) * 3600 / self.time_length
        print(f'Mean: {pph.mean():.2f} ({self.sem(pph):.2f})')

        if self.failed_grasps is not None:
            grasp_rate = self.object_counts / (self.object_counts + self.failed_grasps)
            print(f'Grasp rate: {grasp_rate.mean():.3f} ({self.sem(grasp_rate):.3f})')
        return ''


if __name__ == '__main__':
    random = Evaluation(
        object_counts=[4, 1, 3, 3, 0],
        failed_grasps=[14, 15, 12, 15, 16],
    )
    print(random)

    single = Evaluation(
        object_counts=[12, 14, 12, 13, 13],
        failed_grasps=[0, 0, 1, 0, 1],
    )
    print(single)

    single_prediction = Evaluation(
        object_counts=[16, 15, 16, 18, 17],
        failed_grasps=[1, 2, 1, 0, 1],
    )
    print(single_prediction)

    multiple = Evaluation(
        object_counts=[17, 19, 22, 23, 21],
        failed_grasps=[1, 1, 0, 0, 2],
    )
    print(multiple)

    multiple_prediction = Evaluation(
        object_counts=[23, 25, 24, 23, 22],
        failed_grasps=[2, 1, 0, 1, 3],
    )
    print(multiple_prediction)
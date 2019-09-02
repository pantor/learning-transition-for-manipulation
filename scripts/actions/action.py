from typing import Any

from cfrankr import Affine
from picking.param import Bin, SelectionMethod


class RobotPose(Affine):
    def __init__(self, affine: Affine = None, data: Any = None, d: float = None):
        if affine:
            super().__init__(affine.x, affine.y, affine.z, affine.a, affine.b, affine.c)
            self.d = d if d else 0.0
        elif data:
            super().__init__(**{k: v if v != 'nan' else 0.0 for k, v in data.items() if k != 'd'})
            self.d = data['d'] if 'd' in data else (d if d else 0.0)
        else:
            super().__init__()
            self.d = d if d else 0.0

    def __repr__(self):
        values_str = [f'{i:0.3f}' for i in [self.x, self.y, self.z, self.a, self.b, self.c]]
        return '[' + ', '.join(values_str) + f'], d: {self.d:0.3f}'


class Action(object):
    # For MyPy annotations:
    # images: Dict[str, Any]
    # index: int
    # reward: float
    # estimated_reward: float
    # estimated_reward_std: float
    # collision: bool
    # save: bool
    # bin: Any
    # execution_time: float

    def __init__(self, action_type='', safe=0, data: Any = None):
        self.pose = RobotPose()
        self.type = action_type
        self.safe = safe

        if data:
            self.__dict__.update(data)
            if 'method' in data:
                self.method = SelectionMethod[data['method']]
            if 'bin' in data:
                self.bin = Bin[data['bin']]
            if 'pose' in data:
                self.pose = RobotPose(data=data['pose'])
            if 'final_pose' in data:
                self.final_pose = RobotPose(data=data['final_pose'])
            if 'place_pose' in data:
                self.place_pose = RobotPose(data=data['place_pose'])
            if 'images' in data:
                for suffix in data['images']:
                    self.images[suffix]['pose'] = RobotPose(data=data['images'][suffix]['pose'])

    def __repr__(self):
        result = self.type + ' ' if hasattr(self, 'type') else ''
        result += f'{self.pose}'
        if hasattr(self, 'estimated_reward'):
            result += f' estimated_reward: {self.estimated_reward:0.3f}'
            if hasattr(self, 'estimated_reward_std') and self.estimated_reward_std != 0.0:
                result += f' pm: {self.estimated_reward_std:0.3f}'
        if hasattr(self, 'method'):
            result += f' method: {self.method.name}'
        return result

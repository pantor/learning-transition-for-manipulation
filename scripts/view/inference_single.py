import os

import cv2
import numpy as np

from config import Config
from data.loader import Loader
from inference.planar import InferencePlanarPose
from picking.image import draw_pose


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    np.set_printoptions(suppress=True)

    action, image = Loader.get_action('cylinder-cube-mc-1', '2019-07-02-19-55-54-845', 'ed-v')

    inference = InferencePlanarPose(
        model=Loader.get_model('cylinder-cube-mc-1', 'model-1-mc'),
        box=Config.box,
        monte_carlo=160,
    )
    estimated_reward, estimated_reward_std = inference.infer_at_pose([image], action.pose)
    print(estimated_reward, estimated_reward_std)

    draw_pose(image, action.pose, convert_to_rgb=True)

    cv2.imshow('image', image.mat)
    cv2.waitKey(1500)

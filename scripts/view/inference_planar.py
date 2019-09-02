import time
import cv2
import numpy as np

from actions.converter import Converter
from actions.indexer import GraspIndexer
from config import Config
from data.loader import Loader
from inference.planar import InferencePlanarPose
from picking.image import draw_pose
from picking.param import SelectionMethod


if __name__ == '__main__':
    # inference = InferencePlanarPose(
    #     Loader.get_model('cylinder-cube-mc-1', 'model-1-mc', output_layer='prob'),
    #     box=Config.box,
    #     monte_carlo=160
    # )
    inference = InferencePlanarPose(
        Loader.get_model('cylinder-cube-1', 'model-6-arch-more-layer', output_layer='prob'),
        box=Config.box,
    )
    # inference = InferencePlanarPose(
    #   Loader.get_model('shifting', 'model-3'),
    #   box=Config.box,
    # )

    _, image = Loader.get_action('cylinder-cube-mc-1', '2019-07-02-13-27-22-246', 'ed-v')

    indexer = GraspIndexer(gripper_classes=Config.gripper_classes)

    converter = Converter(grasp_z_offset=Config.grasp_z_offset, box=Config.box)

    times = []
    for i in range(1):
        start = time.time()
        action = inference.infer([image], SelectionMethod.Top5, verbose=1)
        indexer.to_action(action)

        end = time.time()
        times.append(end - start)

    converter.calculate_pose(action, [image])
    print(action)
    print(f'Mean inference time [ms]: {(np.array(times[1:]).mean() * 1000):0.5f}')

    draw_pose(image, action.pose, convert_to_rgb=True)
    cv2.imshow('image', image.mat)
    cv2.waitKey(1500)

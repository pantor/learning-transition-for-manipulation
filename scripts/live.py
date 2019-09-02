#!/usr/bin/python3.6

from time import sleep

from loguru import logger

from cfrankr import Gripper, Robot  # pylint: disable=E0611
from config import Config
from picking.camera import Camera
from picking.frames import Frames
from picking.image import draw_around_box
from picking.param import Bin

import picking.path_fix_ros  # pylint: disable=W0611
import cv2  # pylint: disable=C0411


if __name__ == '__main__':
    camera = Camera(camera_suffixes=['rd'])
    robot = Robot('panda_arm', Config.general_dynamics_rel)
    gripper = Gripper('172.16.0.2', Config.gripper_speed)

    while True:
        camera_pose = robot.current_pose(Frames.camera)
        image = camera.take_images()[0]
        image.pose = camera_pose.inverse() * Frames.get_frame(Bin.Left)

        logger.info(f'current_pose: {camera_pose}')

        draw_around_box(image, box=Config.box)

        cv2.imshow('live', image.mat)
        cv2.waitKey(10)
        sleep(0.04)

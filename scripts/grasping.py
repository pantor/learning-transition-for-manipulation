#!/usr/bin/python3.6

from multiprocessing import Process
from subprocess import Popen
import sys
import random
import time

from loguru import logger

import picking.path_fix_catkin  # pylint: disable=W0611
from picking.camera import Camera

from agents.agent import Agent
from agents.agent_place import Agent as AgentPlace
from agents.agent_predict import Agent as AgentPredict
from agents.agent_shift import Agent as AgentShift
from actions.action import Action, RobotPose
from cfrankr import Affine, Gripper, MotionData, Robot, Waypoint  # pylint: disable=E0611
from config import Config
from data.loader import Loader
from learning.utils.layers import one_hot_gen  # pylint: disable:unused-import
from picking.episode import Episode, EpisodeHistory
from picking.frames import Frames
from picking.saver import Saver
from picking.param import Bin, Mode





def move_to_release(robot: Robot, current_bin: Bin, md: MotionData) -> bool:
    possible_random_affine = Affine()
    if Config.random_pose_before_release:
        possible_random_affine = Config.max_random_affine_before_release.get_inner_random()

    robot.recover_from_errors()

    if Config.mode is Mode.Measure:
        move_to_safety(robot, md)

    if Config.release_in_other_bin:
        if Config.release_as_fast_as_possible:
            waypoints = [
                Waypoint(
                    Frames.release_fastest,
                    Waypoint.ReferenceType.ABSOLUTE
                )
            ]
        else:
            waypoints = [
                Waypoint(
                    Frames.release_midway,
                    Waypoint.ReferenceType.ABSOLUTE
                ),
                Waypoint(
                    Frames.get_release_frame(Frames.get_next_bin(current_bin)) * possible_random_affine,
                    Waypoint.ReferenceType.ABSOLUTE
                )
            ]
        return robot.move_waypoints_cartesian(Frames.gripper, waypoints, MotionData())

    return robot.move_cartesian(
        Frames.gripper,
        Frames.get_release_frame(current_bin) * possible_random_affine,
        MotionData()
    )


def move_to_safety(robot: Robot, md: MotionData) -> bool:
    move_up = max(0.0, 0.16 - robot.current_pose(Frames.gripper).z)
    return robot.move_relative_cartesian(Frames.gripper, Affine(z=move_up), md)


def grasp(
        robot: Robot,
        gripper: Gripper,
        current_episode: Episode,
        current_bin: Bin,
        action: Action,
        action_frame: Affine,
        image_frame: Affine,
        camera: Camera,
        saver: Saver,
        md: MotionData
    ) -> None:
    # md_approach_down = MotionData().with_dynamics(0.12).with_z_force_condition(7.0)
    # md_approach_up = MotionData().with_dynamics(0.6).with_z_force_condition(20.0)

    md_approach_down = MotionData().with_dynamics(0.3).with_z_force_condition(7.0)
    md_approach_up = MotionData().with_dynamics(1.0).with_z_force_condition(20.0)

    action_approch_affine = Affine(z=Config.approach_distance_from_pose)
    action_approach_frame = action_frame * action_approch_affine

    try:
        process_gripper = Process(target=gripper.move, args=(action.pose.d, ))
        process_gripper.start()

        robot.move_cartesian(Frames.gripper, action_approach_frame, md)

        process_gripper.join()
    except OSError:
        gripper.move(0.08)
        robot.move_cartesian(Frames.gripper, action_approach_frame, md)

    robot.move_relative_cartesian(Frames.gripper, action_approch_affine.inverse(), md_approach_down)

    if md_approach_down.did_break:
        robot.recover_from_errors()
        action.collision = True
        robot.move_relative_cartesian(Frames.gripper, Affine(z=0.001), md_approach_up)

    action.final_pose = RobotPose(affine=(image_frame.inverse() * robot.current_pose(Frames.gripper)))

    first_grasp_successful = gripper.clamp()
    if first_grasp_successful:
        logger.info('Grasp successful at first.')
        robot.recover_from_errors()

        # Change here
        action_approch_affine = Affine(z=1.5*Config.approach_distance_from_pose)

        move_up_success = robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)
        if move_up_success and not md_approach_up.did_break:
            if Config.mode is Mode.Measure and Config.take_after_images and not Config.release_in_other_bin:
                robot.move_cartesian(Frames.camera, image_frame, md)
                saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)

            if Config.file_objects:
                raise Exception('File objects not implemented!')

            if Config.release_during_grasp_action:
                move_to_release_success = move_to_release(robot, current_bin, md)
                if move_to_release_success:
                    if gripper.is_grasping():
                        action.reward = 1.0
                        action.final_pose.d = gripper.width()

                    if Config.mode is Mode.Perform:
                        gripper.release(action.final_pose.d + 0.002)  # [m]
                    else:
                        gripper.release(action.pose.d + 0.002)  # [m]
                        move_to_safety(robot, md_approach_up)

                    if Config.mode is Mode.Measure and Config.take_after_images and Config.release_in_other_bin:
                        robot.move_cartesian(Frames.camera, image_frame, md)
                        saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)
            else:
                if Config.mode is not Mode.Perform:
                    move_to_safety(robot, md_approach_up)

                if Config.mode is Mode.Measure and Config.take_after_images and Config.release_in_other_bin:
                    robot.move_cartesian(Frames.camera, image_frame, md)
                    saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)

        else:
            gripper.release(action.pose.d + 0.002)  # [m]

            robot.recover_from_errors()
            robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)
            move_to_safety_success = move_to_safety(robot, md_approach_up)
            if not move_to_safety_success:
                robot.recover_from_errors()
                robot.recover_from_errors()
                move_to_safety(robot, md_approach_up)

            gripper.move(gripper.max_width)

            move_to_safety(robot, md_approach_up)
            if Config.mode is Mode.Measure and Config.take_after_images:
                robot.move_cartesian(Frames.camera, image_frame, md)
                saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)

    else:
        logger.info('Grasp not successful.')
        gripper.release(gripper.width() + 0.002)  # [m]

        robot.recover_from_errors()
        move_up_successful = robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)

        if md_approach_up.did_break or not move_up_successful:
            gripper.release(action.pose.d)  # [m]

            robot.recover_from_errors()
            robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)
            move_to_safety(robot, md_approach_up)

        if Config.mode is Mode.Measure and Config.take_after_images:
            robot.move_cartesian(Frames.camera, image_frame, md)
            saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)


def shift(
        robot: Robot,
        gripper: Gripper,
        current_episode: Episode,
        current_bin: Bin,
        action: Action,
        action_frame: Affine,
        image_frame: Affine,
        camera: Camera,
        saver: Saver,
        md: MotionData
    ) -> None:
    md_approach_down = MotionData().with_dynamics(0.15).with_z_force_condition(6.0)
    md_approach_up = MotionData().with_dynamics(0.6).with_z_force_condition(20.0)
    md_shift = MotionData().with_dynamics(0.1).with_xy_force_condition(10.0)

    action_approch_affine = Affine(z=Config.approach_distance_from_pose)
    action_approach_frame = action_approch_affine * action_frame

    try:
        process_gripper = Process(target=gripper.move, args=(action.pose.d, ))
        process_gripper.start()

        robot.move_cartesian(Frames.gripper, action_approach_frame, md)

        process_gripper.join()
    except OSError:
        gripper.move(0.08)
        robot.move_cartesian(Frames.gripper, action_approach_frame, md)

    robot.move_relative_cartesian(Frames.gripper, action_approch_affine.inverse(), md_approach_down)

    if md_approach_down.did_break:
        robot.recover_from_errors()
        action.collision = True
        robot.move_relative_cartesian(Frames.gripper, Affine(z=0.001), md_approach_up)

    robot.move_relative_cartesian(Frames.gripper, Affine(x=action.shift_motion[0], y=action.shift_motion[1]), md_shift)
    robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)

    # Reward is set outside of this function, due to depedency on agent


def place(
        robot: Robot,
        gripper: Gripper,
        current_episode: Episode,
        current_bin: Bin,
        action: Action,
        action_frame: Affine,
        grasp_action: Action,
        image_frame: Affine,
        camera: Camera,
        saver: Saver,
        md: MotionData
    ) -> None:

    move_to_safety(robot, md)

    md_approach_down = MotionData().with_dynamics(0.3).with_z_force_condition(7.0)
    md_approach_up = MotionData().with_dynamics(1.0).with_z_force_condition(20.0)

    action_approch_affine = Affine(z=Config.approach_distance_from_pose)
    action_approach_frame = action_frame * action_approch_affine

    robot.move_cartesian(Frames.gripper, action_approach_frame, md)
    robot.move_relative_cartesian(Frames.gripper, action_approch_affine.inverse(), md_approach_down)

    if md_approach_down.did_break:
        robot.recover_from_errors()
        action.collision = True
        robot.move_relative_cartesian(Frames.gripper, Affine(z=0.001), md_approach_up)

    action.final_pose = RobotPose(affine=(image_frame.inverse() * robot.current_pose(Frames.gripper)))

    gripper.release(grasp_action.final_pose.d + 0.006)  # [m]

    if Config.mode is not Mode.Perform:
        move_to_safety(robot, md_approach_up)

    if Config.mode is Mode.Measure and Config.take_after_images and Config.release_in_other_bin:
        robot.move_cartesian(Frames.camera, image_frame, md)
        saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)


def take_images(current_bin: Bin, camera: Camera, robot: Robot, image_frame: Affine = None):
    images = camera.take_images()
    if not image_frame:
        image_frame = robot.current_pose(Frames.camera)
    pose = RobotPose(affine=(image_frame.inverse() * Frames.get_frame(current_bin)))

    for image in images:
        image.pose = pose

    return images


def bin_picking():
    # agent = Agent()
    # agent.inference.current_type = 2

    # agent = AgentShift()

    agent = AgentPredict(prediction_model=Loader.get_model('cube-1', 'predict-bi-gen-5'))
    agent.grasp_inference.current_type = 2

    camera = Camera(camera_suffixes=Config.camera_suffixes)
    episode_history = EpisodeHistory()
    gripper = Gripper('172.16.0.2', Config.gripper_speed)
    robot = Robot('panda_arm', Config.general_dynamics_rel)
    saver = Saver(Config.database_url, Config.grasp_database)

    current_bin = Config.start_bin

    md = MotionData().with_dynamics(1.0)

    gripper.stop()

    robot.recover_from_errors()
    move_to_safety(robot, md)
    move_joints_successful = robot.move_joints(Frames.bin_joint_values[current_bin], md)

    if not move_joints_successful:
        gripper.move(0.07)

        robot.recover_from_errors()
        move_to_safety(robot, md)
        move_joints_successful = robot.move_joints(Frames.bin_joint_values[current_bin], md)

    if Config.mode is Mode.Measure and not Config.home_gripper:
        logger.warning('Want to measure without homing gripper?')
    elif Config.mode is Mode.Measure and Config.home_gripper:
        gripper.homing()

    move_to_safety(robot, md)
    gripper.homing()

    overall_start = time.time()

    for epoch in Config.epochs:
        while episode_history.total() < epoch.number_episodes:
            current_episode = Episode()
            current_selection_method = epoch.get_selection_method()
            if Config.mode in [Mode.Evaluate, Mode.Perform]:
                current_selection_method = epoch.get_selection_method_perform(episode_history.failed_grasps_since_last_success_in_bin(current_bin))

            start = time.time()

            if (not Config.predict_images) or agent.reinfer_next_time:
                robot.recover_from_errors()
                robot.move_joints(Frames.bin_joint_values[current_bin], md)

                b, c = random.choice(Config.overview_image_angles) if Config.lateral_overview_image else 0, 0
                camera_frame_overview = Frames.get_camera_frame(current_bin, b=b, c=c)
                if not Frames.is_camera_frame_safe(camera_frame_overview):
                    continue

                if Config.mode is Mode.Measure or Config.lateral_overview_image:
                    robot.move_cartesian(Frames.camera, camera_frame_overview, md)

                image_frame = robot.current_pose(Frames.camera)
                images = take_images(current_bin, camera, robot, image_frame=image_frame)

                if not Frames.is_gripper_frame_safe(robot.current_pose(Frames.gripper)):
                    logger.info('Image frame not safe!')
                    robot.recover_from_errors()
                    continue

            input_images = list(filter(lambda i: i.camera in Config.model_input_suffixes, images))

            # if episode_history.data:
            #     agent.successful_grasp_before = episode_history.data[-1].actions[0].reward > 0

            action = agent.infer(input_images, current_selection_method)
            action.images = {}
            action.save = True

            if Config.mode is Mode.Measure:
                saver.save_image(images, current_episode.id, 'v', action=action)

            if Config.mode is not Mode.Perform:
                saver.save_action_plan(action, current_episode.id)

            logger.info(f'Action: {action} at time {time.time() - overall_start:0.1f}')

            action.reward = 0.0
            action.collision = False
            action.bin = current_bin

            if Config.set_zero_reward:
                action.safe = -1

            if action.type == 'bin_empty':
                action.save = False

            elif action.type == 'new_image':
                action.save = False
                agent.reinfer_next_time = True

            if action.safe == 0:
                logger.warning('Action ignored.')
                action.save = False

            else:
                if Config.mode is Mode.Measure and Config.take_lateral_images and action.save:
                    md_lateral = MotionData().with_dynamics(1.0)

                    for b, c in Config.lateral_images_angles:
                        lateral_frame = Frames.get_camera_frame(current_bin, a=action.pose.a, b=b, c=c, reference_pose=image_frame)

                        if not Frames.is_camera_frame_safe(lateral_frame) or (b == 0.0 and c == 0.0):
                            continue

                        lateral_move_succss = robot.move_cartesian(Frames.camera, lateral_frame, md_lateral)  # Remove a for global b, c pose
                        if lateral_move_succss:
                            saver.save_image(take_images(current_bin, camera, robot), current_episode.id, f'lateral_b{b:0.3f}_c{c:0.3f}'.replace('.', '_'), action=action)

                if action.safe == 1 and action.type not in ['bin_empty', 'new_image']:
                    action_frame = Frames.get_action_pose(action_pose=action.pose, image_pose=image_frame)

                    if Config.mode is Mode.Measure and Config.take_direct_images:
                        robot.move_cartesian(Frames.camera, Affine(z=0.308) * Affine(b=0.0, c=0.0) * action_frame, md)
                        saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'direct', action=action)

                    if action.type == 'grasp':
                        grasp(robot, gripper, current_episode, current_bin, action, action_frame, image_frame, camera, saver, md)

                    elif action.type == 'shift':
                        old_reward_around_action = 0.0
                        shift(robot, gripper, current_episode, current_bin, action, action_frame, image_frame, camera, saver, md)
                        new_reward_around_action = 0.0

                        action.reward = new_reward_around_action - old_reward_around_action

                    elif action.type == 'place':
                        last_grasp = episode_history.data[-1].actions[0]
                        action.grasp_episode_id = episode_history.data[-1].id
                        place(robot, gripper, current_episode, current_bin, action, action_frame, last_grasp, image_frame, camera, saver, md)

                elif action.safe < 0:
                    logger.info('Pose not found.')
                    action.collision = True

                    if Config.take_after_images:
                        robot.move_cartesian(Frames.camera, image_frame, md)
                        saver.save_image(take_images(current_bin, camera, robot), current_episode.id, 'after', action=action)

            action.execution_time = time.time() - start
            logger.info(f'Time for action: {action.execution_time:0.3f} [s]')

            if action.save:
                current_episode.actions.append(action)

                if Config.mode is Mode.Measure:
                    logger.info(f'Save episode {current_episode.id}.')
                    saver.save_episode(current_episode)

            episode_history.append(current_episode)

            logger.info(f'Episodes (reward / done / total): {episode_history.total_reward(action_type="grasp")} / {episode_history.total()} / {sum(e.number_episodes for e in Config.epochs)}')
            logger.info(f'Last success: {episode_history.failed_grasps_since_last_success_in_bin(current_bin)} cycles ago.')

            # episode_history.save_grasp_rate_prediction_step_evaluation(Config.evaluation_path)

            # Change bin
            should_change_bin_for_evaluation = (Config.mode is Mode.Evaluate and episode_history.successful_grasps_in_bin(current_bin) == Config.change_bin_at_number_of_success_grasps)
            should_change_bin = (Config.mode is not Mode.Evaluate and (episode_history.failed_grasps_since_last_success_in_bin(current_bin) >= Config.change_bin_at_number_of_failed_grasps or action.type == 'bin_empty'))
            if should_change_bin_for_evaluation or (Config.change_bins and should_change_bin):
                if Config.mode is Mode.Evaluate:
                    pass

                current_bin = Frames.get_next_bin(current_bin)
                agent.reinfer_next_time = True
                logger.info('Switch to other bin.')

                if Config.mode is not Mode.Perform and Config.home_gripper:
                    gripper.homing()

            # Retrain model
            if Config.train_model and episode_history.total() > 0 and not episode_history.total() % Config.train_model_every_number_cycles:
                logger.warning('Retrain model!')
                with open('/tmp/training.txt', 'wb') as out:
                    p = Popen([sys.executable, str(Config.train_script)], stdout=out)
                    if not Config.train_async:
                        p.communicate()

    logger.info('Finished cleanly.')


if __name__ == '__main__':
    bin_picking()

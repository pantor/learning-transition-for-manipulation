# Learning a Transition Model for Robotic Manipulation

In this repository, we've published the code for our [publication](https://pantor.github.io/learning-transition-model-for-manipulation/) *Learning a Generative Transition Model for Uncertainty-Aware Robotic Manipulation* submitted to [ICRA 2020](http://www.icra2020.org/). As only parts of the code were specifically written for this publication, we introduce the code regarding the overall project idea.

<p align="center">
 <a href="https://drive.google.com/file/d/1h5FS_Q2BolOuQupU4NfN2wpcbErKNix9/view?usp=sharing">
  <img src="docs/assets/block-test.gif?raw=true" alt="Conference Video" />
 </a>
</p>


## General Idea

Let a robot learn how to grasp and manipulate objects by itself. The industrial task of bin picking emphasizes some difficulties like dense clutter, partial observability and unknwon geometry of the target objects. Since teaching the robot grasping in a model-based way is infeasible, we let the robot try to learn grasping via try and error. It needs around 20000 grasp tries using active learning to grasp reliably. This repo includes a range of extensions to the default state-of-the-art planar grasps: Specific grasps, lateral grasps, reactive grasps, model predictions and shifting of objects.

## Structure

The overall structure is as follows:
 - *Database Server* This is a database server for collecting and uploading data and images. The server has a web interface for showing all episodes in a dataset and displaying the latest action live.
 - *Include / Src* The low-level control of the hardware, in particular for the robot and the cameras, written in C++. The robot uses MoveIt! for control. The camera drivers for Ensenso and RealSense are included, either via direct access or an optional ros node. The latter is helpful because the Ensenso needs a long time to connect and crashes sometimes afterwards.
 - *Scripts* The main part of the project is written in Python. This includes the general program logic, calculating the next action with the Tensorflow Python interface, data management, learning, .... It is recommended to export to PYTHONPATH in `.bashrc`: `export PYTHONPATH=$PYTHONPATH:$HOME/Documents/bin_picking/scripts`
 - *Test* Few test cases. We are doing research, who needs tests?

This project is a ROS package with launch files and a package.xml. The ROS node /move_group is set to respawn=true. This enables to call rosnode kill /move_group to restart it.


## Installation

For the robotic hardware, make sure to load `launch/gripper-config.json` as the Franka end-effector configuration. Currently, following dependencies need to be installed:
- ROS Kinetic
- libfranka & franka_ros
- EnsensoSDK

And all requirements for Python 3.6 via Pip and `python3.6 -m pip install -r requirements.txt`. Patching CvBridge for Python3 and CMake >= 3.12 is given by a snippet in GitLab.


## Start

For an easy start, run `sh terminal-setup.sh` for a complete terminal setup. Start the mongodb daemon. Then run `roslaunch bin_picking moveit.launch`, `rosrun bin_picking grasping.py` and check the database server.


## Robot Learning Database

The robot learning database is a database, server and viewer for research around robotic grasping. It is based on MongoDB, Flask, Vue.js. It shows an overview of all episodes as well as live actions. It can also delete recorded episodes. The server can be started via `python3.6 database/app.py`, afterwards open [localhost](127.0.0.1:8080) in your browser.

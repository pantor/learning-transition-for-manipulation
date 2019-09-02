#!/usr/bin/python2

from __future__ import division
from __future__ import print_function

import cv2
from cv_bridge import CvBridge  # pylint: disable=E0611
import rospy  # pylint: disable=E0401
from sensor_msgs.msg import Image  # pylint: disable=E0401
from bin_picking.srv import InferGrasp, InferGraspRequest, InferGraspResponse, EstimateRewardForGrasp  # pylint: disable=E0401, E0611

# from actions.action import Action
from actions.converter import Converter
from actions.indexer import GraspIndexer
from data.loader import Loader
from inference.planar import InferencePlanarPose as PlanarInference
from orthographical import OrthographicImage
from picking.image import clone, draw_pose, draw_around_box
from view.heatmap import Heatmap


class Agent:
    def __init__(self):
        # Parameters
        self.grasp_model = rospy.get_param('graspro/grasp_model', 'graspro-v1')

        self.gripper_classes = rospy.get_param('graspro/gripper_classes')
        self.z_offset = rospy.get_param('graspro/z_offset', 0.0)

        self.ensenso_depth = rospy.get_param('graspro/camera/ensenso_depth')
        self.realsense_depth = rospy.get_param('graspro/camera/realsense_depth')
        self.realsense_color = rospy.get_param('graspro/camera/realsense_color')

        self.lower_random_pose = rospy.get_param('graspro/lower_random_pose', [-0.1, -0.1, 0.0])
        self.upper_random_pose = rospy.get_param('graspro/upper_random_pose', [0.1, 0.1, 0.0])

        self.box_center = rospy.get_param('graspro/bin_center', [0, 0, 0])
        self.box_size = rospy.get_param('graspro/bin_size', False)
        self.box = {'center': self.box_center, 'size': self.box_size}

        self.publish_heatmap = rospy.get_param('graspro/publish_heatmap', False)

        # Inference
        self.inference = PlanarInference(
            model=Loader.get_model(self.grasp_model, output_layer='prob'),
            box=self.box,
            lower_random_pose=self.lower_random_pose,
            upper_random_pose=self.upper_random_pose,
        )
        self.indexer = GraspIndexer(gripper_classes=self.gripper_classes)
        self.converter = Converter(grasp_z_offset=self.z_offset, box=self.box)  # [m]

        if self.publish_heatmap:
            self.heatmapper = Heatmap(self.inference, self.inference.model, box=self.box)
            self.heatmap_publisher = rospy.Publisher('graspro/heatmap')

        self.bridge = CvBridge()
        self.image_publisher = rospy.Publisher('graspro/pose_on_image', Image, queue_size=10)

        s1 = rospy.Service('graspro/infer_grasp', InferGrasp, self.infer_grasp)
        s2 = rospy.Service('graspro/estimate_reward_for_grasp', EstimateRewardForGrasp, self.estimate_reward_for_grasp)
        rospy.spin()

    def infer_grasp(self, req: InferGraspRequest) -> InferGraspResponse:
        image_res = self.get_images()
        cv_image = self.bridge.imgmsg_to_cv2(image_res.image, 'mono16')
        image = OrthographicImage(cv_image, image_res.pixel_size, image_res.min_depth, image_res.max_depth)

        images = [image]

        action = self.inference.infer(images, req.method)

        draw_image = clone(image)
        draw_image.mat = cv2.cvtColor(draw_image.mat, cv2.COLOR_GRAY2RGB)
        if self.draw_on_image:
            draw_around_box(draw_image, box=self.box, draw_lines=True)
            draw_pose(draw_image, action)

        if self.save_tmp_image:
            cv2.imwrite('/tmp/graspro/current-image.png', draw_image.mat)

        draw_image_msg = self.bridge.cv2_to_imgmsg(draw_image.mat, 'rgb8')
        self.image_publisher.publish(draw_image_msg)

        if self.publish_heatmap:
            heatmap = self.heatmapper.render(image, alpha=0.5)
            heatmap_msg = self.bridge.cv2_to_imgmsg(heatmap, 'rgb8')
            self.heatmap_publisher.publish(heatmap_msg)

        return InferGraspResponse(action=action)

    def estimate_reward_for_grasp(self, req):
        pass


if __name__ == '__main__':
    rospy.init_node('graspro')

    try:
        grasp = Agent()
    except rospy.ROSInterruptException:
        pass

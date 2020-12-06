#!/usr/bin/env python
import rospy
import tf
import math
import numpy as np

from tf.transformations import quaternion_matrix

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from actionlib_msgs.msg import GoalStatusArray
from actionlib_msgs.msg import GoalStatus


class StationKeepingControl(object):
    def __init__(self):
        self._tf = tf.TransformListener()

        self._target_distance_error = rospy.get_param(
            'target_distance_error_m', 1)
        self._target_heading_error = math.pi / 180.0 * \
            rospy.get_param('target_heading_error', 1)
        self._target_long_axis = rospy.get_param('target_long_axis', 12)
        self._target_short_axis = rospy.get_param('target_short_axis', 3)

        self._base_link_frame = rospy.get_param(
            "base_link_frame", "cora/base_link")

        self._cmd_vel_topic = rospy.get_param("cmd_vel_topic", "/cora/cmd_vel")
        self._localization_topic = rospy.get_param(
            "localization_topic", "/cora/robot_localization/odometry/filtered")
        self._goal_pose_topic = rospy.get_param(
            "goal_pose_topic", "/move_base_simple/goal")
        self._move_base_status_topic = rospy.get_param(
            "move_base_status_topic", "/move_base/status")

        self._use_algoritm_1 = False

        # Algorithm 1 parameters
        self._distance_correction_speed = rospy.get_param(
            "distance_correction_speed", 1.0)
        self._heading_correction_speed = rospy.get_param(
            "heading_correction_speed", 0.2)

        # Algorithm 2 parameters
        self._alg2_heading_control_threshold = 4.0
        self._alg2_angular_correction_speed = 0.2
        self._alg2_linear_correction_gain = 0.02

        # publishers
        self._cmd_vel_pub = rospy.Publisher(
            self._cmd_vel_topic, Twist, queue_size=10)
        # subscribers
        self._goal_pose_sub = rospy.Subscriber(
            self._goal_pose_topic, PoseStamped, self._goal_pose_callback)
        self._move_base_status_sub = rospy.Subscriber(
            self._move_base_status_topic, GoalStatusArray, self._move_base_status_callback)
        # control loop timer
        #self._control_timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback_2)
        self._control_timer = rospy.Timer(
            rospy.Duration(0.1), self._timer_callback)

        # don't let anything uninitialized
        self._target_pose_frame = None
        self._target_pose_timestamp = None
        self._target_center_pose = None
        self._framing_box_corners = []
        self._target_index = None
        self._station_keeping_is_enabled = False

    def _goal_pose_callback(self, goal_pose_msg):
        rot_matrix = quaternion_matrix([
            goal_pose_msg.pose.orientation.x,
            goal_pose_msg.pose.orientation.y,
            goal_pose_msg.pose.orientation.z,
            goal_pose_msg.pose.orientation.w
        ])

        # numpy numbers columns first
        unit_vector_x = rot_matrix[:3, 0]
        unit_vector_y = rot_matrix[:3, 1]

        target_center_position = np.array([goal_pose_msg.pose.position.x,
                                           goal_pose_msg.pose.position.y,
                                           goal_pose_msg.pose.position.z])

        u = self._target_long_axis / 2.0
        v = self._target_short_axis / 2.0
        framing_box_corners = [
            target_center_position + unit_vector_x * u + unit_vector_y * v,
            target_center_position + unit_vector_x * u - unit_vector_y * v,
            target_center_position - unit_vector_x * u + unit_vector_y * v,
            target_center_position - unit_vector_x * u - unit_vector_y * v
        ]

        self._target_pose_frame = goal_pose_msg.header.frame_id
        self._target_pose_timestamp = rospy.Time.now()

        def get_point_from_array(vector): return Point(
            x=vector[0], y=vector[1], z=vector[2])

        self._target_center_pose = get_point_from_array(target_center_position)
        self._framing_box_corners = [get_point_from_array(
            item) for item in framing_box_corners]

        rospy.loginfo("New goal pose received: " +
                      str(self._target_center_pose))
        self._reset_target()

    def _reset_target(self):
        self._target_index = None
        self._clamp_orientation = False
        self._angular_to_lineal_transition_timestamp = rospy.Time.now()

    def _move_base_status_callback(self, status_array):
        # Check if there's any goal active in the action server
        def is_active(status):
            return (status == GoalStatus.ACTIVE) or (status == GoalStatus.PREEMPTING) or (status == GoalStatus.RECALLING)
        a_goal_is_active = False
        for entry in status_array.status_list:
            a_goal_is_active = a_goal_is_active or is_active(entry.status)
        if a_goal_is_active:
            rospy.loginfo_throttle_identical(
                3.0, "A goal is active, station keeping is disabled")
        else:
            rospy.loginfo_throttle_identical(3, "Station keeping enabled%s" % (
                ", but no goal set" if not self._framing_box_corners else ""))
        self._station_keeping_is_enabled = not a_goal_is_active

    def _convert_position_to_local_frame(self, source_frame, location):
        original = PointStamped()
        original.header.frame_id = source_frame
        original.point = location
        pose = self._tf.transformPoint(self._base_link_frame, original)
        return Point(x=pose.point.x, y=pose.point.y, z=pose.point.z)

    def _timer_callback(self, _):
        if self._use_algoritm_1:
            self._station_keeping_algorithm_1()
        else:
            self._station_keeping_algorithm_2()

    def _station_keeping_algorithm_1(self):
        if not self._station_keeping_is_enabled or not self._framing_box_corners:
            return

        try:
            local_targets = [
                self._convert_position_to_local_frame(
                    self._target_pose_frame, item) for item in self._framing_box_corners]
        except Exception as e:
            rospy.logerr("Error converting from %s to %s: %s" %
                         (self._target_pose_frame, self._base_link_frame, e))
            return

        def distance(v): return math.sqrt(v.x**2 + v.y**2)
        distances = [distance(item) for item in local_targets]

        if self._target_index is None:
            self._target_index = 0
            for i in range(1, 4):
                if (distances[i] > distances[self._target_index]):
                    self._target_index = i

        target = local_targets[self._target_index]
        distance_error = distances[self._target_index]
        heading_error = math.atan(target.y / np.abs(target.x))

        go_backwards = target.x < 0.0

        lineal_speed = 0.0
        angular_speed = 0.0

        if not self._clamp_orientation:
            # Angular correction
            now = rospy.Time.now()
            if (now - self._angular_to_lineal_transition_timestamp > rospy.Duration(3)):
                angular_speed = np.sign(heading_error) * \
                    self._heading_correction_speed
                angular_speed *= (1 if not go_backwards else -1)
                if (np.abs(heading_error) < self._target_heading_error):
                    self._angular_to_lineal_transition_timestamp = rospy.Time.now()
                    self._clamp_orientation = True
        else:
            # Distance correction
            now = rospy.Time.now()
            if (now - self._angular_to_lineal_transition_timestamp > rospy.Duration(3)):
                lineal_speed = min(0.5 * distance_error,
                                   self._distance_correction_speed)
                lineal_speed *= (-1.0 if go_backwards else 1.0)

        travel_slack = target.x * (-1.0 if go_backwards else 1.0)
        if (travel_slack < self._target_distance_error):
            # we arrived
            lineal_speed = 0.0
            angular_speed = 0.0
            # choose a new target
            self._reset_target()

        self._command_controller(
            lineal_speed,
            angular_speed
        )

    def _station_keeping_algorithm_2(self):
        if not self._station_keeping_is_enabled or not self._target_center_pose:
            return

        try:
            local_target_point = self._convert_position_to_local_frame(
                self._target_pose_frame, self._target_center_pose)
        except Exception as e:
            rospy.logerr("Error converting from %s to %s: %s" %
                         (self._target_pose_frame, self._base_link_frame, e))
            return

        def distance(v):
            return math.sqrt(v.x**2 + v.y**2)

        distance_to_target = distance(local_target_point)

        distance_error = local_target_point.x
        heading_error = math.atan2(
            local_target_point.y, np.abs(local_target_point.x))

        lineal_speed = 0.0
        angular_speed = 0.0

        if math.fabs(distance_to_target) > self._alg2_heading_control_threshold:
            angular_speed = self._alg2_angular_correction_speed * \
                (1.0 if heading_error > 0 else -1.0)
        lineal_speed = self._alg2_linear_correction_gain * distance_error

        rospy.loginfo("Linear speed command: %f" % (lineal_speed))
        rospy.loginfo("Angular speed command: %f" % (angular_speed))

        self._command_controller(
            lineal_speed,
            angular_speed
        )

    def _command_controller(self, lineal, angular):
        msg = Twist()
        msg.linear.x = lineal
        msg.angular.z = angular
        self._cmd_vel_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tij_station_keeping_control', anonymous=True)
    StationKeepingControl().run()

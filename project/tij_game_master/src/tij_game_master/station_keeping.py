#!/usr/bin/env python
import rospy

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPoseStamped
from std_srvs.srv import SetBool

from utils import wgs84_to_utm_pose


class StationKeepingTaskManager(object):
    def __init__(self):
        # Initialize everything
        self._move_base_simple_goal_topic = "/move_base_simple/goal"
        self._station_keeping_goal_topic = "/vorc/station_keeping/goal"

        # publishers
        self._move_base_simple_goal_pub = rospy.Publisher(
            self._move_base_simple_goal_topic, PoseStamped, queue_size=10)

        # services
        self._enable_engine_controller = rospy.ServiceProxy(
            '/tij/enable_engine_controller', SetBool)

        # subscribers
        self._station_keeping_goal_sub = rospy.Subscriber(
            self._station_keeping_goal_topic, GeoPoseStamped, self._station_keeping_goal_callback)

    def transitioned_to_ready(self):
        self._enable_engine_controller(True)

    def transitioned_to_running(self):
        pass

    def transitioned_to_finished(self):
        pass

    def _station_keeping_goal_callback(self, goal_msg):
        rospy.loginfo("WGS84 goal: {}".format(goal_msg))
        utm_goal = wgs84_to_utm_pose(goal_msg)
        rospy.loginfo("UTM goal: {}".format(utm_goal))
        self._move_base_simple_goal_pub.publish(utm_goal)

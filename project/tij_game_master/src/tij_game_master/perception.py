#!/usr/bin/env python
import rospy

from copy import deepcopy

import math as m

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPoseStamped
from std_srvs.srv import SetBool

from utils import wgs84_to_utm_pose


class PerceptionTaskManager(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Initialize everything
        self._base_link_frame = "cora/base_link"
        self._detected_objects_topic = "/vorc/perception/landmark"
        self._perception_topic = "/vorc/perception/landmark"
        
        self._object_published = set()
        self._running_stage_on = False
        
        # publishers
        self._perception_pub = rospy.Publisher(
            self._perception_topic, GeoPoseStamped, queue_size=10)

        # services
        self._enable_engine_controller = rospy.ServiceProxy(
            '/tij/enable_engine_controller', SetBool)

        # subscribers
 #       self._station_keeping_goal_sub = rospy.Subscriber(
#            self._detected_objects_topic, GeoPath, self._detected_objects_callback)

        # Timers
        rospy.Timer(rospy.Duration(1), self._timer_callback)

    def transitioned_to_ready(self):
        self._enable_engine_controller(True)

    def transitioned_to_running(self):
        pass

    def transitioned_to_finished(self):
        pass

    def _detected_objects_callback(self, detections_msg):
        if not self._running_stage_on:
            return
        # for buoy_data in detections_msg.buoys:
        #     if buoy.id not in self._object_published:
        #         self._publish_detection(buoy)
        #         set.add(buoy.id)

    def _publish_detection(self, buoy_data):
        rospy.logerr("MISSING CODE!")
#!/usr/bin/env python
import rospy

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPoseStamped
from std_srvs.srv import SetBool
from sensor_msgs.msg import NavSatFix
from tij_object_recognition.msg import DetectionData
from tij_object_recognition.msg import DetectionDataArray

from utils import wgs84_to_utm_pose
from utils import utm_to_wgs84_pose


class PerceptionTaskManager(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Initialize everything
        self._utm_frame = "utm"
        self._detected_objects_topic = "/tij/detections/detection_array"
        self._perception_topic = "/vorc/perception/landmark"
        self._gps_location_topic = "/cora/robot_localization/gps/filtered"

        self._objects_published = set()
        self._running_stage_on = False
        
        self._probability_threshold = 0.5
        self._current_nav_sat_fix = None

        # publishers
        self._perception_pub = rospy.Publisher(
            self._perception_topic, GeoPoseStamped, queue_size=10)

        # services
        self._enable_engine_controller = rospy.ServiceProxy(
            '/tij/enable_engine_controller', SetBool)

        # subscribers
        self._station_keeping_goal_sub = rospy.Subscriber(self._detected_objects_topic, DetectionDataArray, self._detected_objects_callback)
        self._gps_pose_sub = rospy.Subscriber(self._gps_location_topic, NavSatFix, self._gps_location_callback)

    def _gps_location_callback(self, nav_sat_fix_msg):
        self._current_nav_sat_fix = nav_sat_fix_msg

    def transitioned_to_ready(self):
        self._enable_engine_controller(True)

    def transitioned_to_running(self):
        self._running_stage_on = True
        pass

    def transitioned_to_finished(self):
        pass

    def _detected_objects_callback(self, detection_data_array_msg):
        if not self._running_stage_on:
            return
        
        if self._current_nav_sat_fix is None:
            return

        stamp = detection_data_array_msg.header.stamp
        frame = detection_data_array_msg.header.frame_id

        for detection_data in detection_data_array_msg.detections:
            if detection_data.id not in self._objects_published:
                if detection_data.classified and detection_data.probability > self._probability_threshold:
                    self._publish_detection(detection_data, frame, stamp)
                    self._objects_published.add(detection_data.id)
                else:
                    rospy.logwarn("Removing detection from publication because of lack of certainty")

    def _publish_detection(self, detection_data, detection_frame, detection_stamp):
        source_detection_point = PointStamped()
        source_detection_point.header.frame_id = detection_frame
        source_detection_point.header.stamp = detection_stamp
        source_detection_point.point = detection_data.location
        
        transform = self._tf_buffer.lookup_transform(target_frame=self._utm_frame,
                                                     source_frame=detection_frame,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(
                                                         1.0)
                                                     )
        utm_detection_point = tf2_geometry_msgs.do_transform_point(source_detection_point, transform)

        utm_pose = PoseStamped()
        utm_pose.pose.orientation.x = 0.0
        utm_pose.pose.orientation.y = 0.0
        utm_pose.pose.orientation.z = 0.0
        utm_pose.pose.orientation.w = 1.0
        utm_pose.pose.position.x = utm_detection_point.point.x
        utm_pose.pose.position.y = utm_detection_point.point.y
        utm_pose.pose.position.z = utm_detection_point.point.z
        geo_pose = utm_to_wgs84_pose(utm_pose, self._current_nav_sat_fix)
        geo_pose.header.frame_id = detection_data.category
        self._perception_pub.publish(geo_pose)
        rospy.logwarn(geo_pose)

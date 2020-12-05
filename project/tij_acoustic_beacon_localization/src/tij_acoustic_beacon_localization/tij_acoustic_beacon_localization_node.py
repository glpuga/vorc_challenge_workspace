#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs

from tf.transformations import quaternion_matrix

import math as m

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from usv_msgs.msg import RangeBearing


class BlackBoxLocationNode(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._acoustic_sensor_topic = rospy.get_param("~acoustic_sensor_topic",
                                                      "/cora/sensors/pingers/pinger/range_bearing")
        self._black_box_pose_topic = rospy.get_param(
            "~black_box_pose_topic", "/tij/detections/black_box_pose")
        self._black_box_marker_topic = rospy.get_param(
            "~black_box_marker_topic", "/tij/markers/black_box")

        self._world_frame = rospy.get_param("~world_frame", "cora/odom")

        # publishers
        self._black_box_pose_pub = rospy.Publisher(
            self._black_box_pose_topic, PointStamped, queue_size=1)
        self._black_box_marker_pub = rospy.Publisher(
            self._black_box_marker_topic, Marker, queue_size=1)

        # subscribers
        self._acoustic_sensor_sub = rospy.Subscriber(
            self._acoustic_sensor_topic, RangeBearing, self._range_bearing_callback, queue_size=1)

    def _range_bearing_callback(self, bearing_msg):
        x = bearing_msg.range * \
            m.cos(bearing_msg.bearing) * m.cos(bearing_msg.elevation)
        y = bearing_msg.range * \
            m.sin(bearing_msg.bearing) * m.cos(bearing_msg.elevation)
        z = bearing_msg.range * m.sin(bearing_msg.elevation)

        source_frame_point = PointStamped()
        source_frame_point.header.stamp = rospy.Time.now()
        source_frame_point.header.frame_id = bearing_msg.header.frame_id
        source_frame_point.point.x = x
        source_frame_point.point.y = y
        source_frame_point.point.z = z

        transform = self._tf_buffer.lookup_transform(target_frame=self._world_frame,
                                                     source_frame=source_frame_point.header.frame_id,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(
                                                         1.0)
                                                     )

        target_frame_point = tf2_geometry_msgs.do_transform_point(
            source_frame_point, transform)

        rospy.loginfo("Located a beacon at %s" %
                      (" ".join(str(target_frame_point.point).split())))
        rospy.loginfo("  - Distance : %7.2f" % (m.sqrt(x**2 + y**2)))
        rospy.loginfo("  - Depth    : %7.2f" % (z))

        self._black_box_pose_pub.publish(target_frame_point)
        self._publish_marker(target_frame_point)

    def _publish_marker(self, point_in_world):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self._world_frame
        m.ns = "tij"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.MODIFY
        m.pose.position.x = point_in_world.point.x
        m.pose.position.y = point_in_world.point.y
        m.pose.position.z = 5
        m.pose.orientation.w = 1.0
        m.scale = Vector3(x=2, y=2, z=10)
        m.color = ColorRGBA(1.0, 0.0, 0, 1.0)
        m.lifetime = rospy.Duration(3.0)
        m.frame_locked = True
        self._black_box_marker_pub.publish(m)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tij_acoustic_beacon_localization_node', anonymous=True)
    BlackBoxLocationNode().run()

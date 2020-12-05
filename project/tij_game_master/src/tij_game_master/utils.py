#!/usr/bin/env python

import rospy

from tf.transformations import quaternion_matrix

from geographic_msgs.msg import GeoPoseStamped
from geometry_msgs.msg import PoseStamped

from actionlib import GoalStatusArray
from actionlib import GoalStatus
from geodesy.utm import fromMsg
from geodesy.utm import gridZone
from geodesy.utm import UTMPoint


def wgs84_to_utm_pose(wgs84_pose):
    pose = PoseStamped()
    pose.header.stamp = wgs84_pose.header.stamp
    pose.header.frame_id = "utm"
    pose.pose.position = fromMsg(wgs84_pose.pose.position).toPoint()
    pose.pose.orientation = wgs84_pose.pose.orientation
    return pose


def utm_to_wgs84_pose(utm_pose, vehicle_gps_fix):
    zone, band = gridZone(vehicle_gps_fix.latitude,
                          vehicle_gps_fix.longitude)
    utm_point = UTMPoint(
        utm_pose.pose.position.x,
        utm_pose.pose.position.y,
        utm_pose.pose.position.z,
        zone, band)

    geo_pose = GeoPoseStamped()
    geo_pose.header.stamp = utm_pose.header.stamp
    geo_pose.header.frame_id = "wgs84"
    geo_pose.pose.position = utm_point.toMsg()
    geo_pose.pose.orientation = utm_pose.pose.orientation
    return geo_pose


def get_xy_normals_from_quaternion(orientation_quaternion):
    rot_matrix = quaternion_matrix([
        orientation_quaternion.x,
        orientation_quaternion.y,
        orientation_quaternion.z,
        orientation_quaternion.w
    ])
    # numpy numbers columns first
    unit_vector_x = rot_matrix[:3, 0]
    unit_vector_y = rot_matrix[:3, 1]
    return unit_vector_x, unit_vector_y


class MoveBaseObserver(object):

    def __init__(self):
        # Initialize everything
        self._move_base_status_topic = "/move_base/status"
        self._a_goal_is_active = True
        # subscribers
        self._move_base_status_sub = rospy.Subscriber(
            self._move_base_status_topic, GoalStatusArray, self._move_base_status_callback)

    def move_base_is_active(self):
        return self._a_goal_is_active

    def _move_base_status_callback(self, status_array):
        # Check if there's any goal active in the action server
        def is_active(status):
            return (status == GoalStatus.ACTIVE) or (status == GoalStatus.PREEMPTING) or (status == GoalStatus.RECALLING)
        a_goal_is_active = False
        for entry in status_array.status_list:
            a_goal_is_active = a_goal_is_active or is_active(entry.status)
        self._a_goal_is_active = a_goal_is_active

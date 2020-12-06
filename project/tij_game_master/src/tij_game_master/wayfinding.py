#!/usr/bin/env python
import rospy

from copy import deepcopy

import math as m

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPoseStamped
from geographic_msgs.msg import GeoPath
from std_srvs.srv import SetBool

from utils import wgs84_to_utm_pose
from utils import get_xy_normals_from_quaternion
from utils import MoveBaseObserver


class WayfindingTaskManager(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Initialize everything
        self._base_link_frame = "cora/base_link"
        self._move_base_simple_goal_topic = "/move_base_simple/goal"
        self._wayfinding_path_topic = "/vorc/wayfinding/waypoints"
        self._approximation_pose_distance = 25.0
        self._utm_path = None
        self._enable_path_tracer = False
        self._currently_executing_plan = False
        self._move_base_observer = MoveBaseObserver()

        # These would improve behavior, if only the planner were better tuned for backwards motion
        self._use_entry_waypoints = False

        # publishers
        self._move_base_simple_goal_pub = rospy.Publisher(
            self._move_base_simple_goal_topic, PoseStamped, queue_size=10)

        # services
        self._enable_engine_controller = rospy.ServiceProxy(
            '/tij/enable_engine_controller', SetBool)

        # subscribers
        self._station_keeping_goal_sub = rospy.Subscriber(
            self._wayfinding_path_topic, GeoPath, self._wayfinding_path_callback)

        # Timers
        rospy.Timer(rospy.Duration(1), self._timer_callback)

    def transitioned_to_ready(self):
        self._enable_engine_controller(True)

    def transitioned_to_running(self):
        self._enable_path_tracer = True
        pass

    def transitioned_to_finished(self):
        pass

    def _wayfinding_path_callback(self, path_msg):
        self._utm_waypoints = []
        for geo_pose_stamped in path_msg.poses:
            rospy.loginfo("Waypoint WGS84 goal: {}".format(geo_pose_stamped))
            utm_goal = wgs84_to_utm_pose(geo_pose_stamped)
            rospy.loginfo("UTM goal: {}".format(utm_goal))
            self._utm_waypoints.append(utm_goal)

        self._plan_path()
        self._utm_path_index = 0

        # Target fist pose
        self._move_base_simple_goal_pub.publish(
            self._utm_path[self._utm_path_index])

    def _plan_path(self):
        utm_ordered_goals = self._sort_by_distance(self._utm_waypoints)
        if self._use_entry_waypoints:
            self._utm_path = self._decorate_path_with_entry_waypoints(
                utm_ordered_goals)
        else:
            self._utm_path = utm_ordered_goals

    def _timer_callback(self, event):
        if not self._enable_path_tracer:
            rospy.loginfo_throttle_identical(1, "Path execution is disabled")
            return

        if self._utm_path is None:
            rospy.logerr("Wayfinding is not possible, the path is empty!")
            return

        if self._currently_executing_plan and not self._move_base_observer.move_base_is_active():
            rospy.logwarn("Waypoint reached!")
            self._currently_executing_plan = False

        # Notice that this crapy-ass algorithm depend on the timer callback being slow enough
        # so that we can see an update status on the move base action server before the next
        # time the callback gets called, or we might trigger the goal more than once before
        # it actually gets executed
        if not self._currently_executing_plan:
            # Keep executing the plan in a loop
            self._utm_path_index += 1
            if self._utm_path_index >= len(self._utm_path):
                self._utm_path_index = 0
                self._plan_path()
            self._currently_executing_plan = True
            rospy.logwarn("Setting sails to new goal!: {}".format(
                self._utm_path[self._utm_path_index]))
            self._move_base_simple_goal_pub.publish(
                self._utm_path[self._utm_path_index])

    def _sort_by_distance(self, org_utm_path):
        current_pose = self._get_current_pose("utm")

        expendable_copy = deepcopy(org_utm_path)
        dst_utm_path = [current_pose]

        def calculate_xy_distance(pose1, pose2):
            x1 = pose1.pose.position.x
            y1 = pose1.pose.position.y
            x2 = pose2.pose.position.x
            y2 = pose2.pose.position.y
            return m.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        while len(expendable_copy) > 0:
            min_distance = 1e6
            min_index = 0
            for index in range(len(expendable_copy)):
                utm_pose = expendable_copy[index]
                distance_to_pose = calculate_xy_distance(
                    dst_utm_path[-1], utm_pose)
                if distance_to_pose < min_distance:
                    min_distance = distance_to_pose
                    min_index = index
            dst_utm_path.append(expendable_copy[min_index])
            del expendable_copy[min_index]
        # remove the first element, which is the current pose of the boat
        return dst_utm_path[1:]

    def _decorate_path_with_entry_waypoints(self, org_utm_path):
        dst_utm_path = []
        for utm_pose in org_utm_path:
            approximation_pose = self._calculate_approximation_pose(utm_pose)
            dst_utm_path.append(approximation_pose)
            dst_utm_path.append(utm_pose)
        return dst_utm_path

    def _calculate_approximation_pose(self, utm_pose):
        distance = self._approximation_pose_distance
        ux, _ = get_xy_normals_from_quaternion(utm_pose.pose.orientation)
        updated_utm_pose = deepcopy(utm_pose)
        updated_utm_pose.pose.position.x += -distance * ux[0]
        updated_utm_pose.pose.position.y += -distance * ux[1]
        return updated_utm_pose

    def _get_current_pose(self, to_frame):
        source_frame_point = PoseStamped()
        source_frame_point.header.stamp = rospy.Time.now()
        source_frame_point.header.frame_id = self._base_link_frame
        source_frame_point.pose.position.x = 0
        source_frame_point.pose.position.y = 0
        source_frame_point.pose.position.z = 0
        source_frame_point.pose.orientation.x = 0.0
        source_frame_point.pose.orientation.y = 0.0
        source_frame_point.pose.orientation.z = 0.0
        source_frame_point.pose.orientation.w = 1.0

        transform = self._tf_buffer.lookup_transform(target_frame=to_frame,
                                                     source_frame=self._base_link_frame,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(
                                                         1.0)
                                                     )

        return tf2_geometry_msgs.do_transform_pose(
            source_frame_point, transform)

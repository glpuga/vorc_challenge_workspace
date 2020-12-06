#!/usr/bin/env python
import rospy

import math as m
import numpy as np

from tf.transformations import quaternion_from_matrix

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPoseStamped
from std_srvs.srv import SetBool
from sensor_msgs.msg import NavSatFix
from tij_object_recognition.msg import DetectionData
from tij_object_recognition.msg import DetectionDataArray

from utils import MoveBaseObserver
from utils import wgs84_to_utm_pose
from utils import utm_to_wgs84_pose


class GymkhanaTaskManager(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Initialize everything
        self._map_frame = "cora/odom"
        self._base_link_frame = "cora/base_link"
        self._detected_objects_topic = "/tij/detections/detection_array"
        self._move_base_simple_goal_topic = "/move_base_simple/goal"
        self._black_box_pose_topic = "/tij/detections/black_box_pose"

        self._entrance_gate_pose = None
        self._passing_gate_poses = []
        self._visited_gate_poses = []
        self._unclassified_objects = []
        self._exit_gate_pose = None
        self._black_box_pose = None

        self._running_stage_on = False

        self._min_gate_width = 5.0
        self._max_gate_width = 25.0

        self._visited_gate_distance_threshold = 4.0
        self._exploration_forward_step = 10.0

        self._move_base_observer = MoveBaseObserver()

        self._execution_index = 0
        self._execution_phases = [
            self._find_channel_entrance,
            self._get_through_gate,
            self._find_next_passage_gate,
            self._get_through_gate,
            self._find_channel_exit,
            self._get_through_gate,
            self._find_black_box,
            self._station_keeping
        ]

        # publishers
        self._move_base_simple_goal_pub = rospy.Publisher(
            self._move_base_simple_goal_topic, PoseStamped, queue_size=10)

        # services
        self._enable_engine_controller = rospy.ServiceProxy(
            '/tij/enable_engine_controller', SetBool)

        # subscribers
        self._station_keeping_goal_sub = rospy.Subscriber(
            self._detected_objects_topic, DetectionDataArray, self._detected_objects_callback)
        self._black_box_pose_sub = rospy.Subscriber(
            self._black_box_pose_topic, PointStamped, self._black_box_pose_callback)

        # Timers
        rospy.Timer(rospy.Duration(1), self._timer_callback)

    def transitioned_to_ready(self):
        self._enable_engine_controller(True)

    def transitioned_to_running(self):
        self._running_stage_on = True
        pass

    def transitioned_to_finished(self):
        pass

    def _timer_callback(self, event):
        self._execution_phases[self._execution_index]()

    def _detected_objects_callback(self, detection_data_array_msg):
        gates_detected = []

        def convert_location_to_pose_stamped(location):
            p = PoseStamped()
            p.header.frame_id = self._map_frame
            p.header.stamp = rospy.Time.now()
            p.pose.position.x = location.x
            p.pose.position.y = location.y
            p.pose.position.z = location.z
            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = 0.0
            p.pose.orientation.w = 1.0
            return p

        self._unclassified_objects = []
        for ext_data in detection_data_array_msg.detections:
            if not ext_data.classified:
                self._unclassified_objects.append(
                    convert_location_to_pose_stamped(ext_data.location))
                continue
            for int_data in detection_data_array_msg.detections:
                if not int_data.classified:
                    continue
                if ext_data.id == int_data.id:
                    continue
                result = self._analyze_potential_gate(ext_data, int_data)
                if result is not None:
                    gate_type, gate_pose = result
                    gates_detected.append((gate_type, gate_pose))

        self._entrance_gate_pose = None
        self._passing_gate_poses = []
        self._exit_gate_pose = None

        for gate_type, gate_pose in gates_detected:
            if gate_type == "entrance_gate":
                self._entrance_gate_pose = gate_pose
            elif gate_type == "passage_gate":
                self._passing_gate_poses.append(gate_pose)
            elif gate_type == "exit_gate":
                self._exit_gate_pose = gate_pose

    def _analyze_potential_gate(self, b1, b2):
        def distance(b1, b2):
            return m.sqrt((b1.location.x - b2.location.x)**2 + (b1.location.y - b2.location.y)**2)

        def is_pair(a1, b2, pair_type):
            test_pair = set()
            test_pair.add(b1.category)
            test_pair.add(b2.category)
            ref_pair = set(pair_type)
            return test_pair == ref_pair

        def get_right_buoy(b1, b2, category):
            if b1.category == category:
                return b1
            else:
                return b2

        def get_gate_center(b1, b2):
            x = (b1.location.x + b2.location.x) / 2.0
            y = (b1.location.y + b2.location.y) / 2.0
            z = (b1.location.z + b2.location.z) / 2.0
            return x, y, z

        def calculate_gate_orientation(gate_center, right_buoy):
            vx = right_buoy.location.x - gate_center[0]
            vy = right_buoy.location.y - gate_center[1]
            return self._calculate_quaternion_for_x_vector((-vy, vx))

        gate_center = get_gate_center(b1, b2)

        gate_width = distance(b1, b2)
        if (gate_width < self._min_gate_width) or (gate_width > self._max_gate_width):
            return None

        if is_pair(b1, b2, ("surmark950410", "surmark46104")):
            gate_type = "entrance_gate"
            right_buoy = get_right_buoy(b1, b2, "surmark950410")
        elif is_pair(b1, b2, ("surmark950410", "surmark950400")):
            gate_type = "passage_gate"
            right_buoy = get_right_buoy(b1, b2, "surmark950410")
        elif is_pair(b1, b2, ("surmark950410", "blue_totem")):
            gate_type = "exit_gate"
            right_buoy = get_right_buoy(b1, b2, "surmark950410")
        else:
            return None

        gate_orientation = calculate_gate_orientation(gate_center,  right_buoy)

        gate_pose = PoseStamped()
        gate_pose.header.stamp = rospy.Time.now()
        gate_pose.header.frame_id = self._map_frame

        gate_pose.pose.position.x = gate_center[0]
        gate_pose.pose.position.y = gate_center[1]
        gate_pose.pose.position.z = gate_center[2]
        gate_pose.pose.orientation.x = gate_orientation[0]
        gate_pose.pose.orientation.y = gate_orientation[1]
        gate_pose.pose.orientation.z = gate_orientation[2]
        gate_pose.pose.orientation.w = gate_orientation[3]

        return gate_type, gate_pose

    def _get_through_gate(self):
        running_command = self._move_base_observer.move_base_is_active()
        if not running_command:
            rospy.loginfo("Got through gate")
            self._execution_index += 1

    def _find_channel_entrance(self):
        running_command = self._move_base_observer.move_base_is_active()
        if self._entrance_gate_pose is not None and self._running_stage_on:
            rospy.loginfo("Going to the entrance gate")
            self._move_base_simple_goal_pub.publish(self._entrance_gate_pose)
            self._execution_index += 1
        elif not running_command:
            rospy.loginfo("No entrance gate on sight, exploring")
            self._move_base_simple_goal_pub.publish(
                self._get_exploratory_pose())

    def _find_next_passage_gate(self):
        running_command = self._move_base_observer.move_base_is_active()
        next_gate = self._find_next_unvisited_passage_gate()
        if not running_command:
            if next_gate is not None:
                rospy.loginfo("Going to the next passage gate")
                self._move_base_simple_goal_pub.publish(next_gate)
            else:
                self._execution_index += 1

    def _find_channel_exit(self):
        running_command = self._move_base_observer.move_base_is_active()
        if self._exit_gate_pose is not None:
            rospy.loginfo("Going towards the exit gate")
            self._move_base_simple_goal_pub.publish(self._exit_gate_pose)
            self._execution_index += 1
        elif not running_command:
            rospy.loginfo("No exit gate on sight, exploring")
            self._move_base_simple_goal_pub.publish(
                self._get_exploratory_pose())

    def _find_black_box(self):
        running_command = self._move_base_observer.move_base_is_active()
        if self._black_box_pose is not None:
            rospy.loginfo("Going to the estimated black box pose")
            self._move_base_simple_goal_pub.publish(self._black_box_pose)
            self._execution_index += 1
        elif not running_command:
            rospy.loginfo("No black box pose available, exploring")
            self._move_base_simple_goal_pub.publish(
                self._get_exploratory_pose())

    def _station_keeping(self):
        # do nothing, it'll take care of itself
        rospy.loginfo_throttle_identical(
            2.0, "Station keeping, no more execution stages")

    def _find_next_unvisited_passage_gate(self):
        transform = self._tf_buffer.lookup_transform(target_frame=self._base_link_frame,
                                                     source_frame=self._map_frame,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(
                                                         1.0)
                                                     )

        passage_gates_in_local_frame = [tf2_geometry_msgs.do_transform_pose(
            pose, transform) for pose in self._passing_gate_poses]

        def distance_to_gate(gate_pose):
            x = gate_pose.pose.position.x
            y = gate_pose.pose.position.y
            return m.sqrt(x**2 + y**2)

        def distance_between_gates(p1, p2):
            x = p1.pose.position.x - p2.pose.position.x
            y = p1.pose.position.y - p2.pose.position.y
            return m.sqrt(x**2 + y**2)

        # find the closest gate that we haven't yet visited
        min_index = None
        min_distance = 10e3
        for index in range(len(passage_gates_in_local_frame)):
            local_gate_pose = passage_gates_in_local_frame[index]
            map_gate_pose = self._passing_gate_poses[index]

            # if we have already visited this gate, do not consider it
            distances_to_visited_gates = [
                distance_between_gates(map_gate_pose, visited_gate_pose) for visited_gate_pose in self._visited_gate_poses]
            has_been_visited = [
                d < self._visited_gate_distance_threshold for d in distances_to_visited_gates]
            if any(has_been_visited):
                continue

            distance = distance_to_gate(local_gate_pose)
            if min_index is None or distance < min_distance:
                min_distance = distance
                min_index = index

        if min_index is None:
            # No more passage gates to cross
            return None
        else:
            gate_pose = self._passing_gate_poses[min_index]
            self._visited_gate_poses.append(gate_pose)
            return gate_pose

    def _get_exploratory_pose(self):
        goal_pose = None
        transform = self._tf_buffer.lookup_transform(target_frame=self._base_link_frame,
                                                     source_frame=self._map_frame,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(
                                                         1.0)
                                                     )

        inverse_transform = self._tf_buffer.lookup_transform(target_frame=self._map_frame,
                                                             source_frame=self._base_link_frame,
                                                             time=rospy.Time.now(),
                                                             timeout=rospy.Duration(
                                                                 1.0)
                                                             )

        unclassified_objects_in_local_frame = [tf2_geometry_msgs.do_transform_pose(
            pose, transform) for pose in self._unclassified_objects]

        def distance_to_object(gate_pose):
            x = gate_pose.pose.position.x
            y = gate_pose.pose.position.y
            return m.sqrt(x**2 + y**2)

        min_index = None
        min_distance = 0.0
        for index in range(len(unclassified_objects_in_local_frame)):
            distance = distance_to_object(
                unclassified_objects_in_local_frame[index])
            if min_index is None or (distance < min_distance and distance > 10.0):
                min_index = index
                min_distance = distance

        if min_index is not None:
            # Set sails halfway to the nearest unknown object
            local_goal_pose = unclassified_objects_in_local_frame[min_index]
            local_goal_pose.pose.position.x /= 3.0
            local_goal_pose.pose.position.y /= 3.0
            # Rewrite the orientation so that it's looking away from us
            quaternion = self._calculate_quaternion_for_x_vector(
                (local_goal_pose.pose.position.x,
                 local_goal_pose.pose.position.y)
            )
            local_goal_pose.pose.orientation.x = quaternion[0]
            local_goal_pose.pose.orientation.y = quaternion[1]
            local_goal_pose.pose.orientation.z = quaternion[2]
            local_goal_pose.pose.orientation.w = quaternion[3]

            goal_pose = tf2_geometry_msgs.do_transform_pose(
                local_goal_pose, inverse_transform)

        # if we found no uknown goal with the previous approach, move forward a bit, maybe
        # something will come our way
        if goal_pose is None:
            goal_pose = self._get_current_map_pose()
            goal_pose.pose.position.x += self._exploration_forward_step

        return goal_pose

    def _calculate_quaternion_for_x_vector(self, vec):
        vx, vy = vec
        norm = m.sqrt(vx**2 + vy**2)
        nvx, nvy = vx / norm, vy / norm
        director_x = np.asarray((nvx, nvy, 0.0, 0.0))
        director_y = np.asarray((-nvy, nvx, 0.0, 0.0))
        director_z = np.asarray((0.0, 0.0, 1.0, 0.0))
        translation = np.asarray((0.0, 0.0, 0.0, 1.0))
        rot_matrix = np.asarray(
            (director_x, director_y, director_z, translation)).transpose()
        quaternion = quaternion_from_matrix(rot_matrix)
        return quaternion

    def _get_current_map_pose(self):
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

        transform = self._tf_buffer.lookup_transform(target_frame=self._map_frame,
                                                     source_frame=self._base_link_frame,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(
                                                         1.0)
                                                     )

        return tf2_geometry_msgs.do_transform_pose(
            source_frame_point, transform)

    def _black_box_pose_callback(self, point_msg):
        rospy.loginfo_throttle_identical(3, "Receiving infomation from the acoustic sensor...")
        pose = PoseStamped()
        pose.header = point_msg.header
        pose.pose.position = point_msg.point
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        self._black_box_pose = pose
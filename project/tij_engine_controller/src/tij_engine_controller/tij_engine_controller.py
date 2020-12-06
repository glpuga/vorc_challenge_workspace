#!/usr/bin/env python
import rospy
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_srvs.srv import SetBool, SetBoolResponse


class PIControlClass(object):
    def __init__(self, id, ki, kp):
        self._id = id
        self._ki = ki
        self._kp = kp
        self.reset_state()

    def reset_state(self):
        self._accumulator_state = 0.0

    def update_filter_state(self, error, delta_t):
        # calculate filter branches
        new_accumulator = self._accumulator_state + error
        ibranch = new_accumulator * self._ki * delta_t
        pbranch = error * self._kp
        # calculate filter control_action
        control_action = ibranch + pbranch
        rospy.loginfo("[%s] Loop update:\n  error=%f\n  ibranch=%f\n  pbranch=%f\n  acc=%f\n  output=%f" % (
            self._id, error, ibranch, pbranch, new_accumulator, control_action))
        self._accumulator_state = new_accumulator
        return control_action


class ThrottleControlNode(object):
    def __init__(self):
        self._linear_control_ki = 0.1
        self._linear_control_kp = 0.4

        self._angular_control_ki = 1.0
        self._angular_control_kp = 2.0

        self._max_valid_odometry_velocity = 15
        self._max_command_duration = 1.0
        self._latest_command_timestamp = rospy.Time.now()

        # initialize controller state
        self._controller_enabled = True
        self._state_timestamp = None
        self._reset_control_setpoints()

        self._linear_control = PIControlClass(
            "linear_control",
            self._linear_control_ki,
            self._linear_control_kp
        )
        self._angular_control = PIControlClass(
            "angular_control",
            self._angular_control_ki,
            self._angular_control_kp
        )

        # Publishers
        self._left_engine_pub = rospy.Publisher(
            '/cora/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self._right_engine_pub = rospy.Publisher(
            '/cora/thrusters/right_thrust_cmd', Float32, queue_size=10)

        # service providers
        self._enable_controller_service = rospy.Service(
            '/tij/enable_engine_controller', SetBool, self._enable_controller_service_callback)

        # Subscribers
        self._cmd_vel_sub = rospy.Subscriber(
            "/cora/cmd_vel", Twist, self._twist_callback)
        self._cmd_vel_sub = rospy.Subscriber(
            "/cora/robot_localization/odometry/filtered", Odometry, self._pose_callback)

    def _enable_controller_service_callback(self, req):
        self._controller_enabled = req.data
        if self._controller_enabled:
            rospy.logwarn("Engine controller set to enabled")
        else:
            rospy.logwarn("Engine controller set to disabled")
        return SetBoolResponse(success = True, message = "")

    def _reset_control_setpoints(self):
        self._expected_linear_speed = 0.0
        self._expected_angular_speed = 0.0

    def _twist_callback(self, twist_msg):
        self._expected_linear_speed = twist_msg.linear.x
        self._expected_angular_speed = twist_msg.angular.z
        self._latest_command_timestamp = rospy.Time.now()
        rospy.loginfo_throttle_identical(
            1, "Linear speed setpoint: %f" % (self._expected_linear_speed))
        rospy.loginfo_throttle_identical(
            1, "Angular speed setpoint: %f" % (self._expected_angular_speed))

    def _pose_callback(self, pose_msg):
        if self._state_timestamp is None:
            self._state_timestamp = pose_msg.header.stamp
            return

        if not self._controller_enabled:
            rospy.logwarn_throttle_identical(1, "Engine controller disabled, control loop disengaged")
            return

        # Validate input based on speed estimation. This is just a hack to
        # solve a transient in the EKF filter during startup.
        # TODO Figure out why there's a transient in the ekf filter during startup
        absolute_velocity = math.sqrt(
            pose_msg.twist.twist.linear.x**2 + pose_msg.twist.twist.linear.y**2)
        if (absolute_velocity > self._max_valid_odometry_velocity):
            # Ignore this sample
            rospy.logwarn_throttle_identical(
                3, "Ignoring pose message, speed out of range")
            return

        # If we haven't received a command in a while, reset expected values
        now = rospy.Time.now()
        age_of_command = (now - self._latest_command_timestamp).to_sec()
        if (age_of_command > self._max_command_duration):
            self._latest_command_timestamp = now
            rospy.logwarn(
                "Ran too long with no commands, resetting loop setpoints to zero.")
            self._reset_control_setpoints()

        delta_t = (pose_msg.header.stamp - self._state_timestamp).to_sec()
        self._state_timestamp = pose_msg.header.stamp

        rospy.loginfo_throttle_identical(1, "Loop time delta: %f" % (delta_t))

        current_linear_speed = pose_msg.twist.twist.linear.x
        current_angular_speed = pose_msg.twist.twist.angular.z

        rospy.loginfo("Loop inputs:\n  linear setpoint=%f\n  linear feedback=%f\n  angular setpoint=%f\n  angular feedback=%f"
                      % (self._expected_linear_speed, current_linear_speed, self._expected_angular_speed, current_angular_speed))

        linear_speed_error = self._expected_linear_speed - current_linear_speed
        angular_speed_error = self._expected_angular_speed - current_angular_speed

        linear_speed_control_action = self._linear_control.update_filter_state(
            linear_speed_error, delta_t)
        angular_speed_control_action = self._angular_control.update_filter_state(
            angular_speed_error, delta_t)

        within_actuator_range = self.send_engine_messages(
            linear_speed_control_action - angular_speed_control_action,
            linear_speed_control_action + angular_speed_control_action)

        if (not within_actuator_range):
            rospy.logwarn(
                "Actuators outside of control range, anti-windup activated!")
            self._linear_control.reset_state()
            self._angular_control.reset_state()

    def send_engine_messages(self, left_value, right_value):
        within_actuator_range = True
        if (abs(left_value) > 1.0) or (abs(right_value) > 1.0):
            # We are beyond the range of the actuator. We'll normalize
            # the engine outputs to keep some balance, and
            # communicate back the error to avoid updating the filter
            # state in this condition
            norm = max(abs(left_value), abs(right_value))
            left_value /= norm
            right_value /= norm
            within_actuator_range = False

        rospy.loginfo("Throttling engines: left %f, right %f" %
                      (left_value, right_value))
        left_msg = Float32()
        left_msg.data = left_value
        self._left_engine_pub.publish(left_msg)
        right_msg = Float32()
        right_msg.data = right_value
        self._right_engine_pub.publish(right_msg)
        return within_actuator_range

    def run(self):
        # TODO Improve. This is to skip the localization transient on startup.
        rospy.sleep(20.0)
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tij_engine_controller', anonymous=True)
    ThrottleControlNode().run()

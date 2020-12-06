#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from vrx_gazebo.msg import Task

from station_keeping import StationKeepingTaskManager
from wayfinding import WayfindingTaskManager
from perception import PerceptionTaskManager
from gymkhana import GymkhanaTaskManager


class DummyTaskManager(object):
    def __init__(self):
        pass

    def transitioned_to_ready(self):
        pass

    def transitioned_to_running(self):
        pass

    def transitioned_to_finished(self):
        pass


class GameMasterNode(object):
    def __init__(self):
        self._task_info_topic = rospy.get_param("~task_info_topic",
                                                "/vorc/task/info")

        # don't let anything uninitialized
        self._task_name = None
        self._current_task_state = None
        self._task_manager = DummyTaskManager

        self._task_manager_catalog = {
            "stationkeeping": StationKeepingTaskManager,
            "wayfinding": WayfindingTaskManager,
            "perception": PerceptionTaskManager,
            "gymkhana": GymkhanaTaskManager
        }

        # publishers

        # subscribers
        self._task_info_sub = rospy.Subscriber(
            self._task_info_topic, Task, self._task_callback)

    def _task_callback(self, task_msg):
        task_name = task_msg.name
        task_state = task_msg.state
        task_ready_time = task_msg.ready_time
        task_running_time = task_msg.running_time
        task_elapsed_time = task_msg.elapsed_time
        task_current_score = task_msg.score

        if task_state != self._current_task_state:
            self._current_task_state = task_state
            self._transition_to_state(task_name, task_state)

        # rospy.loginfo("Task message received:")
        # rospy.loginfo("  - Task name     : {}".format(task_name))
        # rospy.loginfo("  - Task state    : {}".format(task_state))
        # rospy.loginfo("  - Ready time    : {}".format(task_ready_time))
        # rospy.loginfo("  - Running time  : {}".format(task_running_time))
        # rospy.loginfo("  - Elapsed time  : {}".format(task_elapsed_time))
        # rospy.loginfo("  - Current score : {:.2f}".format(task_current_score))

    def _instantiate_task_manager(self, task_name):
        if not self._task_manager_catalog.has_key(task_name):
            rospy.logerr(
                "Unknown task type (%s), no task manager will be created!" % (task_name))
        else:
            self._task_manager = self._task_manager_catalog[task_name]()

    def _transition_to_state(self, task_name, task_state):
        rospy.logwarn("Transitioning task state to %s" % (task_state))

        if self._task_name is None:
            self._task_name = task_name
            rospy.logwarn("Instantiating task manager for %s" % (task_name))
            self._instantiate_task_manager(task_name)
        else:
            if task_state == "initial":
                pass
            elif task_state == "ready":
                self._task_manager.transitioned_to_ready()
            elif task_state == "running":
                self._task_manager.transitioned_to_running()
            elif task_state == "finished":
                self._task_manager.transitioned_to_finished()
            else:
                rospy.logerr("Unknown task state! What is %s?" % (task_state))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tij_game_master_node', anonymous=True)
    GameMasterNode().run()

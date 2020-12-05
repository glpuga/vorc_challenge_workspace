#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from tf.transformations import quaternion_matrix
from sensor_msgs import point_cloud2

import numpy as np
import cv2 as cv
import csv
from matplotlib import pyplot as plt
from collections import Counter

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tij_object_recognition.msg import DetectionData
from tij_object_recognition.msg import DetectionDataArray


class PhysicalBuoy:

    def __init__(self, location, buffer_min=10, buffer_max=30, probability_threshold=0.7, negative_observation_threshold=10):
        self._observations = []
        self._location = location
        self._probability_threshold = probability_threshold
        self._buffer_min = buffer_min
        self._buffer_max = buffer_max
        self._negative_observation_threshold = negative_observation_threshold
        self._negative_observations_counter = int(
            negative_observation_threshold / 2)

    def get_location(self):
        return self._location

    def add_positive_observation(self, classification):
        self._observations.append(classification)
        # keep the list shorter than a maximum length
        if len(self._observations) > self._buffer_max:
            del(self._observations[0])
        # Decrease the negative observations counter
        if self._negative_observations_counter:
            self._negative_observations_counter -= 1

    def add_negative_observation(self):
        self._negative_observations_counter += 1

    def is_dead(self):
        return self._negative_observations_counter > self._negative_observation_threshold

    def get_most_likely_classification(self):
        observation_count = len(self._observations)
        if observation_count < self._buffer_min:
            return None
        # count observations
        counters = Counter(self._observations)
        scores = counters.most_common()
        lead_category, lead_count = scores[0]
        lead_probability = float(lead_count) / observation_count
        if lead_probability < self._probability_threshold:
            return None
        return lead_category, lead_probability


class BuoyClassifier:

    def __init__(self, filename):
        with open(filename) as fd:
            reader = csv.reader(fd)
            reader = list(reader)
            reader = reader[1:]  # skip the header
        rgb_data = [(sclass, float(width), float(height), float(r), float(g), float(b))
                    for (sclass, height, width, r, g, b, h, s, v) in reader]
        hsv_data = [(sclass, float(width), float(height), float(h), float(s), float(v))
                    for (sclass, height, width, r, g, b, h, s, v) in reader]

        self.class_tags = [item[0] for item in rgb_data]

        self.rgb_data = np.asarray([item[1:] for item in rgb_data])
        self.rgb_data_normalization_vector = self._get_normalization_vector(
            self.rgb_data)
        self.rgb_data /= self.rgb_data_normalization_vector

        self.hsv_data = np.asarray([item[1:] for item in hsv_data])
        self.hsv_data_normalization_vector = self._get_normalization_vector(
            self.hsv_data)
        self.hsv_data /= self.hsv_data_normalization_vector

    def _get_normalization_vector(self, table):
        return table.max(axis=0)

    def classify_buoy_sample_rgb(self, size, rgb_color, k=5, filter_coef=[1, 1, 1, 1, 1]):
        vfilter = np.array(filter_coef)
        vsample = np.array(size + rgb_color) / \
            self.rgb_data_normalization_vector
        sqred_distances = np.sum(
            (self.rgb_data - vsample)**2 * vfilter, axis=1)
        cd_pairs = zip(self.class_tags, sqred_distances)
        sorted_cd_pairs = sorted(cd_pairs, key=lambda cd: cd[1])
        k_neighbours = Counter(
            [class_tag for class_tag, _ in sorted_cd_pairs[:5]])
        return k_neighbours.most_common()[0][0]

    def classify_buoy_sample_hsv(self, size, hsv_color, k=5, filter_coef=[1, 1, 1, 1, 1]):
        vfilter = np.array(filter_coef)
        vsample = np.array(size + hsv_color) / \
            self.hsv_data_normalization_vector
        sqred_distances = np.sum(
            (self.hsv_data - vsample)**2 * vfilter, axis=1)
        cd_pairs = zip(self.class_tags, sqred_distances)
        sorted_cd_pairs = sorted(cd_pairs, key=lambda cd: cd[1])
        k_neighbours = Counter(
            [class_tag for class_tag, _ in sorted_cd_pairs[:5]])
        return k_neighbours.most_common()[0][0]


class BuoyRecognitionNode(object):
    def __init__(self):
        self._tf_buffer = tf2_ros.Buffer(
            rospy.Duration(1200.0))  # tf buffer length
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._camera_input_image_raw_topic = rospy.get_param("~camera_input_image_raw_topic",
                                                             "/cora/sensors/cameras/front_left_camera/image_raw")
        self._camera_input_camera_info_topic = rospy.get_param("~camera_input_camera_info_topic",
                                                               "/cora/sensors/cameras/front_left_camera/camera_info")
        self._tagged_image_output_topic = rospy.get_param("~tagged_image_output_topic",
                                                          "/tij/camera/tagged_image")
        self._point_cloud_data_topic = rospy.get_param("~pointcloud_data",
                                                       "/cora/sensors/lidars/front_lidar/points")
        self._buoy_markers_topic = rospy.get_param("~buoy_markers_topic",
                                                   "/tij/objects/markers")
        self._detection_array_topic = rospy.get_param("~detection_array_topic",
                                                      "/tij/detections/detection_array")

        self._buoy_color_samples_database_file = rospy.get_param("~buoy_color_samples_database_file",
                                                                 "samples.csv")
        self._depth_sensor_min_distance = rospy.get_param(
            "~depth_sensor_min_distance", 5)
        self._depth_sensor_max_distance = rospy.get_param(
            "~depth_sensor_max_distance", 100)
        self._depth_sensor_clustering_range = rospy.get_param(
            "~depth_sensor_clustering_range", 1.0)

        self._max_perception_reach = rospy.get_param(
            "~max_perception_reach", 100)
        self._max_classification_reach = rospy.get_param(
            "~max_classification_reach", 40)

        self._canny_threshold_min = rospy.get_param(
            "~canny_threshold_min", 100)
        self._canny_threshold_max = rospy.get_param(
            "~canny_threshold_max", 300)
        self._border_reduction_kernel_size = rospy.get_param(
            "~border_reduction_kernel_size", 7)

        self._map_frame_id = rospy.get_param(
            "~map_frame_id", "cora/odom")

        self._next_known_object_id = 0

        # don't let anything uninitialized
        self._image_mask = None
        self._laser_detected_objects = None

        self._cv_bridge = CvBridge()
        self._camera_model = None
        self._known_objects = {}

        self._buoy_classifier = BuoyClassifier(
            self._buoy_color_samples_database_file)

        # publishers
        self._tagged_image_output_pub = rospy.Publisher(
            self._tagged_image_output_topic, Image, queue_size=10)
        self._marker_publisher = rospy.Publisher(
            self._buoy_markers_topic, MarkerArray, queue_size=10)
        self._detection_array_pub = rospy.Publisher(
            self._detection_array_topic, DetectionDataArray, queue_size=10)

        # subscribers
        self._image_raw_sub = rospy.Subscriber(
            self._camera_input_image_raw_topic, Image, self._image_raw_callback, queue_size=1)
        self._camera_info_sub = rospy.Subscriber(
            self._camera_input_camera_info_topic, CameraInfo, self._camera_info_callback)
        self._camera_info_sub = rospy.Subscriber(
            self._point_cloud_data_topic, PointCloud2, self._point_cloud_data_callback)

    def _get_laser_detections_in_camera_frame(self, camera_frame_id, image_timestamp):
        # Convert objects detected in the point cloud data to the camera frame and
        # pixel coordiantes
        laser_detected_objects_in_camera_frame = []
        if not self._laser_detected_objects:
            return laser_detected_objects_in_camera_frame
        if not self._can_transform_between_frames(
                self._laser_detected_objects["frame_id"],
                self._laser_detected_objects["timestamp"],
                camera_frame_id,
                image_timestamp):
            return laser_detected_objects_in_camera_frame

        locations_in_world = [
            self._transform_location_between_frames(
                self._laser_detected_objects["frame_id"],
                self._laser_detected_objects["timestamp"],
                camera_frame_id,
                image_timestamp,
                detection_location) for detection_location in self._laser_detected_objects["detections"]
        ]
        laser_detected_objects_in_camera_frame = [
            {
                "location_in_world": loc,
                "location_in_image": self._camera_model.project3dToPixel((loc.x, loc.y, loc.z))
            } for loc in locations_in_world]
        return laser_detected_objects_in_camera_frame

    def _find_rectangle_center(self, rect):
        x, y, w, h = rect
        return (x + w / 2.0), (y + h / 2.0)

    def _is_within_rectangle(self, coord, rect):
        xt, yt = coord
        x1, y1, w, h = rect
        x2 = x1 + w
        y2 = y1 + h
        if ((xt >= x1) and (xt <= x2) and (yt >= y1) and (yt <= y2)):
            return True
        return False

    def _get_distance_between_points(self, coord_1, coord_2):
        x1, y1, = coord_1
        x2, y2, = coord_2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _match_laser_to_camera(self, laser_detection_data, camera_detection_data, camera_frame_id, image_timestamp):
        objects_found_data = []
        contours, rectangles, areas, rgb_colors, hsv_colors = camera_detection_data

        for laser_hit in laser_detection_data:
            loc_in_world = laser_hit["location_in_world"]
            loc_in_image = laser_hit["location_in_image"]
            # Don't look of laser hits behind the camera
            if loc_in_world.z < 0.0:
                continue

            # Look for the camera detection that's located closest to the point of impact
            # of the laser
            best_rgb = None
            best_hsv = None
            best_rect = None
            best_distance = 1e6  # should be larger than anything possible
            for rect, rgb, hsv in zip(rectangles, rgb_colors, hsv_colors):
                rect_center = self._find_rectangle_center(rect)
                distance_to_center = self._get_distance_between_points(
                    rect_center, loc_in_image)
                if self._is_within_rectangle(loc_in_image, rect) and (distance_to_center < best_distance):
                    best_rgb = rgb
                    best_hsv = hsv
                    best_rect = rect
                    best_distance = distance_to_center
            # if we found a match, we create a register
            if best_rgb is not None and loc_in_world.z < self._max_perception_reach:
                object_width = best_rect[2] * \
                    loc_in_world.z / self._camera_model.fx()
                object_height = best_rect[3] * \
                    loc_in_world.z / self._camera_model.fy()
                hit_info = {}
                hit_info["location_in_world"] = self._transform_location_between_frames(
                    camera_frame_id, image_timestamp,
                    self._map_frame_id, image_timestamp, loc_in_world)
                hit_info["rectangle_in_image"] = best_rect
                hit_info["rgb_color"] = best_rgb
                hit_info["hsv_color"] = best_hsv
                hit_info["z_distance"] = loc_in_world.z
                hit_info["object_width"] = object_width
                hit_info["object_height"] = object_height
                hit_info["object_type"] = self._buoy_classifier.classify_buoy_sample_hsv(
                    (object_width, object_height), best_hsv
                )
                objects_found_data.append(hit_info)
        return objects_found_data

    def _check_if_objecs_is_already_known(self, new_entry):
       # walk through the elemets known to see if they match in the same location
        def distance_between_map_locations(p1, p2):
            # Don't measure distance in z, since that's height
            distance = np.sqrt((p1.x - p2.x)**2 +
                               (p1.y - p2.y)**2)
            return distance
        new_entry_location = new_entry["location_in_world"]
        for _, known_object in self._known_objects.items():
            known_entry_location = known_object.get_location()
            distance = distance_between_map_locations(
                new_entry_location, known_entry_location)
            identity_distance = max(1.0, new_entry["object_width"])
            if (distance < identity_distance):
                if (new_entry["z_distance"] < self._max_classification_reach):
                    known_object.add_positive_observation(
                        new_entry["object_type"])
                return True
        return False

    def _insert_new_object_in_database(self, new_entry):
        next_index = self._next_known_object_id
        self._next_known_object_id += 1
        self._known_objects[next_index] = PhysicalBuoy(
            new_entry["location_in_world"])
        if new_entry["z_distance"] < self._max_classification_reach:
            self._known_objects[next_index].add_positive_observation(
                new_entry["object_type"])

    def _match_known_objects_to_image(self, camera_detection_data, camera_frame_id, image_timestamp):
        contours, rectangles, areas, rgb_colors, hsv_colors = camera_detection_data
        image_height, image_width = self._image_mask.shape
        for _, known_object in self._known_objects.items():
            p = self._transform_location_between_frames(
                self._map_frame_id, image_timestamp,
                camera_frame_id, image_timestamp,
                known_object.get_location())
            # is this object within visual range?
            if p.z <= 0 or p.z >= self._max_perception_reach:
                continue
            # if it were there, would it be visible in the image?
            u, v = self._camera_model.project3dToPixel((p.x, p.y, p.z))
            u, v = int(u), int(v)
            if (u < 0) or (u >= image_width) or (v < 0) or (v >= image_height) or (self._image_mask[v, u] != 0):
                continue
            # it should be visible in the image, but is it?
            if not any([self._is_within_rectangle((u, v), rect) for rect in rectangles]):
                known_object.add_negative_observation()

    def _remove_dead_objects(self):
        objects_to_remove = [
            index for index, known_object in self._known_objects.items() if known_object.is_dead()]
        for key in objects_to_remove:
            classification = self._known_objects[key].get_most_likely_classification(
            )
            loc_str = str(self._known_objects[key].get_location())
            if classification:
                category, prob = classification
                rospy.logwarn("Removing dead object of type {} ({}) at location {}".format(
                    category, prob, loc_str))
            else:
                rospy.logwarn(
                    "Removing unclassified dead object at location {}".format(loc_str))
            self._known_objects.pop(key, None)

    def _publish_known_objects_info(self):
        if not len(self._known_objects):
            return
        rospy.loginfo("There are {} objects that have been located".format(
            len(self._known_objects)))

        detections_array_msg = DetectionDataArray()
        detections_array_msg.header.stamp = rospy.Time.now()
        detections_array_msg.header.frame_id = self._map_frame_id

        for det_id, known_object in self._known_objects.items():
            classification = known_object.get_most_likely_classification()

            detection_msg = DetectionData()
            detection_msg.id = det_id
            detection_msg.location = known_object.get_location()

            location_str = " ".join(str(known_object.get_location()).split())
            if classification:
                category, probability = classification

                detection_msg.classified = True
                detection_msg.probability = probability
                detection_msg.category = category

                rospy.loginfo("  - category: {:20s} - probability: {:5.2f} - location: {:30s}".format(
                    category, probability, location_str))
            else:
                detection_msg.classified = False
                detection_msg.probability = 0.0
                detection_msg.category = ""

                rospy.loginfo(
                    "  - category: {:20s} - probability: {:5.2f} - location: {:30s}".format("UNKNOWN", 0.0, location_str))

            detections_array_msg.detections.append(detection_msg)
        self._detection_array_pub.publish(detections_array_msg)

    def _process_objects_known_and_new(self, objects_found_data, camera_detections_data, camera_frame_id, image_timestamp):
        for new_entry in objects_found_data:
            known = self._check_if_objecs_is_already_known(new_entry)
            if not known:
                self._insert_new_object_in_database(new_entry)
        self._match_known_objects_to_image(
            camera_detections_data, camera_frame_id, image_timestamp)
        self._remove_dead_objects()
        self._publish_known_objects_info()

    def _publish_markers(self):
        marray = MarkerArray()
        for index, known_object in self._known_objects.items():
            loc = known_object.get_location()
            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = self._map_frame_id
            m.ns = "tij"
            m.id = index
            m.type = Marker.CUBE
            m.action = Marker.MODIFY
            m.pose.position = loc
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=1, y=1, z=1)
            m.color = ColorRGBA(0, 1.0, 0, 1.0)
            m.lifetime = rospy.Duration(1.0)
            m.frame_locked = True
            marray.markers.append(m)
        self._marker_publisher.publish(marray)

    def _tag_image_detections_on_output_image(self, output_cv_image, camera_detections_data):
        contours, rectangles, areas, rgb_colors, hsv_colors = camera_detections_data
        # Create the mask only once, then reuse
        for cnt, rect in zip(contours, rectangles):
            x, y, w, h = rect
            # mark the visual detection in the output image
            cv.rectangle(output_cv_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv.drawContours(output_cv_image, [cnt], -1, (0, 255, 0), 2)

    def _tag_laser_detections_on_output_image(self, output_cv_image, laser_detections_data):
        locations_in_image = [item["location_in_image"]
                              for item in laser_detections_data if item["location_in_world"].z >= 0]
        for loc in locations_in_image:
            x, y = loc
            x, y = int(x), int(y)
            cv.circle(output_cv_image, (x, y), 5, (255, 255, 0), 2)

    def _tag_hit_objects_in_image(self, output_cv_image, objects_found_data):
        # Create the mask only once, then reuse
        for detection in objects_found_data:
            x, y, w, h = detection["rectangle_in_image"]
            # mark the visual detection in the output image
            cv.putText(output_cv_image,
                       detection["object_type"], (x+5, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    def _image_raw_callback(self, image_msg):
        # Convert the image from ros msg to opencv domain
        input_cv_image = self._cv_bridge.imgmsg_to_cv2(
            image_msg, desired_encoding='rgb8')

        # If camera model is not ready, we can't do anything
        if self._camera_model is None:
            rospy.logwarn("Waiting for camera model information to arrive...")
            return
        # Generate the image mask only once.
        if self._image_mask is None:
            self._image_mask = self._create_view_for_image(input_cv_image)
        # Recover information from objects detected in the pointcloud
        laser_detections_data = self._get_laser_detections_in_camera_frame(
            image_msg.header.frame_id, image_msg.header.stamp)
        # process image to detect objects
        camera_detections_data = self._get_image_contours(
            input_cv_image, (self._canny_threshold_min,
                             self._canny_threshold_max),
            self._border_reduction_kernel_size)
        # match laser hits to camera, and filter remaining data
        objects_found_data = self._match_laser_to_camera(
            laser_detections_data, camera_detections_data, image_msg.header.frame_id, image_msg.header.stamp)
        # process objects to determine which are new and which are not
        self._process_objects_known_and_new(
            objects_found_data, camera_detections_data, image_msg.header.frame_id, image_msg.header.stamp)
        # publish markers
        self._publish_markers()
        # create output image by tagging known information in it
        output_cv_image = input_cv_image.copy()
        # add information recovered through detection to the image
        self._tag_image_detections_on_output_image(
            output_cv_image, camera_detections_data)
        self._tag_laser_detections_on_output_image(
            output_cv_image, laser_detections_data)
        self._tag_hit_objects_in_image(
            output_cv_image, objects_found_data)
        # Convert the image from opencv domain to ros msg
        output_image = self._cv_bridge.cv2_to_imgmsg(
            output_cv_image, encoding='rgb8')
        # Publish information
        self._tagged_image_output_pub.publish(output_image)

    def _camera_info_callback(self, camera_info_msg):
        if not self._camera_model:
            self._camera_model = PinholeCameraModel()
            self._camera_model.fromCameraInfo(camera_info_msg)
            rospy.loginfo("Got camera description message:")
            rospy.loginfo(" K = {0}, D = {1}".format(
                self._camera_model.intrinsicMatrix(),
                self._camera_model.distortionCoeffs()))

    def _point_cloud_data_callback(self, point_cloud_2_msg):
        # Convert the pointcloud2 data to a list of detected clusters
        # with coordinates, so that we deal with a smaller number of measurements
        detected_objects = self._clusterize_pointcloud_into_objects(
            point_cloud_2_msg)

        def np_vector_to_point(v): return Point(x=v[0], y=v[1], z=v[2])
        self._laser_detected_objects = {
            "frame_id": point_cloud_2_msg.header.frame_id,
            "timestamp": point_cloud_2_msg.header.stamp,
            "detections": [np_vector_to_point(v) for v in detected_objects]
        }

    def _clusterize_pointcloud_into_objects(self, point_cloud_2_msg):
        # Object count will be used to generate unique ids for clusters
        object_count = 0
        detected_objects = {}
        for hit in point_cloud2.read_points(point_cloud_2_msg, skip_nans=True):
            hit_pos = np.asarray(hit[0:3])
            # Reject measurements too close or too far
            laser_distance = np.sqrt(sum(hit_pos**2))
            if laser_distance < self._depth_sensor_min_distance or laser_distance > self._depth_sensor_max_distance:
                continue
            # Check if any of the previously known objects is within clustering distance of
            # the new detection. Also, for known objects, we average coordinates of multiple
            # hits within the cluster, to get a better idea of the center location.
            binned = False
            for object_id, obj_data in detected_objects.items():
                # accum_pos is the sum of all previous hits in this cluster, multiplicity
                # is the number of hits in the cluster
                accum_pos, multiplicity = obj_data
                distance = np.sqrt(
                    np.sum((accum_pos / multiplicity - hit_pos)**2))
                if (distance < self._depth_sensor_clustering_range):
                    # add this hit to the cluster
                    detected_objects[object_id] = (
                        accum_pos + hit_pos, multiplicity + 1)
                    binned = True
                    break
            # if we did not find a previous cluster near the hit, create a new cluster
            if not binned:
                detected_objects[object_count] = (hit_pos, 1)
                object_count += 1
        # average all the hits within each cluster
        objects = [value[0] / value[1]
                   for key, value in detected_objects.items()]
        return objects

    def _create_view_for_image(self, input_image):
        mask = np.zeros(input_image.shape[:2], np.uint8)
        height, width, _ = input_image.shape
        y1 = int(height * 0.25)
        y2 = int(height * 0.55)
        x1 = int(width * 0.2)
        x2 = int(width / 2.0)
        x3 = width - x1
        pts = np.array([(x1, height), (x2, y2), (x3, height)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.fillPoly(mask, [pts], True, 255)
        cv.rectangle(mask, (0, 0), (width, y1), 255, -1)
        return mask

    def _get_image_contours(self, input_image, canny_args=(200, 300), kernel_size=7):
        hsv_image = cv.cvtColor(input_image, cv.COLOR_RGB2HSV)
        edges = cv.Canny(input_image, canny_args[0], canny_args[1])
        blurred = cv.GaussianBlur(edges, (3, 9), 0)
        _, contours, __ = cv.findContours(
            blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rectangles = [cv.boundingRect(cnt) for cnt in contours]
        areas = [cv.contourArea(cnt) for cnt in contours]
        rgb = []
        hsv = []
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # contours = [cv.convexHull(cnt) for cnt in contours] TODO add this again
        for cnt in contours:
            mask = np.zeros(input_image.shape[:2], np.uint8)
            cv.drawContours(mask, [cnt], 0, 255, -1)
            eroded_mask = cv.erode(mask, kernel, iterations=1)
            rgb_color = cv.mean(input_image, mask=eroded_mask)
            hsv_color = cv.mean(hsv_image, mask=eroded_mask)
            rgb.append(rgb_color[:3])
            hsv.append(hsv_color[:3])

        def masked_out(x, y, w, h):
            xc, yc = self._find_rectangle_center((x, y, w, h))
            xc, yc = int(xc), int(yc)
            return (self._image_mask is None or self._image_mask[yc, xc] != 0)

        visibility_filter = [not masked_out(*rect) for rect in rectangles]

        fcontours = [contours[i] for i in range(
            len(contours)) if visibility_filter[i]]
        frectangles = [rectangles[i] for i in range(
            len(contours)) if visibility_filter[i]]
        fareas = [areas[i]
                  for i in range(len(contours)) if visibility_filter[i]]
        frgb = [rgb[i]
                for i in range(len(contours)) if visibility_filter[i]]
        fhsv = [hsv[i]
                for i in range(len(contours)) if visibility_filter[i]]

        return fcontours, frectangles, fareas, frgb, fhsv

    def _transform_location_between_frames(self, source_frame, source_timestamp, dest_frame, dest_timestamp, location):
        transform = self._tf_buffer.lookup_transform_full(target_frame=dest_frame,
                                                          target_time=dest_timestamp,
                                                          source_frame=source_frame,
                                                          source_time=source_timestamp,
                                                          fixed_frame="cora/odom",
                                                          timeout=rospy.Duration(
                                                              1.0)
                                                          )

        original = PointStamped()
        original.header.frame_id = source_frame
        original.header.stamp = source_timestamp
        original.point = location
        pose = tf2_geometry_msgs.do_transform_point(original, transform)
        return Point(x=pose.point.x, y=pose.point.y, z=pose.point.z)

    def _can_transform_between_frames(self, source_frame, source_timestamp, dest_frame, dest_timestamp):
        return self._tf_buffer.can_transform_full(target_frame=dest_frame,
                                                  target_time=dest_timestamp,
                                                  source_frame=source_frame,
                                                  source_time=source_timestamp,
                                                  fixed_frame="cora/odom",
                                                  timeout=rospy.Duration(1.0))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('tij_buoy_recognition_node', anonymous=True)
    BuoyRecognitionNode().run()

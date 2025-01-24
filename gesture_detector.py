#!/usr/bin/python3.8

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO

class GestureDetector:

    def __init__(self):
        rospy.init_node('gesture_detector')
        self.bridge = CvBridge()

        best_pt_path = "./model.pt"
        try:
            self.model = YOLO(best_pt_path)
        except Exception as error:
            rospy.logerror(f"Error to load Model Yolo")
            rospy.logerror(f"{error}")


        # Init subscribers
        rospy.Subscriber("/uav2/rgbd/infra1/image_raw", Image, self.callbackSubscriberCameraInfra, queue_size=1)

        # Init publishers
        self.pub = rospy.Publisher('/uav2/gesture_detector/detection', String, queue_size=1)

        self.msg_list = []
        rospy.loginfo("[GestureDetector - Detect]: initialized!")
        self.processed = False

    def callbackSubscriberCameraInfra(self, imageInfra):
        camera_image = self.bridge.imgmsg_to_cv2(imageInfra, "8UC1")

        image = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)

        # Run YOLO inference
        results = self.model(image)

        # Extract class names from results
        detection = False
        confidence = 0
        class_id = None
        class_name = None
        string = String()
        for result in results:
            for box in result.boxes:
                class_id = box.cls.item()  # Get class ID
                class_name = self.model.names[int(class_id)]
                confidence = box.conf.item()  # Get confidence score
                detection = True

        # Plot the results (draw bounding boxes, labels, etc.)
        if detection:
            annotated_frame = results[0].plot()
            print(confidence)

            if confidence >= 0.7:
                if class_name == 'dislike':
                    string.data = 'land'
                    self.msg_list.append(string)
                elif class_name == 'fist':
                    string.data = 'move_ahead'
                    self.msg_list.append(string)
                elif class_name == 'palm':
                    string.data = 'move_back'
                    self.msg_list.append(string)
                elif class_name == 'like':
                    string.data = 'yaw_to_left'
                    self.msg_list.append(string)
                elif class_name == 'peace':
                    string.data = 'yaw_to_right'
                    self.msg_list.append(string)
                elif class_name == 'two_up_inverted':
                    string.data = 'move_to_left'
                    self.msg_list.append(string)
                elif class_name == 'two_up':
                    string.data = 'move_to_right'
                    self.msg_list.append(string)
                elif class_name == 'one':
                    string.data = 'takeoff'
                    self.msg_list.append(string)
                elif class_name == 'ok':
                    string.data = 'move_up'
                    self.msg_list.append(string)
                elif class_name == 'rock':
                    string.data = 'move_down'
                    self.msg_list.append(string)

            print(self.msg_list)
            if len(self.msg_list) >= 3 and self.msg_list[-1].data != self.msg_list[-2].data:
                self.msg_list = []
            if len(self.msg_list) >= 3:
                print("sending msg")
                self.pub.publish(self.msg_list[0])
                self.msg_list = []

if __name__ == '__main__':
    gesture_detector = GestureDetector()
    rospy.spin()

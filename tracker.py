import rospy

from pytracking.evaluation import Tracker
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

import select
import cv2
import numpy as np
import socket
import sys
import pickle
import struct

class ObjectTracker:

    def __init__(self):
        self.clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            self.clientsocket.connect(('172.17.0.1', 5556))
            #self.clientsocket.setblocking(False)
        except Exception as e:
            self.clientsocket = None

        rospy.init_node("tracker")
        rospy.Subscriber("image", Image, self.image_callback, queue_size=1)
        self.bb_pub = rospy.Publisher("bounding_box", Float32MultiArray, queue_size=10)
        self.tracker = Tracker("dimp", "dimp18")
        self.init = False
        self.bridge = CvBridge()

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        frame = cv_image.copy()
        if not self.init:
            if self.clientsocket is not None:
                data = pickle.dumps(cv_image)
                message_size = struct.pack("L", len(data)) ### CHANGED
                self.clientsocket.sendall(message_size + data)
                ready_to_read, _, _ = select.select([self.clientsocket], [], [], 20)
                if ready_to_read:
                    bounding_box = self.clientsocket.recv(1024)
                    bounding_box = [int(value) for value in bounding_box.decode().split(",")]
                    self.tracker.init(frame, optional_box=bounding_box)
                    print(bounding_box)
            self.init = True
        x, y, w, h, flag, score =  self.tracker.run_frame(cv_image)
        bb_msg = Float32MultiArray()
        flag = 1 if flag == "normal" else 0
        bb_msg.data = [x, y, w, h, flag, score]
        self.bb_pub.publish(bb_msg)
        cv2.rectangle(cv_image, (x, y), (w, h),
                         (0, 255, 0), 5)
        data = pickle.dumps(cv_image)
        message_size = struct.pack("L", len(data)) ### CHANGED
        if self.clientsocket is not None:
            self.clientsocket.sendall(message_size + data)
            ready_to_read, _, _ = select.select([self.clientsocket], [], [], 0.01)
            if ready_to_read:
                temp = self.clientsocket.recv(1024)
                bounding_box = self.clientsocket.recv(1024)
                bounding_box = [int(value) for value in bounding_box.decode().split(",")]
                self.tracker.init(frame, optional_box=bounding_box)
                print(bounding_box)


if __name__ == '__main__':
   try:
       tracker = ObjectTracker()
       rospy.spin()
   except rospy.ROSInterruptException:
       pass

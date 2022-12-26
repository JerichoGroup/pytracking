import cv2

import rospy 
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class SendImage:

    def __init__(self, video_path):
        rospy.init_node("stream")
        self.pub = rospy.Publisher('image', Image, queue_size=1)
        self.cap = cv2.VideoCapture(video_path)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(10)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            self.rate.sleep()

 
if __name__ == '__main__':
    try:
        SendImage("/home/jetson/beU/semi_stable_door1.mp4").run()
    except rospy.ROSInterruptException:
         pass

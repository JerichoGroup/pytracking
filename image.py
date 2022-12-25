import rospy
from pytracking.evaluation import Tracker
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError

class ObjectTracker:

    def __init__(self):
        rospy.init_node("tracker")
        rospy.Subscriber("image", Image, self.image_callback)
        self.bb_pub = rospy.Publisher("bounding_box", Int32MultiArray, queue_size=10)
        self.tracker = Tracker("dimp", "dimp18")
        self.init = False
        self.bridge = CvBridge()

    def image_callback(self, image_msg):

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        print("receive image")
        if not self.init:
            self.tracker.init(cv_image)
        x, y,w, h =  self.tracker.run_frame(cv_image)
        bb_msg = Int32MultiArray()
        bb_msg.data = [x,y,w,h]

        self.bb_pub.publish(bb_msg)



    def image_msg_to_np(self, image_msg):
        pass


    def find_object(self, image):
        pass

 
if __name__ == '__main__':
   try:
       x = ObjectTracker()
       rospy.spin()
   except rospy.ROSInterruptException:
       pass

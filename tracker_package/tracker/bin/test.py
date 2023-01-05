import sys, unittest, time
import numpy as np
import rospy, rostest


from std_msgs.msg import Float32MultiArray

class TestTracker(unittest.TestCase):
    def __init__(self, *args):
        super(TestTracker, self).__init__(*args)
        self.bounding_box = None
    
    def callback(self, msg):
        self.bounding_box = msg.data  # Store to check values in test cases
        self.done = True # Set done to true in order to end waiting loop

    # First test case
    def test_tracker(self):
        self.done = False
        rospy.init_node("test")
        sub = rospy.Subscriber("/bounding_box", Float32MultiArray, self.callback)
        
        # Wait ten seconds -- can use ROS blocking functions instead
        timeout_t = time.time() + 30.0 #10 seconds
        while not rospy.is_shutdown() and not self.done and time.time() < timeout_t:
            pass
        
        self.assertNotEqual(self.bounding_box, None)

if __name__ == '__main__':
    rostest.rosrun("test", "test", TestTracker, sys.argv)

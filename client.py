import cv2
import numpy as np
import socket
import sys
import pickle
import struct

cap=cv2.VideoCapture("/home/jetson/beU/semi_stable_door1.mp4")
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost', 5555))

while True:
    ret,frame=cap.read()
    # Serialize frame
    data = pickle.dumps(frame)

    # Send message length first
    message_size = struct.pack("L", len(data)) ### CHANGED

    # Then data
    clientsocket.sendall(message_size + data)

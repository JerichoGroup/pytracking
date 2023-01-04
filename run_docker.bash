sudo docker  run --runtime=nvidia -v /usr/bin/ninja:/usr/bin/ninja -v /home/jetson:/home/jetson -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1  -it 0656148dad46 /bin/bash
PYTHONPATH=$PYTHONPATH:/home/jetson/archiconda3/envs/track/lib/python3.6/site-packages/:/home/jetson/.local/lib/python3.6/site-packages/:/home/jetson/ros/pytracking/
source /workspace/ros_catkin_ws/devel/setup.bash

FROM ros:humble

RUN apt update && apt install -y \
    ros-humble-demo-nodes-py \
    ros-humble-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

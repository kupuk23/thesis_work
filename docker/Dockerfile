FROM osrf/ros:humble-desktop-full-jammy

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rqt-runtime-monitor \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS 2 environment variables
ENV ROS_DISTRO=humble

# Set the workspace directory
RUN mkdir -p /root/ros2_ws/src \
    && mkdir -p /root/discower_ws/src

# clone the git repo to src
WORKDIR /root/discower_ws/src
RUN git clone https://github.com/DISCOWER/px4-mpc.git --recurse-submodules \
    && git clone https://github.com/kupuk23/px4-mpvs.git --recurse-submodules



WORKDIR /root/ros2_ws/src
RUN git clone https://github.com/kupuk23/thesis_work.git --recurse-submodules



# Install workspace dependencies and build (uncomment if needed)
# RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
#     colcon build --symlink-install

# Source ROS setup script on container start
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set display environment for RViz
ENV DISPLAY=:1

CMD ["bash"]

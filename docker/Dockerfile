FROM osrf/ros:humble-desktop-full-jammy

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rqt-runtime-monitor

# INSTALL GAZEBO HARMONIC
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release ca-certificates

RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable \
    $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

RUN apt-get update && apt-get install -y\
    gz-harmonic

ENV GZ_VERSION=harmonic     

# Set up ROS 2 environment variables
ENV ROS_DISTRO=humble

# Set the workspace directory
RUN mkdir -p /root/ros2_ws/src \
    && mkdir -p /root/discower_ws/src

# clone the git repo to src
WORKDIR /root/discower_ws/src
RUN git clone https://github.com/DISCOWER/px4-mpc.git --recurse-submodules \
    && git clone https://github.com/kupuk23/px4-mpvs.git --recurse-submodules \
    && git clone https://github.com/DISCOWER/px4_msgs.git \
    && git clone https://github.com/Jaeyoung-Lim/px4-offboard.git


WORKDIR /root/ros2_ws/src
RUN git clone https://github.com/kupuk23/thesis_work.git --recurse-submodules 

# Intall dependencies for PX4 and MPC
WORKDIR /root/discower_ws/
RUN rosdep update && \
    rosdep install -y --from-paths src --ignore-src

# build the packages
RUN . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install

WORKDIR /root/ros2_ws/
RUN rosdep update && \
    rosdep install -y --from-paths src --ignore-src

RUN . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install \
    && colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --packages-select go_icp_vendor pose_estimation_pcl --symlink-install --parallel-workers $(nproc)

RUN apt-get clean && rm -rf /var/lib/apt/lists/*







# Install workspace dependencies and build (uncomment if needed)
# RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
#     colcon build --symlink-install

# Source ROS setup script on container start
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set display environment for RViz
ENV DISPLAY=:1

CMD ["bash"]

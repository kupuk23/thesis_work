# FROM px4-space:base
FROM osrf/ros:humble-desktop-full-jammy

ARG  NB_THREADS=8             

ENV  DEBIAN_FRONTEND=noninteractive
ENV user=px4space
ENV home=/home/$user

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ros-humble-rqt-runtime-monitor \
    git build-essential cmake pkg-config \
    python3-venv python3-dev \
    ca-certificates \
    nano gedit gdb

# # INSTALL GAZEBO HARMONIC IF NEEDED
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl gnupg lsb-release ca-certificates

# RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
#     -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
#     echo "deb [arch=$(dpkg --print-architecture) \
#     signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
#     http://packages.osrfoundation.org/gazebo/ubuntu-stable \
#     $(lsb_release -cs) main" \
#     | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# RUN apt-get update && apt-get install -y\
#     gz-harmonic

# ENV GZ_VERSION=harmonic     

# # Set up ROS 2 environment variables
# ENV ROS_DISTRO=humble

# Set the workspace directory
RUN mkdir -p $home/tafarrel_ws/src \
    && mkdir -p $home/discower_ws/src

# clone the git repo to src
WORKDIR $home/discower_ws/src
RUN git clone https://github.com/DISCOWER/px4-mpc.git --recurse-submodules \
    && git clone https://github.com/kupuk23/px4-mpvs.git --recurse-submodules \
    && git clone https://github.com/DISCOWER/px4_msgs.git \
    && git clone https://github.com/Jaeyoung-Lim/px4-offboard.git

RUN . /opt/ros/humble/setup.sh \
    && cd $home/discower_ws \
    && rosdep update \
    && rosdep install -y --from-paths src --ignore-src \
    && colcon build --symlink-install


WORKDIR $home/tafarrel_ws/src

RUN git clone https://github.com/kupuk23/thesis_work.git --recurse-submodules 
RUN . /opt/ros/humble/setup.sh \
    && cd $home/tafarrel_ws \
    && rosdep update && \
    rosdep install -y --from-paths src --ignore-src \
    && colcon build --symlink-install \
    && colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --packages-select go_icp_vendor pose_estimation_pcl --symlink-install --parallel-workers $(nproc)



WORKDIR $home/
# ---- clone and compile acados -----------------------------------------------------------
RUN git clone https://github.com/acados/acados.git \
    && cd acados && git submodule update --recursive --init \
    && mkdir build && cd build \
    && cmake -DACADOS_WITH_QPOASES=ON .. \
    && make  install -j$NB_THREADS


# INSTALL and setting VIRTUALENV 
RUN python3 -m venv $home/env \
    && $home/env/bin/pip install --upgrade pip \
    && $home/env/bin/pip install -e $home/acados/interfaces/acados_template

# Add the path to the compiled shared libraries
ENV ACADOS_SOURCE_DIR="$home/acados"
ENV LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
# Add the path to the acados Python interface
ENV PYTHONPATH=$PYTHONPATH:$ACADOS_SOURCE_DIR/interfaces/acados_template/:$home/env/lib/python3.10/site-packages
ENV PATH=$home/env/bin:$PATH


RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install workspace dependencies and build (uncomment if needed)
# RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
#     colcon build --symlink-install

# Source ROS setup script on container start
RUN echo "source $home/tafarrel_ws/install/setup.bash" >> ~/.bashrc
RUN echo "source $home/discower_ws/install/setup.bash" >> ~/.bashrc
RUN echo "PROMPT_COMMAND='history -a'" >> ~/.bashrc
# Set display environment for RViz
ENV DISPLAY=:1

CMD ["bash"]

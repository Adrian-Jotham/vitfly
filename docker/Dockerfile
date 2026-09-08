FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# apt deps: build-essential/cmake/libzmqpp-dev/libopencv-dev/unzip/python3-catkin-tools
# match upstream setup_ros.bash. The libgl/libx* set is added on top because the base
# image is a headless ROS image, not a full desktop Ubuntu install, and the Flightmare
# Unity binary needs real GLX/X11 libs present to render through host GPU passthrough.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip vim \
    libzmqpp-dev libopencv-dev \
    python3-catkin-tools python3-pip python3-rosdep \
    libgl1 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 libxext6 \
    mesa-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements-ml.txt

RUN rosdep update

WORKDIR /root/catkin_ws
RUN mkdir -p src

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

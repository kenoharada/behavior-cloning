# Select base image: https://catalog.ngc.nvidia.com/containers
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    python3-dev python3 python3-pip python3-venv \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf swig \
    xvfb libglfw3-dev libosmesa-dev python-opengl \
    wget curl unzip git zsh vim ffmpeg

RUN apt update && apt install -y xvfb x11vnc python-opengl icewm python3-tk

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
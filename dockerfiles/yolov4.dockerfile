FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update 

RUN echo "== Install Basic Tools ==" &&\
    apt update &&\
    apt install -y --allow-unauthenticated \
        openssh-server vim nano htop tmux sudo \
        git unzip build-essential iputils-ping net-tools ufw \
        python3 python3-pip curl dpkg libgtk2.0-dev \
        cmake libwebp-dev ca-certificates gnupg git \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
        libatlas-base-dev gfortran \
        libgl1-mesa-glx libglu1-mesa-dev x11-utils x11-apps && \
    apt clean &&\
    rm -rf /var/lib/apt/list/*

RUN echo "== Install Dev Tolls ==" &&\
    pip3 install tensorflow==2.10 &&\
    pip3 install opencv-python &&\
    pip3 install matplotlib &&\
    pip3 install pillow &&\
    pip3 install protobuf==3.20.* &&\
    pip3 install tqdm &&\
    pip3 install tensorflow_datasets &&\
    pip3 install gdown &&\
    pip3 install pyyaml &&\
    pip3 install numpy==1.26 &&\
    pip3 install scipy

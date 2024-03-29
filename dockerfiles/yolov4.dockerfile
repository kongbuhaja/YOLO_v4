# sudo docker build --force-rm -f dockerfiles/yolov4.dockerfile -t yolov4:latest .
# sudo apt-get install x11-xserver-utils
# xhost +
# sudo docker run --gpus all --cpuset-cpus=0-31 -m 250g --shm-size=32g -it -v /tmp/.x11-unix:/tmp/.x11-unix -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 -p 6006:6006 -e DISPLAY=unix$DISPLAY --name yolov4 yolov4:latest
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update 

RUN echo "== Install Basic Tools ==" &&\
    apt install -y --allow-unauthenticated \
    openssh-server vim nano htop tmux sudo git unzip build-essential\
    python3 python3-pip curl dpkg libgtk2.0-dev \
    cmake libwebp-dev ca-certificates gnupg git \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran \
    libgl1-mesa-glx libglu1-mesa-dev x11-utils x11-apps
# CUDA 없애고 해보기 /cuda/이렇게 
# ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    # echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    # ldconfig

RUN echo "== Install Dev Tolls ==" &&\
    pip3 install tensorflow==2.8 &&\
    pip3 install opencv-python &&\
    pip3 install matplotlib &&\
    pip3 install pillow &&\
    pip3 install protobuf==3.20.* &&\
    pip3 install tqdm &&\
    pip3 install tensorflow_datasets &&\
    pip3 install gdown &&\
    pip3 install pyyaml &&\
    pip3 install numpy &&\
    pip3 install scipy

RUN cd /home/ &&\
    git clone https://github.com/kongbuhaja/YOLO_v4.git
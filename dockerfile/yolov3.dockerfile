# sudo docker build --force-rm -f yolov3.dockerfile -t yolov3:1.0 .
# sudo apt-get install x11-xserver-utils
# xhost +
# sudo docker run --gpus all -it -v /tmp/.x11-unix:/tmp/.x11-unix -e DISPLAY=unix$DISPLAY --name yolov3 yolov3:1.0
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update 

RUN echo "== Install Basic Tools ==" && \
    apt install -y --allow-unauthenticated \
    openssh-server vim nano htop tmux sudo git unzip build-essential\
    python3 python3-pip curl dpkg libgtk2.0-dev \
    cmake libwebp-dev ca-certificates gnupg git \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran \
    libgl1-mesa-glx libglu1-mesa-dev x11-utils x11-apps

ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

RUN pip3 install tensorflow==2.8 &&\
    pip3 install opencv-python &&\
    pip3 install matplotlib &&\
    pip3 install pillow &&\
    pip3 install protobuf==3.20.* &&\
    pip3 install tqdm &&\
    pip3 install tensorflow_datasets &&\
    pip3 install gdown

RUN cd /home/ &&\
    git clone https://github.com/kongbuhaja/YOLO_v3.git
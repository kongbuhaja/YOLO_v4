# sudo docker build --force-rm -f dockerfiles/yolov4.dockerfile -t yolov4:latest .
# sudo apt-get install x11-xserver-utils
# xhost +
# sudo docker run --gpus '"device=0"' --cpuset-cpus=0-7 -m 64g --shm-size=8g -it -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 --name yolov4_0 yolov4:latest
sudo docker run --gpus '"device=1"' --cpuset-cpus=8-15 -m 64g --shm-size=8g -it -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 --name yolov4_1 yolov4:latest
sudo docker run --gpus '"device=2"' --cpuset-cpus=16-23 -m 64g --shm-size=8g -it -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 --name yolov4_2 yolov4:latest
# sudo docker run --gpus all --cpuset-cpus=0-31 -m 250g --shm-size=32g -it -v /tmp/.x11-unix:/tmp/.x11-unix -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 -p 6006:6006 -e DISPLAY=unix$DISPLAY --name yolov4 yolov4:latest
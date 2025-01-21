# sudo docker build --force-rm -f yolov4.dockerfile -t yolov4:latest .
# # sudo apt-get install x11-xserver-utils
# # xhost +
# sudo docker run --gpus '"device=0,1,2"' --cpuset-cpus=32-63 -m 128g --shm-size=16g -it -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 -p 6006:6006 --name yolov4 yolov4:tf28
sudo docker build --force-rm -f yolov4.dockerfile -t yolov4:tf210 .
sudo docker run --gpus '"device=0,1,2,3"' --cpuset-cpus=0-39 -m 128g --shm-size=16g -it -v /home/dblab/ML/YOLO_v4:/home/YOLO_v4 --name yolov4_10 yolov4:tf210

#take a picture :)
fswebcam -r 1280x720 --no-banner image.jpg 
fswebcam -r 640x480 --jpeg 85 -D 1 HTN/img/web-cam-shot1.jpg




v4l2-ctl --list-devices

# list all video devices picked up by the kernel
ls -ltrh /dev/video*

# cpoy tu rpi
scp pi@192.168.0.24:/home/pi/HTN/img/*.jpg /home/tuannd

# kiem tra dung luong
df -h

### 1. Install Raspberry and Cars:
Visit [SunFounder](https://www.sunfounder.com/learn/category/Smart-Video-Car-for-Raspberry-Pi.html) site

### 2. Start VNC Session using Raspberry IP address 
> - **User**     : `pi`
> - **Password** : `raspberry`

You can retreive IP Address connecting display on Raspberry HDMI connector 


To change Display resolution :  
- [RealVNC Support](https://support.realvnc.com/Knowledgebase/Article/View/523/2/troubleshooting-vnc-server-on-the-raspberry-pi)
- [Raspberry Config Documentation](https://www.raspberrypi.org/documentation/configuration/config-txt.md)
    
File to be modified :  `boot/config.txt`
- Launch `sudo vi /boot/config.txt`
- Uncomment `hdmi_force_hotplug=1`
- Uncomment `hdmi_group` and set it to 2 (DTM mode)
- Uncomment `hdmi_mode` and set 
    - to `16` to force a resolution of 1024x768 at 60Hz
    - to `23` to force a resolution of 1280x768 at 60Hz
    - to `35` to force a resolution of 1280x1024 at 60Hz



### 3. Install source code on your PC and Raspberry :
Repo to clone : `git clone https://github.com/muratory/deep_drive.git`

- On Raspberry, clone repo in `$HOME`
    - Raspberry code is located in `$HOME/deep_drive/raspberry`
- Clone repo on your PC in `<path>`
    - Computer source code is located in `<path>/deep_drive/computer`

### 4.  Calibration :
-  Raspberry :
    - Folder    : `$HOME/deep_drive/raspberry/server`
    - Launch    : `sudo python cali_server.py`
    - or use alias  : `sc`
- On your PC :
    - Folder    : <path>\deep_drive\computer\client_sf\
    - Script    : client_App.py  `HOST = 'xxx.xxx.xxx.xxx'    # Server(Raspberry Pi) IP address`
    - Launch    : `python cali_client.py`
        
### 5. Check Camera :
- Folder    : `/home/pi/deep_drive/raspberry/mjpg-streamer/picam/mjpg-streamer`
- Launch    : `./start.sh &`
- or use alias  : `si`

On your computer launch following address : http://192.168.0.xxx:8000/stream.html
    
### 6. Raspberry TCP server : 
- Folder    : `$HOME/deep_drive/raspberry/server`
- Launch    : `sudo python tcp_server.py`
- or use alias  : `st`


### 7. Summary and shorcut
Create `.bash_aliases` et add the following lines :
```
alias si='cd /home/pi/deep_drive/raspberry/mjpg-streamer/picam/mjpg-streamer;./start.sh &'
alias st='cd /home/pi/deep_drive/raspberry/server;sudo python tcp_server.py'
alias ss='cd /home/pi/deep_drive/raspberry/server;sudo python sensor_server.py'
for calibration only :
alias sc='cd /home/pi/deep_drive/raspberry/server;sudo python cali_server.py'
alias sigps='cd /home/pi/deep_drive/raspberry/mjpg-streamer/mjpg-streamer;./startgps.sh &'
```

please note you may have to use regular mjpeg streamer if the Camera is a USB camera :
alias si='cd /home/pi/deep_drive/raspberry/mjpg-streamer/mjpg-streamer;./start.sh &'

The start.sh that is launched for video streaming should be :
for car (with picam and a 320x240 picture on port 8000):
./mjpg_streamer -o "output_http.so -w ./www -p 8000" -i "input_raspicam.so -x 320 -y 240 -fps 30"

for gps (with usb cam and a 640x480 video on port 8020):
./mjpg_streamer -i "./input_uvc.so -r 640x480 -f 20" -o "./output_http.so -w ./www -p 8020"




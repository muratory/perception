# Deep drive project

## Introduction
This README explains step by step all procedures to control the car by:
- Neural Network (using different type of NN to control the car) and/or;
- Path Control (using PID, graphSlam, GPS homemade and Land mark to control the car ) and/or;
- The road line detection (that detect side road to control the car).

Obviously you can use a mixture of all of these above procedures.


## 1) Car Calibration

In order to have the car moving correctly we recommend to follow all the steps from:
- [SunFounder](https://www.sunfounder.com/learn/category/Smart-Video-Car-for-Raspberry-Pi.html) site.

Especially the car [calibration](https://www.sunfounder.com/learn/Smart-Video-Car-for-Raspberry-Pi/51-55-smart-video-car.html) procedure that is necessary to have the car moving straigth correctly.


## 2) Run Servers on Raspberrys
Please follow `raspberry/README.md`:
- On the car:
  - `st` - server for sterring;
  - `si` - server for image;
  - `ss` - server for sensor ultrasonic.
- On the gps:
  - `sigps` - server for gps image.


## 3) GPS Calibration
**Warning** - To be done at least once (optional if you use your own Road but mandatory if 2 differents team use same circuit).

Please follow procedure in `gps_camera_calib_folder/README` to make your circuit undistorded.
Basically at the end of the procedure you will have a perfect undistorded and scaled '8' that can be use anywhere.

Please note you can find the road circuit used for this poc under `docs/misc/road/` directory.


## 4) Video Car Calibration
### File `car_calibration.py`
It is recommended to use a picamera on the car instead of the one delivered in the kit. The most important is to put the camera to see as close as possible the front of the car, the whole road and at least 50 cm above the car.

`car_calibration.py` can help you to set the middle properly and check your camera and your steering control works with the keys.


## 5) System Common Settings
### File `commonDeepDriveDefine.py`
You have to change this file according to the setup you are using, especially the IP address of the module you want to use and the fetaure you want to select as well as the car name (color of the car) if using GPS system.


## 6) Collect Training Sequences
### File `nn_collectTrainingData.py`
In order to have working Neural Network (NN), you need to record and drive yourself the car (arrow key). This creates a training set with image of this road labelized with steering angle.

Please note you may need to record different usecase associated to different NN such as :
- IDLE NN (dedicated to drive on simple road without intersection);
- RIGHT,LEFT and STRAIGHT NN (dedicated respectiveley to turn right, left and go straight at intersection).

So at least 4 NN would need to be trained to be able to drive on our road.

In order to simplify the record of those NN, the GPS position can give us the position of the intersection and then we associate the label for STRAIGHT NN during intersection, and IDLE otherwise. you can use it if your GPS is enable (`gps_main.py`) and you are under GPS selection. This allows to record image of 2 NN at once.

Also you can force the Label to LEFT/RIGHT/IDLE/STRAIGHT with `l`,`r`,`i`,`s` key. 
Once your record is done, you shoudl see under training_data/ directory the training set (`.npz` file).

Please note you record only when moving forward.


## 7) Data Augmentation 
### File `nn_dataAugmentation_main.py training_data/trainingSet.npz` 
Once the record is done, you may have to augment the training with shifted image. this is achieved by `nn_dataAugmentation_main.py`.


## 8) Training Neural Network
### File `nn_training.py`

This file will allow you to train the neural network. Please uncomment the supported neural network to train it in file `nn_training.py` such as :
``` 
#modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_RIGHT_TURN','CANNY','RIGHT_TURN'))
#modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_LEFT_TURN','CANNY','LEFT_TURN'))
modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_IDLE','CANNY','IDLE'))
modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_STRAIGHT','CANNY','STRAIGHT'))
```

As far as you have recorded STRAIGHT and IDLE usecases, you can do the above uncommented selection.
It is recommended to use CANNY filter selection for the beginning.

After the training this will create under `NNmodels/modelDirectory/` the trained neural network.

## 9) Check Results 
### File `showImages.py training_data/trainingSet.npz`

This script can be use to check your neural network and road line detector behavior using a given training data set.
You can go back and forth in the video with arrow key and see what result angle you predict with the model.
You can also read full video using 'a' key for automatic mode. It will read all image till end of the file.

Obviously you need 
- to uncomment the Neural network you want to use (same as in `nn_traing.py` script).

You can pass as parameter either NPZ, or MP4 or both
- showImages.py training_data/trainingSet.npz 
- showImages.py training_data/*.npz
- showImages.py RoadLaneDetection\video_in\*.mp4
- showImages.py training_data/*.npz RoadLaneDetection\video_in\*.mp4

## 10) Launch NN 
### File `nn_main.py`

Once you have selected and uncomment in this file the NN you want to use (same as `nn_training.py`), you can launch the module `nn_main.py` that will create a server for steering position.

Please note that if you may want to use GPS module (`gps_main.py`) to determine usecase for NN. 

## 11) Autonomous Driving
### File `car_main.py`

You can then use the `car_main.py` to connect to NN server to get steering estimate :
- `gps_main.py` if you want to have GPS features;
- `nn_main.py` if you want to have nn steering;
- `pathControl_main.py` if you wan tto have Steering estimate from path control (Slam for instance);
- `car_main.py` to control the car with all input from all module.



## 12) PathControl module
### File 'pathControl_main.py'

The path control module handle all what is related to path for the car.
The Steering server provide steeering angle based on graph slam estimate
the Command server provides to other module information such as IDLE, STRAIGHT, 
LEFT_TURN, RIGHT_TURN, STOP, FORWARD, CHANGE_LINE_R, CHANGE_LINE_L ... 
it is highly dependant on GPS, Object detection and all other module that could 
provide Land mark
Config flag :
pathControlSteeringEnable -> enable the server/client for Steering from path control
pathControlCommandEnable -> enable the server/client for command control
graphSlamEnable -> enable graphslam steering prediction



## 13) Roadline detection module
### File 'roadLine_main.py'
Road line detection process handles now all what is related by the Road line marker detection.
Mainly it provides a steering Angle server based on Road line detection
Config flag :
roadLineSteeringEnable -> enable the steering/client steering based on road line



## Misc and Notes
### Script Summary
- `commonDeepDriveDefine.py` -> all parameter need to configure your setup;
- `car_calibration.py` -> use to calibrate video of the car + make sure control of the car works;
- `comonDeepDriveTools.py` -> all the class and object that is used all over the scripts (such as commonThread, socket TCP server...);
- `gps_calibrate_camera.py` -> used to calibrate the GPS camera and have same setup all over the world;
- `gps_camera_checker.py` -> used to check the calibration of the GPS camera;
- `gps_camera_roi.py` -> used to create the scaling of the GPS camera (to switch into mm vs pixel);
- `gps_main.py` -> main task to launch for GPS. It takes in input the GPS image and provide through a server the GPS fix position of all vehicle in the list of color vehicles defined in `gpsFixThread.py`;
- `gpsClientThread.py` -> class to connect to GPS server;
- `gpsdata.py` -> table of GPS point for the road... Not sure it will be sued in the futur;
- `gpsFixThread.py` -> thread used to decode GPS image and create gps fix for each vehicle in the list;
- `graph_slam.py` -> functions used for PID and Slam ;
- `graphSlamThread.py` -> thread that control Slam and compute the Steering angle based on graphSlam;
- `KeyboardThread/py` -> thread used for keyboard control (commonly used);
- `showImages.py` -> used to see how behave NN estimate and Road Line Detection playing back the training set or video file;
- `nn_collectTrainingData_main.py` -> main script to collect data for NN;
- `nn_dataAugmentation_main.py` -> main script to augment the data with shifted sequences;
- `nn_evaluateModels_main.py` -> main script that could be used to evaluate model ;
- `nn_main.py` -> main script for Neural network control 5to be launched if you wan tto have a server providing steering angle based on NN estimate);
- `nn_training.py` -> main script to train neural network;
- `nnClientThread.py` -> class object to connect to NN server (used to get steering NN angle from server);
- `nncommonDefine.py` -> common define used for NN (such as NN flag to enable a given NN);
- `pathControl.py` -> main Script to launch to control path Cotnrol Module (provide through server  a steering angle based on GPS or Slam);
- `pathControlClientThread.py` -> class used to connect and get steering angle from pathControl server;
- `SensorThread.py` -> class client that connects to ultrasonic sensor serever;
- `SteerThread.py` -> class client that connects to steering server and push steering angle to the car;
- `VideoThread.py` -> class client that conects to a video stream (from car or GPS) and decode last image seen.


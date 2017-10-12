
# CAMERA CALIBRATION UTILITIES AND TOOLS FOR MAPPING

## GPS Camera Calibration 
To calibrate the camera (For example GPS camera for race circuit) do the following steps:
- The folder `gps_camera_calib_folder\` contains all Chess Boards Reference Images 
- Please name the image `calibration1.bmp` to `calibration20.bmp` at least


## FOR FOLLOWING TOOLS, MAKE SURE TO MODIFY IP ADDRESS IN THE SCRIPT TO PUT YOUR GPS RASPBERRY


### 1. get image of chess board
You can use in `gps_camera_calib_folder\` the tools `snapShotGpsImage.py` and 
the keyboard `s` to save the jpg image (it will indent automatically files for you)

Please follow the tuto [here](http://boofcv.org/index.php?title=Tutorial_Camera_Calibration) to take good pictures and good number of picture.



### 2. Compute Matrix and Distorsion Coefficient run the following command
`python gps_calibrate_camera.py -f .`

### 3. To determine the zone of interrest for the circuit run the following command:
`python gps_camera_roi.py -i undistort_gps_calibration.png`

### 4. To check the calibration camera parameters and circuit zone of interrest run the following checker:
`python gps_camera_checker.py -c calibrate_gps_camera.p -r undistort_gps_calibration.tpz`


At this tep the calibration is done. 


## MAPPING and CAR COLOR
Return in `computer/` directory and you can use now the `gps_main.py` process to record and see/capture color of the car.

Note:
- the tool `gps_main.py` requires [gnuplot installation](https://pypi.python.org/pypi/gnuplot-py);
- extract file and execute in the directory extracted:
`sudo python setup.py install` then `sudo apt-get install gnuplot`

### 1. color range of the car
Launch `gps_main.py` and move the mouse on the car to determine hsv color pointed by the mouse
When done you can report in `commonDeepDriveDefine` this color in the range according to the color if not done yet
(typically if the car is not detected , it is good to extend the range of this color with this new data).

### 2. create your own map
Launch `gps_main.py` and use only one color and car, wait for a fix.

#### First method : Point by point
- When the fix is done and you are OK with the car position, hit 'space' to record one point then move the car to another point (it is recommended to use `car_calibration.py` to move the car) and hit again space to record the second point, etc ...
- Leaving the program will save in `gps_plot.txt` all the point you recorded
- Create another file `gps_map.txt` with good input for gniplot
- Plot into `gps_map.png` picture of what you did

#### Second method : Map creation based on itinerary
- Use any car control (such as `car_calibration.py` to move the car) on the itinerary you want to follow
- Hit 'M' to start the Map creation. Every STEP_MAP_DISTANCE, a point will be recorded

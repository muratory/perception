import sys
import os


#IP mapping and definition
#Car IP 
# CAR_IP = '192.168.0.3'    # Server(Raspberry Pi) IP address
CAR_IP = '192.168.0.101'    # Server(Raspberry Pi) IP address

#POSITION_IP 
PATH_CONTROL_IP = 'localhost'   #IP for PC handling path control 

#NN server IP IP 
NN_IP = 'localhost'   #NN server IP for PC handling Neural Network

#NN server IP IP 
ROADLINE_IP = 'localhost'   #NN server IP for PC handling Neural Network

#GPS video server IP
GPS_VIDEO_IP = '192.168.0.81'    # Server(Raspberry Pi) IP address for GPS video

#GPS server IP
GPS_IP = '192.168.0.96'    # Server PC that get the GPS video stream and provide GPS fix

#Perception IP server 
PERCEPTION_IP = 'localhost'

#CAR color for GPS tracking
VEHICULE_NAME = 'RED'
#VEHICULE_NAME = 'GREEN'


#video connected on the car front (must be the same on the car )
PORT_VIDEO_CAR_SERVER = 8000
#video used for Neural network on the car (must be the same on the car )
PORT_VIDEO_NN_SERVER = 8000
#server port to push data for steering on the car (must be the same on the car)
PORT_STEER_SERVER = 8001
#server that provide sensor info (must be the same on the car)
PORT_SENSOR_SERVER = 8002
#server that provide path control steering estimate from Slam
PORT_PATH_CONTROL_STEERING_SERVER = 8003
#server that provide steering estimate from NN
PORT_NN_STEERING_SERVER = 8004
#server that provide GPS fix to all client
PORT_GPS_FIX_SERVER = 8005
#server which provide control car command
PORT_PATH_CONTROL_COMMAND_SERVER = 8006
#server which provide control car command
PORT_ROADLINE_STEERING_SERVER = 8007
#server which provide perception object 
PORT_PERCEPTION_SERVER = 8008

#Video server of the image of the road for gps estimate (must be the same on the GPS module)
PORT_VIDEO_GPS_SERVER = 8020

ADDR_VIDEO_NN_SERVER = (CAR_IP, PORT_VIDEO_NN_SERVER)
ADDR_VIDEO_CAR_SERVER = (CAR_IP, PORT_VIDEO_CAR_SERVER)
ADDR_VIDEO_GPS_SERVER = (GPS_VIDEO_IP, PORT_VIDEO_GPS_SERVER)
ADDR_STEER_SERVER = (CAR_IP, PORT_STEER_SERVER)
ADDR_SENSOR_SERVER = (CAR_IP, PORT_SENSOR_SERVER)
ADDR_GPS_FIX_SERVER = (GPS_IP,PORT_GPS_FIX_SERVER)
ADDR_PATH_CONTROL_STEERING_SERVER = (PATH_CONTROL_IP, PORT_PATH_CONTROL_STEERING_SERVER)
ADDR_NN_SERVER = (NN_IP,PORT_NN_STEERING_SERVER)
ADDR_ROADLINE_SERVER = (ROADLINE_IP,PORT_ROADLINE_STEERING_SERVER)
ADDR_PATH_CONTROL_COMMAND_SERVER = (PATH_CONTROL_IP,PORT_PATH_CONTROL_COMMAND_SERVER)
ADDR_PERCEPTION_SERVER = (PERCEPTION_IP,PORT_PERCEPTION_SERVER)

#client to enable for Neural Network client to get Video camera
videoClientEnable = True
#Client to enable for Car to display the car view
videoCarClientEnable = True
#client to enable to push steer command to the car
steerClientEnable = True
#Client to enable to get ultrasonic sensor distance
sensorClientEnable = False
#true if we have gps hardware available and stream connected
gpsEnable = False
#true if the path control Server/client need to be activated
pathControlSteeringEnable = False
#true if the path control command  server/client is enable.
pathControlCommandEnable = True
#true if we want to compute GraphSlam steering estimate
graphSlamEnable = False
#true if the neural network steering prediction (with client and server) is enable 
nnSteeringEnable = False
#true if the roadLine detection steering is enable 
roadLineSteeringEnable = False
#true if the roadLine detection steering is enable 
perceptionEnable = True




#set if you want to debug and see more image
showAllImage = False

#car speed in mm/s
INITIAL_CAR_SPEED = 500
INITIAL_CAR_SPEED_COMMAND = 50
MAX_CAR_SPEED = 1000
P_SPEED_REGULATION = 0.1
I_SPEED_REGULATION = 0.1
D_SPEED_REGULATION = 0.2

#max car speed in PWM unit
MAX_CAR_SPEED_COMMAND = 100


STOP_DISTANCE = 50


#Image size used
IMAGE_PIXELS_X = 320
IMAGE_PIXELS_Y = 240

# min max angle for the car
MIN_ANGLE    = -50
MAX_ANGLE    = 50

# STEP_CAPTURE is the angle step MIN_ANGLE to MAX_ANGLE from when hitting left/rigth key
STEP_CAPTURE = 1

# value of video FPS used to determine for instance runnin gaverage
VIDEO_FPS = 30
VIDEO_FPS_TIME = 0.033

#sampling of the image into vstack array (FPS record)
FPS_RECORD_TIME = 0.1



# sampling time for
STEERING_KEYBOARD_SAMPLING_TIME = 0.02

# this is the max delta angle allowed between prediction and keyboard control
MAX_KEYBOARD_DELTA_ANGLE_TO_PREDICTION = 15

#Steering sampling time for prediction. it used for prediction running average table and update steer when prediction is good
STEERING_PREDICTION_SAMPLING_TIME = 0.1

#number of sample for running average for NNs steer angle 
NB_SAMPLE_RUNNING_AVERAGE_PREDICTION = int(VIDEO_FPS * STEERING_PREDICTION_SAMPLING_TIME) + 1

#number of sample for running average for main NN (select the road use case)
NB_SAMPLE_RUNNING_AVERAGE_PREDICTION_MAIN_NN = 4

#sampling in s to send control command to all client
PATH_CONTROL_COMMAND_SAMPLING_TIME = 1

#sampling to compute the speed in s (we wait for this timing before to compute speed)
SPEED_SAMPLING_TIME = 0.5


######################## GPS related define ####################
GPS_FIX_BROADCAST_DELAY = 0.15

#estimated time in second for GPS to provide the fix. 
GPS_LAG_TIME = 0.0


""" COLOR DEFINITION """
COLOR = {
    'BLUE':
        [[110, 50, 50],
        [130, 255, 255]],
    'GREEN':
        [[70, 90, 20],
        [90, 255, 150]],
    'PINK':
        [[0, 100, 100],
        [10, 255, 255],
        [160, 100, 100],
        [180, 255, 255]],
    'RED':
        [[0, 120, 15],
        [10, 255, 167],
        [173,145,20],
        [185,255,165]],

    'ORANGE' :
        [[15,40,90],
        [27,255,255]],
    }


""" VEHICULE LIST THAT is returned by GPS Thread server"""
VEHICULE_LIST = [
            {'name':'RED','color1':'RED'},
            {'name':'GREEN','color1':'GREEN'},
            #{'name':'PINK','color1':'PINK'}
    ]


# Map parameters in mm which determines the center of the map and the circle where the intersection is
mapCenter_x    = 2520
mapCenter_Y    = 1496
straightRadius = 900



# Canny parameters
CANNY_TH1 = 100
CANNY_TH2 = 200



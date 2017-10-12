import cv2
import numpy as np
import math

import RoadLaneDetection.global_vars as rld_gb
import RoadLaneDetection.laneconfiguration as rld_cfg

# from RoadLaneDetection.line_fit_video import *
from commonDeepDriveDefine import *
from commonDeepDriveTools import *

# Color for left / rigth / middle
color_left  = (0,255,255)
color_right = (0, 0, 255)
color_mid   = (255, 0, 255)
color_wheel = (0, 255, 0)

# Line for left / right wheel of the car
LEFT_IDLE_X      =  40
RIGHT_IDLE_X     = 295
LEFT_IDLE_ANGLE  =  25
RIGHT_IDLE_ANGLE = -32
DELTA_X          =  10

# Line Detection global variables
# Store previous x bottom to select the nearest of the car
previous_x_right_bottom = None
previous_x_left_bottom  = None
counter_left  = 0
counter_right = 0

# Calibration Variable
calibration = False
offset_average         = []
angle_average          = []
x_right_bottom_average = []
x_left_bottom_average  = []
x_right_angle_average  = []
x_left_angle_average   = []

# Default Values for filters used in Line Detection Model
# Hough parameters
HOUGH_MIN_LINE_LENGHT = 10 # Minimum length of line. Line segments shorter than this are rejected
HOUGH_MAX_LINE_GAP    =  3 # Maximum allowed gap between line segments to treat them as single line.
HOUGH_THRESHOLD       = 18 # 26 # 11 # Minimum number of intersection to detect a line


# Values for Touran Configuration
# mask_top_left     = 0.45 #  55% of IMAGE_PIXELS_X
# mask_top_right    = 0.3  #  40% of IMAGE_PIXELS_X
# mask_bottom_left  = 1.5  # 155% of IMAGE_PIXELS_X
# mask_bottom_right = 1.3  # 130% of IMAGE_PIXELS_X
# mask_height       = 0.5
# crop_x_left      = 2

# Offset used to display text 
Y_TEXT_OFFSET = 12

class LineDetectionModel(object):   
    def __init__(self,name, pixel_x=IMAGE_PIXELS_X, pixel_y=IMAGE_PIXELS_Y):
        # Store name
        self.name = name

        # Default values for DeepDrive Video
        self.mask_top_left     = 0.55 #  55% of IMAGE_PIXELS_X
        self.mask_top_right    = 0.4  #  40% of IMAGE_PIXELS_X
        self.mask_bottom_left  = 1.55 # 155% of IMAGE_PIXELS_X
        self.mask_bottom_right = 1.35 # 130% of IMAGE_PIXELS_X
        self.mask_height       = 0.65 # 70% of IMAGE_PIXELS_Y
        self.crop_x_left       = 0

        # Hough parameters
        self.minLineLength = HOUGH_MIN_LINE_LENGHT
        self.maxLineGap    = HOUGH_MAX_LINE_GAP
        self.threshold     = HOUGH_THRESHOLD

        # Canny parameters
        self.threshold1 = CANNY_TH1
        self.threshold2 = CANNY_TH2

        # Coeff to convert angle to wheel direction
        self.convert_angle_coeff = 0.75

        self.delta_x = DELTA_X
        
        # Define varable to return latest angle computed in case no lines are detected.
        self.angle = 0

        # Initialize offset
        self.offset = 999

        # Define Region Of Interrest
        self.defineROI(pixel_x, pixel_y)

        # Line for left / right wheel of the car
        self.left_idle_angle  = LEFT_IDLE_ANGLE
        self.right_idle_angle = RIGHT_IDLE_ANGLE
        self.left_idle_x      = LEFT_IDLE_X
        self.right_idle_x     = RIGHT_IDLE_X

        # Init RoadLineDetector globals 
        rld_gb.init()

        if showAllImage == True:
            self.trackbars()

    """
    @function      : load ()
    @description   : Empty function to be generic with model list used
    @return        : None
    """    
    def load(self):
        pass

    """
    @function      : predict (sample)
    @description   : Return 
                        => Model Offset
                        => Model prediction for 1 sample
    @param samples : Sample to predict 
    @type  samples : array of 2D image shape
    @return        :  offset , prediction value
    @rtype         : float, int
    """    
    def predict(self, samples, roadLabel):
        global previous_x_right_bottom
        global previous_x_left_bottom
        global counter_left
        global counter_right
        global x_right_bottom_average
        global x_left_bottom_average
        global offset_average
        global x_right_angle_average
        global x_left_angle_average
        
        # Remove crop_x_left pixels on the left
        # Crop from crop_x_left, 0, image_pixel_y, image_pixel_x+crop_x_left 
        samples = samples[0:self.image_pixel_y, 
            self.crop_x_left:self.image_pixel_x+self.crop_x_left] 
        
        # Apply filter felected 
        canny_image = cv2.Canny(samples,self.threshold1,self.threshold2,apertureSize = 3)

        # apply the mask
        masked_image = cv2.bitwise_and(canny_image, self.mask)


        linesP = cv2.HoughLinesP(masked_image,1, np.pi/180,self.threshold, self.minLineLength, self.maxLineGap)

        # List used to store point for right/left lines
        x_right_top_list = []
        x_right_roi_list = []
        x_right_bottom_list = []
        x_left_top_list = []
        x_left_roi_list = []
        x_left_bottom_list = []
        lines_found = False
        x_inter = None
        self.offset = 999

        # Reset Calibration parameters 
        if calibration :
            if (len(offset_average) > 50):
                offset_average[:]         = []
                angle_average[:]          = []
                x_right_bottom_average[:] = []
                x_left_bottom_average[:]  = []
                x_right_angle_average[:]  = []
                x_left_angle_average[:]   = []

        # Ensure at least some lines were found
        if linesP is not None :
            for index in range(linesP.shape[0]):
                for x1,y1,x2,y2 in linesP[index]:
                    # Line : Y = a X + b   with a = (y1-y2)/(x1-x2)  and  b = y1 - a * x1
                    if y1 > self.Yroi and y2 > self.Yroi:
                        if showAllImage == True :
                            cv2.line(samples, (x1, y1), (x2,y2), (255, 255, 0),2)
                        if abs(x1 - x2) >0:
                            a = float(y1-y2)/float(x1-x2)
                            b = y1 - a * x1
                            # Search intersetion with image TOP and BOTTOM
                            # print "abs(a)=",abs(a),"\tx1=",x1,"\tx2=",x2
                            if abs(a) > 0.05:
                                lines_found = True
                                x_top    = -b/a
                                x_roi    = (self.Yroi - b)/a
                                x_bottom = (self.image_pixel_y - b)/a
                                # print "Xbottom=", x_bottom, "\t a=", a
                        else:
                            lines_found = True
                            # print "X1=X2=",x1,x2
                            x_top    = x1
                            x_roi    = x1
                            x_bottom = x1
                            a = 1

                        if lines_found == True:
                            # Right lines if x_bottom > (self.image_pixel_x/2)
                            if roadLabel == 'IDLE' or roadLabel == 'RIGHT_TURN' :
                                if x1>(self.image_pixel_x/2) and x2>(self.image_pixel_x/2) and x_bottom > (self.image_pixel_x/2) and (previous_x_right_bottom is None or abs(previous_x_right_bottom-x_bottom)<80 ):
                                    if (abs(a) > 0.2 and x_bottom > self.image_pixel_x) or (abs(a) > 0.5 and x_bottom < self.image_pixel_x):
                                        # print "RIGHT Line found","\ta=",a,"\tx1=",x1,"\ty1=",y1
                                        if showAllImage == True :
                                            cv2.line(samples, (x1, y1), (x2,y2), color_right,2)
                                        x_right_top_list.append(x_top)
                                        x_right_roi_list.append(x_roi)
                                        x_right_bottom_list.append(x_bottom)

                            # Left line if x_bottom < (self.image_pixel_x/2)
                            if roadLabel == 'IDLE' or roadLabel == 'STRAIGHT' or roadLabel == 'LEFT_TURN' :
                                if x1<(self.image_pixel_x*1.5/2) and x2<(self.image_pixel_x*1.5/2) and x_bottom > -200 and x_bottom < (self.image_pixel_x/2) and (previous_x_left_bottom is None or abs(previous_x_left_bottom-x_bottom)<100) :
                                    if abs(a) > 0.2 and abs(a) < 6:
                                        # print "LEFT Line found","\ta=",a,"\tx1=",x1,"\ty1=",y1    
                                        if showAllImage == True :
                                            cv2.line(samples, (x1, y1), (x2,y2), color_left,2)
                                        x_left_top_list.append(x_top)
                                        x_left_roi_list.append(x_roi)
                                        x_left_bottom_list.append(x_bottom)

            # If right line found, Draw line on the right
            if len(x_right_top_list) != 0:
                x_right_top    = np.median (np.array(x_right_top_list))
                x_right_roi    = np.median (np.array(x_right_roi_list))
                x_right_bottom = np.median (np.array(x_right_bottom_list))
                previous_x_right_bottom = x_right_bottom
                #print "RIGHT : Xbottom=", x_right_bottom, "\t a=", a
                cv2.line(samples, (int(x_right_roi), self.Yroi), (int(x_right_bottom),(self.image_pixel_y)), color_right,3)

                # Compute angle for right line
                # if x_right_top != x_right_bottom:
                angle_r = math.atan(float((x_right_roi - x_right_bottom) / (self.image_pixel_y - self.Yroi))) * 180 / np.pi

                cv2.putText(samples,"angle=" + str(int(angle_r)),(self.image_pixel_x-85 ,12+Y_TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_right,1)
                cv2.putText(samples,"x_bot=" + str(int(x_right_bottom)),(self.image_pixel_x-85,24+Y_TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_right,1)
                cv2.putText(samples,"x_roi=" + str(int(x_right_roi)),(self.image_pixel_x-85,36+Y_TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_right,1)
                counter_right = 0
                
                # Calibration
                if calibration:
                    x_right_bottom_average.append(x_right_bottom)
                    x_right_angle_average.append(angle_r)

            else:
                counter_right = counter_left + 1
                if counter_right == 10:
                    previous_x_right_bottom = None
                    counter_right = 0

            # If left line found, Draw line on the right
            if len(x_left_top_list) != 0:
                x_left_top    = np.median (np.array(x_left_top_list))
                x_left_roi    = np.median (np.array(x_left_roi_list))
                x_left_bottom = np.median (np.array(x_left_bottom_list))
                previous_x_left_bottom = x_left_bottom
                # print "LEFT : Xbottom=", x_left_bottom, "\t a=", a
                cv2.line(samples, (int(x_left_roi), self.Yroi), (int(x_left_bottom),(self.image_pixel_y)), color_left,3)
                
                # Compute angle for left line
                #if x_left_top != x_left_bottom:
                angle_l = math.atan(float((x_left_roi - x_left_bottom) / (self.image_pixel_y - self.Yroi))) * 180 / np.pi

                cv2.putText(samples,"angle=" + str(int(angle_l)),(1,12+Y_TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_left,1)
                cv2.putText(samples,"x_bot=" + str(int(x_left_bottom)),(1,24+Y_TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_left,1)
                cv2.putText(samples,"x_roi=" + str(int(x_left_roi)),(1,36+Y_TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_left,1)
                counter_left = 0
                
                # Calibration
                if calibration:
                    x_left_bottom_average.append(x_left_bottom)
                    x_left_angle_average.append(angle_l)

            else:
                counter_left = counter_left + 1
                if counter_left == 10:
                    previous_x_left_bottom  = None
                    counter_left = 0

            # if both line found, Draw pink line
            if len(x_right_top_list)!=0  and len(x_left_top_list) != 0:
                # Middle of 2 lines on top of picture
                x_roi = float(x_right_roi + x_left_roi)/2

                # Compute Angle
                self.angle = math.atan (float(( x_roi - (self.image_pixel_x/2 ) ) / (self.image_pixel_y - self.Yroi))) * 180 / np.pi
                self.angle = math.atan (float(( x_roi - (self.image_pixel_x/2  - self.delta_x )) / (self.image_pixel_y - self.Yroi))) * 180 / np.pi

                # Compute offset 
                self.offset = round((previous_x_right_bottom + previous_x_left_bottom) / 2 - (self.image_pixel_x/2) - self.delta_x,0)
                
                # Calibration
                if calibration:
                    offset_average.append(self.offset) 
                    angle_average.append(self.angle)

                    if (len(offset_average) == 10):
                        print ("Offset = ", int(sum(offset_average) / float(len(offset_average))),
                        "Angle = ", int(sum(angle_average) / float(len(angle_average))),
                        "X Left = ",int(sum(x_left_bottom_average) / float(len(x_left_bottom_average))),
                        "Angle Left = ",int(sum(x_left_angle_average) / float(len(x_left_angle_average))),
                        "X Right = ",int(sum(x_right_bottom_average) / float(len(x_right_bottom_average))),
                        "Angle Right = ",int(sum(x_right_angle_average) / float(len(x_right_angle_average))))

                
            # Only Left line found 
            elif len(x_left_top_list) != 0:
                # Check x_bottom and adjust angle based on x_bottom position for idle
                # 30 is the x bottom position of the left line in idle
                self.offset =  round(x_left_bottom - self.left_idle_x,0)
                angle_adjust  = math.atan ( float (self.offset / (self.image_pixel_y - self.Yroi)))  * 180 / np.pi
                # print "angle_adjust_left=",angle_adjust
                self.angle = angle_l - self.left_idle_angle + angle_adjust

            # Only Right line found 
            elif len(x_right_top_list)!=0:
                # Check x_bottom and adjust angle based on x_bottom position for idle
                # 280 is the x bottom position of the right line in idle
                self.offset = round(x_right_bottom - self.right_idle_x,0)
                angle_adjust  = math.atan ( float(self.offset / (self.image_pixel_y - self.Yroi))) * 180 / np.pi
                #print "angle_adjust_right=",angle_adjust
                self.angle = angle_r - self.right_idle_angle + angle_adjust

            # Draw line with current direction of the wheel
            # x_dir = (self.image_pixel_x/2) + (float(self.image_pixel_y - self.Yroi) * math.tan( self.angle/ self.convert_angle_coeff / 180.0 * np.pi))
            x_dir = (self.image_pixel_x/2) + (float(self.image_pixel_y - self.Yroi) * math.tan( self.angle / 180.0 * np.pi))
            cv2.line(samples, ((self.image_pixel_x/2), self.image_pixel_y), (int(x_dir), self.Yroi), color_mid,2)
            # cv2.line(samples,          ((self.image_pixel_x/2), self.image_pixel_y), (int(x_dir), self.Yroi), color_mid,2)
            cv2.putText(samples,"angle=" + str(round(self.angle,2)),(int(self.image_pixel_x/2)-40,24), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_mid,1)
            cv2.putText(samples,"x_dir=" + str(int(x_dir)),(int(self.image_pixel_x/2)-40,36), cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_mid,1)
            # print "Angle = ", str(round(self.angle,2))
            
        # Show images
        if showAllImage == True :
            cv2.drawContours(samples, self.roi_corners, -1, (255,255,255), 1)
            cv2.imshow("canny_image",canny_image)
            cv2.imshow("mask_image",masked_image)
            
        if samples.shape[0] > 2*IMAGE_PIXELS_Y:
            samples = cv2.pyrDown(samples);

        # Display the output of neural network in a nice way
        # self.showPrediction(self.angle)

        return (self.offset, int(self.convert_angle_coeff * self.angle))
        
    """
    @funtion          : computePredictionRate (test_set, labels_set)
    @description      : Predict response for full training set 
                        Store in self.weight the percentage of correct answer
    @param test_set   : Full Test set
    @type test_set    : Array of shape (m, NN_INPUT_LAYER)
    @param labels_set : Labels for test_set
    @type labels_set  : Array of shape (m, NN_OUTPUT_NUMBER)
    """
    def computePredictionRate(self, test_set, labels_set):      
        # Check Prediction all trained image_1D using for the current model
        print 'Testing...'
        mse=0
        for image,angle in zip(test_set,labels_set):
            w,p = self.predict(image)
            mse = mse + sqrt((p - angle)*(p - angle))

        mse = mse / test_set.shape[0]

        self.weight = ((MAX_ANGLE-MIN_ANGLE)-mse)*100/(MAX_ANGLE-MIN_ANGLE)

        cv2.destroyAllWindows()

    """
    @funtion         : showPrediction (predictionArray)
    @description     : Display Predicted responses in Gray Scale
    @param resp      : Predicted responses
    @type FileName   : Array of shape (1,NN_OUTPUT_NUMBER)
    """   
    def showPrediction(self, prediction):
        DNN_OUTPUT_NUMBER = 21
        imgDetectOutput = np.zeros((50,DNN_OUTPUT_NUMBER*50),dtype = np.uint8)
       
        # In order to compare value with DNN, convert angle to NN_OUTPUT_NUMBER
        output = int (round((((prediction + MAX_ANGLE) / ((MAX_ANGLE - MIN_ANGLE) / (DNN_OUTPUT_NUMBER-1) ))),0))
        imgDetectOutput[:,output*50:output*50+50] = 255
            
        cv2.imshow("LineDetectionModel_pred", imgDetectOutput)
        cv2.waitKey(1)
            

    """
    @function      : shapeRoadCaseImage(imgArray)
    @description   : shape the image according to NN input and
                     according to the use case 
    @param fileName: array of image or image alone (first shape to discriminate)
    @rtype         : None
    @return        : None
    """   
    def shapeRoadCaseImage(self,imgArray):
        if (imgArray.ndim > 2 ):
            return imgArray[:,0:self.image_pixel_y/2,:]
        else:
            return imgArray[0:self.image_pixel_y/2,:]
       

    """
    @function      : tuneParameter(param, value)
    @description   : Allow hough parameter tuning when picture is displayed
    @param param   : Wheel angle 
    @type  param   : string :  MAX_LINE_GAP, MIN_LINE_LENGHT, THRESHOLD
    @param value   : Value to set on the curent parameter
    @type value    : float
    @return        : None
    """   
    def tuneParameter(self, param, value):
        # Hough Filter Parmeters
        if param == 'MAX_LINE_GAP':
            self.maxLineGap = value
        elif param == 'MIN_LINE_LENGHT':
            self.minLineLength = value
        elif param == 'THRESHOLD':
            self.threshold = value
        # Convert Angle coeff
        elif param == 'COEFF_ANGLE':
            self.convert_angle_coeff = self.convert_angle_coeff + value
        # Canny Filter Parmeters
        elif param == 'CANNY_TH1':
            self.threshold1 = value
        elif param == 'CANNY_TH2':
            self.threshold2 = value
        else:
            print "tuneParameter -> BAD PARAMETER :", param
            
        print "tuneParameter - Set ",param," to ", value

    """
    Filter the image to include only yellow and white pixels
    """
    def filter_colors(self, image):
        # Filter white pixels
        white_threshold = 200
        lower_white = np.array([white_threshold, white_threshold, white_threshold])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image, lower_white, upper_white)
        white_image = cv2.bitwise_and(image, image, mask=white_mask)

        # Filter yellow pixels
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([89,95,95])
        upper_yellow = np.array([110,255,255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

        # Combine the two above images
        image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

        return image2

    """
    @function      : trackbars ()
    @description   : Create window with trackbars to tune Canny and Hough filters parameters
    @return        : None
    """    
    def trackbars(self):
        cv2.namedWindow('Trackbars',cv2.WINDOW_FREERATIO)
        cv2.resizeWindow('Trackbars',1000,300)
        cv2.createTrackbar('Canny_Th1','Trackbars',CANNY_TH1,500, self.tuneFilterParameters)
        cv2.createTrackbar('Canny_Th2','Trackbars',CANNY_TH2,500,self.tuneFilterParameters)
        cv2.createTrackbar('Hough_Gap','Trackbars',HOUGH_MAX_LINE_GAP,50, self.tuneFilterParameters)
        cv2.createTrackbar('Hough_Len','Trackbars',HOUGH_MIN_LINE_LENGHT,50, self.tuneFilterParameters)
        cv2.createTrackbar('Hough_Th','Trackbars',HOUGH_THRESHOLD,100, self.tuneFilterParameters)
        cv2.createTrackbar('Left_X','Trackbars',LEFT_IDLE_X,100, self.tuneFilterParameters)
        cv2.createTrackbar('Right_X','Trackbars',RIGHT_IDLE_X,320, self.tuneFilterParameters)
        cv2.createTrackbar('Left_An','Trackbars',LEFT_IDLE_ANGLE,50, self.tuneFilterParameters)
        cv2.createTrackbar('Right_An','Trackbars',-RIGHT_IDLE_ANGLE,50, self.tuneFilterParameters)
        cv2.createTrackbar('Coeff','Trackbars',int(self.convert_angle_coeff*100),100, self.tuneFilterParameters)

    """
    @function      : tuneFilterParameters ()
    @description   : Call back called when a trackbar is used
    @param value   : Trackbar value
    @return        : None
    """    
    def tuneFilterParameters(self,value):
        print "Value=",value
        # Canny Parameters
        if value == cv2.getTrackbarPos('Canny_Th1','Trackbars'):
            self.tuneParameter ('CANNY_TH1',value)
        if value == cv2.getTrackbarPos('Canny_Th2','Trackbars'):    
            self.tuneParameter ('CANNY_TH2', value)
            
        # HoughP parameters
        if value == cv2.getTrackbarPos('Hough_Gap','Trackbars'):
            self.tuneParameter ('MAX_LINE_GAP',value)
        if value == cv2.getTrackbarPos('Hough_Len','Trackbars'):    
            self.tuneParameter ('MIN_LINE_LENGHT', value)
        if value == cv2.getTrackbarPos('Hough_Th','Trackbars'):
            self.tuneParameter ('THRESHOLD',value)
            
        # Parameter to compute Angle
        if value == cv2.getTrackbarPos('Left_X','Trackbars'):
            self.left_idle_x = value
            print "left_idle_x set to :",self.left_idle_x
        if value == cv2.getTrackbarPos('Right_X','Trackbars'):
            self.right_idle_x = value
            print "right_idle_x set to :",self.right_idle_x
        if value == cv2.getTrackbarPos('Left_An','Trackbars'):
            self.left_idle_angle = value
            print "left_idle_angle set to :",self.left_idle_angle
        if value == cv2.getTrackbarPos('Right_An','Trackbars'):
            self.right_idle_angle = -value
            print "right_idle_angle set to :",self.right_idle_angle
       
        if value == cv2.getTrackbarPos('Coeff','Trackbars'):
            self.convert_angle_coeff = value/100.0
            print "convert_angle_coeff set to :",value, "and set to", self.convert_angle_coeff

    """
    @function      : defineROI ()
    @description   : Define Region Of Interrest to search lines
    @return        : None
    """    
    def defineROI(self,pixel_x=IMAGE_PIXELS_X, pixel_y=IMAGE_PIXELS_Y):   
        # Set image size and Ymin to draw lines 
        self.image_pixel_x = int(pixel_x - self.crop_x_left)
        self.image_pixel_y = int(pixel_y)
        self.Yroi = int(self.image_pixel_y * (1 - self.mask_height))
            
        samples = None

        # Create mask to apply to find lines
        self.mask = np.zeros((self.image_pixel_y,self.image_pixel_x), dtype=np.uint8)
        self.roi_corners = np.array([[(int(self.image_pixel_x * (1-self.mask_bottom_left) / 2) ,self.image_pixel_y), 
            (int(self.image_pixel_x * (1-self.mask_top_left) / 2),int (self.image_pixel_y * (1 - self.mask_height ))), 
            (int(self.image_pixel_x * (1+self.mask_top_right) / 2),int (self.image_pixel_y * (1 - self.mask_height ))),
            (int(self.image_pixel_x * (1+self.mask_bottom_right) / 2) ,self.image_pixel_y) ]], dtype=np.int32)
        
        ignore_mask_color = (255,)
        cv2.fillPoly(self.mask, self.roi_corners, ignore_mask_color)


    def initROIforVideo(self,pixel_x=IMAGE_PIXELS_X, pixel_y=IMAGE_PIXELS_Y):
        # Values for RoadLaneDetection Video
        self.mask_top_left     = 0.25
        self.mask_top_right    = 0.25
        self.mask_bottom_left  = 0.9
        self.mask_bottom_right = 0.9
        self.mask_height       = 0.35

        # Define Region Of Interrest
        self.defineROI(pixel_x, pixel_y)

import cv2
import numpy as np
import glob
import sys
import math
import time
import pygame
import os
import RoadLaneDetection.global_vars as rld_gb
import RoadLaneDetection.laneconfiguration as rld_cfg

from RoadLaneDetection.line_fit_video import annotate_image
from RoadLaneDetection.road_lane_detector import PID_process

from pygame.locals import *
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from nnCommonDefine import *

from NNmodels.DNNclassModel import DNNclassificationModel
from NNmodels.CnnVggRegressionModel import CnnVggRegressionModel
from NNmodels.DNNclassificationKerasModel import DnnClassificationKerasModel
from NNmodels.DNNclassificationKerasModelMainNN import DnnClassificationKerasModelMainNN
from NNmodels.DNNclassificationModelMainNN import DNNclassificationModelMainNN
from NNmodels.CnnAutoEncoder import CnnAutoEncoder
from NNmodels.NNRegressionModelWithAutoencoder import NNRegressionModelWithAutoencoder
from NNmodels.RNNRegressionModelWithAutoencoder import RNNRegressionModelWithAutoencoder
from roadLineDetectionModel import LineDetectionModel

def initRldGlobals(lineDetectionModel,width, height):
    # Init RoadLineDetector globals 
    rld_gb.init()

    if npzFile==True:
        # Toulouse specific setting
        # ROI + Filter Type settings (Toulouse)
        rld_gb.trap_bottom_width = 1
        rld_gb.trap_top_width = 0.4
        rld_gb.trap_height = 0.55
        rld_gb.trap_warped_ratio = rld_cfg.TRAP_WARPED_RATIO_CFG2
        rld_gb.combined_filter_type = rld_cfg.FILTER_GRAD_MAG
        rld_gb.mag_sobel_kernel_size = 3
                
        rld_gb.trackbar_enabled = True

        rld_gb.detected=False
        rld_gb.cold_boot = True
        rld_gb.detect_fast_mode_allowed = True
        rld_gb.detect_mode = rld_cfg.SLOW_MODE
        rld_gb.degraded_viz_mode_allowed = True
        rld_gb.degraded_viz_mode = False
        rld_gb.degraded_viz_count = 0
        rld_gb.degraded_viz_threshold = rld_cfg.DEGRAD_VIZ_THRESHOLD
        rld_gb.span_y_ratio_threshold = 1/6
        rld_gb.left_line.reset_fit()
        rld_gb.right_line.reset_fit()
    else:
        lineDetectionModel.initROIforVideo(width, height)


#define the max error to stop the next image 'n' key action
MAX_ERROR_TO_STOP = 15

#initializing variables
pygame.init()
screen=pygame.display.set_mode((200,200),0,24)
pygame.display.set_caption("Key Press Test")

# Default file type is NPZ
npzFile=True

# List of models
nnModelList = []

# Add below all model you want to show prediction and info
if DNNclassificationKerasModelIsEnable:
    #nnModelList.append(DnnClassificationKerasModelMainNN('DNNKerasClassification_CANNY_MAIN_NN','CANNY'))
    pass

if DNNclassificationModelIsEnable:
    #nnModelList.append(DNNclassificationModelMainNN('DNNclassificationModel_CANNY_MAIN_NN','CANNY'))
    pass

#create one neural network for each roadLabel
if DNNclassificationKerasModelIsEnable:
    #nnModelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_RIGHT_TURN','CANNY','RIGHT_TURN'))
    #nnModelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_LEFT_TURN','CANNY','LEFT_TURN'))
    # nnModelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_IDLE','CANNY','IDLE'))
    # nnModelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_STRAIGHT','CANNY','STRAIGHT'))
    pass

if CnnVggRegressionModelIsEnabled:
    #nnModelList.append(CnnVggRegressionModel('CnnVggRegressionModel_RIGHT_TURN','RIGHT_TURN'))
    #nnModelList.append(CnnVggRegressionModel('CnnVggRegressionModel_LEFT_TURN','LEFT_TURN'))
    #nnModelList.append(CnnVggRegressionModel('CnnVggRegressionModel_IDLE','IDLE'))
    #nnModelList.append(CnnVggRegressionModel('CnnVggRegressionModel_STRAIGHT','STRAIGHT'))
    pass

if CnnAutoEncoderIsEnabled:
    #nnModelList.append(CnnAutoEncoder('CnnAutoEncoder'))
    pass

if NNRegressionModelWithAutoEncoderIsEnabled:
    #nnModelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_IDLE', 'IDLE'))
    #nnModelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_STRAIGHT','STRAIGHT'))
    #nnModelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_LEFT', 'LEFT_TURN'))
    #nnModelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_RIGHT','RIGHT_TURN'))
    pass

# Evaluate all model in the list of model with data coming from the path name below
for model in nnModelList :
    model.load()

# Loop on all files passed as parameter
for files in sys.argv[1:]:
    # Get single file in case '*' is used                    
    training_data = glob.glob(files)
    for single_file in training_data:
        print "File : ", single_file

        # Retrieve file extension
        filename, file_extension = os.path.splitext(single_file)
        
        # Check if file is video training sequence (NPZ) or a video
        if file_extension == ".npz":
            npzFile=True
            with np.load(single_file) as data:
                print "Files in the archive : ", data.files
                image_array = data['train']
                steerAngle_array = data['steerAngle_label_array']
                roadLabel_Array = data['NNroadUseCase_label_array']
                
                # Get number of frame 
                nbFrames = len(image_array)
                print "Number of frames : " + str(nbFrames)
                
                # Get FRAME_WIDTH and FRAME_HEIGHT  of the frames in the NPZ file.
                height = image_array[0].shape[0]
                width  = image_array[0].shape[1]
                print "Set IMAGE_PIXELS_Y to ",image_array[0].shape[0]
        else:
            # Initialise stream
            npzFile=False
            stream = cv2.VideoCapture(str(single_file))
            if(stream.isOpened()):
                print "Stream is open on : ", str(single_file)

                # Get number of frame 
                nbFrames = stream.get(cv2.CAP_PROP_FRAME_COUNT)
                print "Number of frames : " + str(nbFrames)

                # Get FRAME_WIDTH and FRAME_HEIGHT  of the frames in the video stream.
                width  = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print "Image size : ",width," x ",height

        prevIndex   = 0
        index       = 1
        last_key    = 0
        check_event = True
        auto_mode   = False

        # Create Line Detection Model
        lineDetectionModel = LineDetectionModel('LineDetectionModel', width, height)
        initRldGlobals(lineDetectionModel,width, height)

        while check_event:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    # Handle image number
                    if event.key == K_UP:
                        last_key = K_UP
                        auto_mode = False
                    elif event.key == K_DOWN:
                        last_key = K_DOWN
                        auto_mode = False
                    # Modify coeff to apply 
                    elif event.key == K_a:
                        last_key = K_UP
                        if auto_mode == False and index == nbFrames-1 :
                            index = 0
                        auto_mode = True
                    elif event.key == K_ESCAPE or event.key == K_q or event.key == K_a :
                        check_event = False
                elif event.type == KEYUP:
                    key_input = pygame.key.get_pressed()
                    #when keyup, only test if all keys are UP
                    if ((key_input[pygame.K_UP] == 0) and
                        (key_input[pygame.K_DOWN] == 0)):
                            last_key = 0

            if last_key == K_UP or auto_mode==True:
                index = index + 1
                
            elif last_key == K_DOWN:
                index = index -1  

            if index <= 0:
                index = 0
            elif index >= nbFrames-1:
                index = nbFrames-1
                auto_mode = False
                                
            delta_predict = 0
            # make prediction and stop if there is big delta compared to True label
            if prevIndex != index:
                # print "Frame Index = ",index
                prevIndex = index
                
                if npzFile==True:
                    image       = image_array[index].copy()
                    roadLabel   = num2RoadLabel(roadLabel_Array[index])
                    steerAngle  = steerAngle_array[index]
                else :
                    # Set strem at index position
                    ret = stream.set(cv2.CAP_PROP_POS_FRAMES, index)
                    if not ret:
                        print ('stream.set(',index,') FAILED !!!')
                        break;
                    
                    # Get current position and
                    pos = stream.get(cv2.CAP_PROP_POS_FRAMES)
                    if pos != index:
                        print ('stream.get() differs from index :', pos, '/',index,') !!!')

                    # Read frame at index
                    ret, image  = stream.read()
                    if not ret:
                        print ('stream.read(',index,') FAILED !!!')

                    roadLabel   = 'IDLE'
                    steerAngle  = 'Unknown'

                if image is not None :
                    ################################################
                    # 1 - Call predict for all NN models activated #
                    ################################################
                    e1 = cv2.getTickCount()
                    for model in nnModelList: 
                        #make sure the use case Road label is the good one for the model used
                        if model.NNroadLabel == num2RoadLabel(roadLabel):
                            (w,p) = model.predict(image)
                            
                            if w == 0:
                                print 'zero Weight found at ',index,', prediction =',p,' label =', steerAngle,' for model =', model.name
                                #force stop video replay
                                last_key = 0
                                auto_mode = False
                                
                            if steerAngle != 'Unknown':
                                delta_predict = abs(p - steerAngle)
                                if delta_predict > MAX_ERROR_TO_STOP :
                                    #force stop video replay
                                    last_key = 0
                                    auto_mode = False
                                    print 'Big Error found at ',index,', value =',delta_predict,', prediction =',p,' label =', steerAngle,' for model =', model.name
                            
                    e2 = cv2.getTickCount()
                    t = (e2 - e1)/cv2.getTickFrequency()
                    # print ('NN Models prediction = ',t,'ms')

                    ##################################
                    # 2 - Call Line Detection Models #
                    ##################################
                    e1 = cv2.getTickCount()
                    if len(image.shape) < 3:
                        lineDetectionImage=cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
                    else:
                        lineDetectionImage=image.copy()

                    roadLineOffset, roadLineSteerCommand = lineDetectionModel.predict(lineDetectionImage,roadLabel)
                        
                    # Steer from PID based on offset
                    if roadLineOffset != 999 :
                            pidSteerCommand = PID_process (roadLineOffset)
                            print 'roadLineOffset=',roadLineOffset,'\troadLineSteerCommand = ',roadLineSteerCommand,'\tpidSteerCommand = ',pidSteerCommand

                    
                    e2 = cv2.getTickCount()
                    t = (e2 - e1)/cv2.getTickFrequency()
                    # print ('lineDetectionModel prediction = ',t,'ms')

                    ###################################################
                    # 3 - Call annotate_image to draw the drive space #
                    ###################################################            
                    # Convert image to RGB, format needed by annotate_image
                    if len(image.shape) < 3:
                        annotateImage=cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
                    else:
                        annotateImage=cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                            
                    e1 = cv2.getTickCount()
                    # annotateImage=annotate_image(annotateImage)
                    e2 = cv2.getTickCount()
                    t = (e2 - e1)/cv2.getTickFrequency()
                    # print ('annotate_image duration = ',t,'ms')
                    
                    annotateImage = cv2.cvtColor(annotateImage, cv2.COLOR_RGB2BGR)
                    
                    image = cv2.addWeighted(lineDetectionImage, 0.5, annotateImage, 0.5, 0)

                    ##################################################
                    # 4 - Display label, gps position and road label #
                    ##################################################
                    if npzFile==True:
                        # Add labels and steering angle on the image
                        cv2.putText(image, 'RoadUseCase = ' + str(roadLabel), (0,IMAGE_PIXELS_Y/2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1) 
                        cv2.putText(image, 'PredicTime  = ' + str(int(t*1000)), (0,IMAGE_PIXELS_Y/2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(image, 'SteerAngle  = ' + str(int(steerAngle)), (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

                        showLabel(steerAngle,'image', image)
                    cv2.imshow('image', image)
                    
        # Release the capture
        if npzFile==False:
            stream.release()

        # Release all windows
        cv2.destroyAllWindows()

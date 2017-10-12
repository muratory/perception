import cv2
import numpy as np
import glob
import sys
import math
import time
import pygame
from pygame.locals import *
from commonDeepDriveDefine import *
from NNmodels.Filter import *

#number of image to be added with a shift right or left (half each)
# 5 means 5 image right added and 5 image left added
imageAugmentation = 5

#number of Pixel to shift the image each time
PIXEL_SHIFT = 25

coefWheel = 4

training_data = glob.glob(str(sys.argv[1]))

 
for single_npz in training_data:
    print "File : ", single_npz
    with np.load(single_npz) as data:
        image_array = data['train']
        steerAngle_array = data['steerAngle_label_array']
        roadLabel_array = data['NNroadUseCase_label_array']

        print "Number of samples : " + str(len(image_array))

        indexImage =0
        #reset array
        new_image_array = np.zeros((1,IMAGE_PIXELS_Y, IMAGE_PIXELS_X), dtype=np.uint8)
        new_steerAngle_label_array = np.zeros(1, dtype=np.uint8)
        new_roadUseCase_label_array = np.zeros(1, dtype=np.uint8)
        
        for image,label,roadLabel in zip(image_array,steerAngle_array,roadLabel_array):


            for idx in range(-imageAugmentation,imageAugmentation+1):
                if idx == 0 :
                    #no change for index 0
                    continue
                 
                #compute new angle based on original angle
                new_angle = label + (idx*PIXEL_SHIFT*coefWheel)/10
                
                if new_angle > MAX_ANGLE :
                    new_angle = MAX_ANGLE
                elif new_angle < MIN_ANGLE :
                    new_angle = MIN_ANGLE
                
                #print 'old , new =',label[0],new_angle
                
                #cv2.imshow('newImage_canny', randomImage)

                if idx < 0:
                    #we want to shift right the image to handle left angle augmentation
                    new_image = image[ :,-idx*PIXEL_SHIFT : IMAGE_PIXELS_X]
                    #create the right missing part extending the last column 
                    extendedPart =np.array([image[:,IMAGE_PIXELS_X-1],]*(abs(idx*PIXEL_SHIFT))).transpose()
                    #concatenate extended image
                    new_image = np.concatenate((new_image,extendedPart),axis=1)                  
                else:
                    #we want to shift left the image to handle left angle augmentation
                    new_image = image[ :,0:IMAGE_PIXELS_X-idx*PIXEL_SHIFT]
                    #create the left missing part extending the last column 
                    extendedPart =np.array([image[:,0],]*(abs(idx*PIXEL_SHIFT))).transpose()
                    #concatenate extended image
                    new_image = np.concatenate((extendedPart,new_image),axis=1)
                    
                if showAllImage == True:
                    cv2.imshow('newImage', new_image)
                    cv2.waitKey(1)
                    
                #add this image to the new set
                temp_array = np.expand_dims(new_image, axis=0)
                
                new_image_array = np.vstack((new_image_array, temp_array))
                new_roadUseCase_label_array = np.vstack((new_roadUseCase_label_array, np.array([roadLabel])))
                new_steerAngle_label_array = np.vstack((new_steerAngle_label_array, np.array([new_angle])))
                
                
            indexImage = indexImage+1
            if indexImage%10 == 0 :
                print 'index image',indexImage
            
            #every 500 save the image into file to sped up the vstacking
            if new_image_array.shape[0] > 500 :
                # Build file name based on date/time
                timestr  = time.strftime("%Y%m%d-%H%M%S")
                fileName = 'training_data/trainingSetAugmented_' + timestr
                
                if len(new_image_array) > 1:
                    # save training images for NN one (main NN to select road case)
                    image_array = new_image_array[1:, :]
                    steerAngle_label_array = new_steerAngle_label_array[1:, :]
                    NNroadUseCase_label_array = new_roadUseCase_label_array[1:, :]
                    
                    # save training data as a numpy file
                    np.savez(fileName, train=image_array, steerAngle_label_array=steerAngle_label_array, NNroadUseCase_label_array=NNroadUseCase_label_array)
                 
                    print 'file saved in ', fileName,' with train shape', image_array.shape
                
                    #reset array
                    new_image_array = np.zeros((1,IMAGE_PIXELS_Y, IMAGE_PIXELS_X), dtype=np.uint8)
                    new_steerAngle_label_array = np.zeros(1, dtype=np.uint8)
                    new_roadUseCase_label_array = np.zeros(1, dtype=np.uint8)
                
                
        # Build file name based on date/time
        timestr  = time.strftime("%Y%m%d-%H%M%S")
        fileName = 'training_data/trainingSetAugmented_' + timestr
        
        if len(new_image_array) > 1:
            # save training images for NN one (main NN to select road case)
            image_array = new_image_array[1:, :]
            steerAngle_label_array = new_steerAngle_label_array[1:, :]
            NNroadUseCase_label_array = new_roadUseCase_label_array[1:, :]
            
            # save training data as a numpy file
            np.savez(fileName, train=image_array, steerAngle_label_array=steerAngle_label_array, NNroadUseCase_label_array=NNroadUseCase_label_array)
         
            print 'image shape       =', image_array.shape
            print 'steer angle shape =', steerAngle_label_array.shape
            print 'road label shape  =', NNroadUseCase_label_array.shape



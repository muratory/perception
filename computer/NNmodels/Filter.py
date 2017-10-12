import cv2
import numpy as np
from scipy.misc import imshow
from commonDeepDriveDefine import * 


class Filter (object):
    def __init__(self, type):
        self.type = type
        print 'Filter type created : ' , self.type

    def apply(self,image_array,len):
        for i in range(0,len):
            # Apply selected filter
            image_array[i,:,:] = self.applyOneSample(image_array[i,:,:])

        if showAllImage == True :
            for image in image_array:
                cv2.imshow(self.type + '_image', image)
                cv2.waitKey(1) & 0xFF     

        # Detroy Windows used to show prediction
        cv2.destroyAllWindows()

        return (image_array)

    def applyOneSample(self,image_array):
        # Apply selected filter
        if self.type == 'CANNY':
            image_array = self.applyCannyFilter(image_array)
        elif self.type == 'B_AND_W':
            image_array = self.applyBandWFilter(image_array)  
        
        return (image_array)

    def applyCannyFilter (self,image):
        return (cv2.Canny(image,CANNY_TH1,CANNY_TH2,apertureSize = 3))

    def applyBandWFilter (self,image):
        return (cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1])
        

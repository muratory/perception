'''
NN model dedicated to the selection of the road . 
output is how the car has to manage the road
right_turn, lef_turn or straigth (idle)
'''

import os

import cv2
import numpy as np

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from commonDeepDriveDefine import *

from NNmodels.DNNclassModel import DNNclassificationModel

# layers size of NN
# input image divide by 2 the total image on Y to take only the lowest part
NN_INPUT_LAYER = IMAGE_PIXELS_X * (IMAGE_PIXELS_Y / 2)
# output number is equal to 2 which is INTERSECTION or OTHER 
NN_OUTPUT_NUMBER = 2
NN_HIDDEN_LAYER = 32


class DNNclassificationModelMainNN(DNNclassificationModel):
    def __init__(self, name, filterType=None):
        DNNclassificationModel.__init__(self,name,filterType)

    """
    @function      : create ()
    @description   : Create Model 
    @param         : None 
    @return        : None
    """
    def create(self):
        self.numberOutput = NN_OUTPUT_NUMBER
        self.numberInput  = NN_INPUT_LAYER
        self.hiddenLayer  = NN_HIDDEN_LAYER
        self.imgNNoutput = np.zeros((50,self.numberOutput*50),dtype = np.uint8)
        self.stepReplay = (MAX_ANGLE - MIN_ANGLE) / (self.numberOutput - 1)
        # Create model
        layer_sizes = np.int32([self.numberInput, self.hiddenLayer, self.numberOutput])
        print 'build model ', self.name, ' layer size = ', layer_sizes
        self.model = cv2.ANN_MLP()
        self.model.create(layer_sizes)
        
        


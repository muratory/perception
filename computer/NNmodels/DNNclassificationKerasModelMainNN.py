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

from NNmodels.DNNclassificationKerasModel import DnnClassificationKerasModel

# layers size of NN
# input image divide by 2 the total image on Y to take only the lowest part
NN_INPUT_LAYER = IMAGE_PIXELS_X * (IMAGE_PIXELS_Y / 2)
# output number is equal to value that road use case can take : Right , left or cross
NN_OUTPUT_NUMBER = 2
NN_HIDDEN_LAYER = 32


class DnnClassificationKerasModelMainNN(DnnClassificationKerasModel):
    WITH_SCALER = True

    def __init__(self, name, filterType=None):
        DnnClassificationKerasModel.__init__(self,name,filterType)

    """
    @function      : create ()
    @description   : Create Model 
    @param         : None 
    @return        : None
    """
    def create(self):
        print 'create keras model', self.name
        self.numberOutput = NN_OUTPUT_NUMBER
        self.numberInput  = NN_INPUT_LAYER
        
        self.hiddenLayer  = NN_HIDDEN_LAYER
        self.imgNNoutput = np.zeros((50,self.numberOutput*50),dtype = np.uint8)
        
        
        self.model = Sequential()
        self.model.add(Dense(input_dim = self.numberInput, output_dim=self.hiddenLayer, init='uniform', activation='tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_dim=self.numberOutput, activation='softmax'))

        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        


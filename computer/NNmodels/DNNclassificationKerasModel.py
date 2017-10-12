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
# output number is equal to max number of wanted angle + 1 for the zero
NN_OUTPUT_NUMBER = 21
NN_HIDDEN_LAYER = 32

class DnnClassificationKerasModel(DNNclassificationModel):
    WITH_SCALER = True

    def __init__(self, name, filterType=None, roadUseCase = None):
        # call init
        DNNclassificationModel.__init__(self,name,filterType,roadUseCase)
        #change what could be specific to this model here (called after init)
        self.scaler_file = os.path.join(self.output_folder_base, self.name, 'scaler.pkl')
        self.model_file = os.path.join(self.output_folder_base, self.name, 'model.h5')

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
        self.stepReplay = (MAX_ANGLE - MIN_ANGLE) / (self.numberOutput - 1)
        
        self.model = Sequential()
        self.model.add(Dense(input_dim = self.numberInput, output_dim=self.hiddenLayer, init='uniform', activation='tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_dim=self.numberOutput, activation='softmax'))
        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        
    """
    @function      : train (fileName)
    @description   : Will train model using training set pass as parameters
                     70% of training set will be used to train the model
                     30% of training set will be used to test the model 
                     It will compute Mean Square Error and save it as weight for prediction 
    @param fileName: Name of the training set file name 
    @return        : None
    """    
    def train(self, trainingSetFileName):
        # Load data from NPZ file        
        (train, train_labels, indexCorrection) = self.loadTrainingSet(trainingSetFileName,self.NNroadLabel)

        eye_array = np.eye  (self.numberOutput, dtype=np.uint8)
        Y = np.zeros((1, self.numberOutput), dtype=np.uint8)
        
        # Apply filter selected 
        if (self.filter):
            train = self.filter.apply(train,len(train))     
        
        # get good shape image according to the use case 
        train = self.shapeRoadCaseImage(train)
        
        print 'after shaping',train.shape
        
        # Convert arrays as float32 and reshape as a 1D image to make X
        X = train.reshape(len(train),IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)

        #Create Y classification for MainNN (road label) and other NN (angles)
        if self.name.find('MAIN_NN') > 0:
            # Build training labels from 0 .. N  based on angle_array
            for roadLabel in train_labels:
                Y = np.vstack ((Y, eye_array[roadLabel]))
        else:    
            # Build training labels from 0 .. N  based on angle_array
            for angle in train_labels:
                angle = int (round((((angle + MAX_ANGLE) / self.stepReplay )),0))
                Y = np.vstack ((Y, eye_array[angle]))
                
        #remove first elem added by vstack and change in float
        Y = Y[1:,:]
        Y = np.asarray (Y, np.float32)

        # scale data on the whole set 
        if self.WITH_SCALER:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Split the training set into train + test set
        # here we want to keep the correction set intact so we only provide
        # in input the train set without correction set
        X_train, X_test, Y_train, Y_test = train_test_split(X[0:indexCorrection], Y[0:indexCorrection], test_size=0.3)
        print 'after split ', X_train.shape 
        
        #add now the correction datas to the train data set to make sure those correction data are used for model fit
        X_train = np.append(X_train,X[indexCorrection:],axis=0)
        Y_train = np.append(Y_train,Y[indexCorrection:],axis=0)
        
        print 'after concatenate ', X_train.shape

        # Train Model
        self.trainModel(X_train, Y_train)
        
        #evaluate performance
        trainScore = self.model.evaluate(X_train, Y_train, batch_size=32)
        print "train MSE = {}" , trainScore

        testScore = self.model.evaluate(X_test, Y_test, batch_size=32)
        # Evaluate the model performance on the training, cv and test sets
        print "test MSE  = {}" , testScore
        
        weight = (1-testScore[1]) * 100
        print "Weight applied for each prediction : ", weight
        
        # Save weight to apply to prediction in text file
        f = open (self.weightFileName, 'w')
        f.write(str(weight))
        f.close()

        # save model
        self.save()


        
    def trainModel(self, train, train_labels):
        # Set start time
        e1 = cv2.getTickCount()

        print 'Training Keras DNN ...'
        print train.shape
        print train_labels.shape
        
        # distribute all the class the same way
        numeric_Y = np.dot(train_labels, range(0,self.numberOutput))
        classes = np.unique(numeric_Y)
        class_weight = compute_class_weight("balanced", classes, numeric_Y)
        print 'class weigtht =', class_weight
        
        early_stopping = EarlyStopping(monitor='acc', patience=5, verbose=0, mode='auto')
        self.model.fit(train, train_labels,nb_epoch=100, batch_size=32,shuffle=True,class_weight=class_weight,callbacks=[early_stopping])    
        #self.model.fit(train, train_labels,nb_epoch=50, batch_size=32,shuffle=True,class_weight=class_weight)         


        # Set end time
        e2 = cv2.getTickCount()
        time = (e2 - e1)/cv2.getTickFrequency()
        print 'Training duration:', time

        
    def save(self):
        if self.model is not None:
            print 'save model in ',self.model_file
            self.model.save(self.model_file)
        if self.WITH_SCALER and self.scaler is not None:
            print 'save scaler in ',self.scaler_file
            joblib.dump(self.scaler, self.scaler_file)
            
    """
    @function      : load ()
    @description   : Load 
                        => model  from H5  file 
                        => Scaler from PKL file
                        => Weight from TXT file
    @param         : None 
    @return        : None
    """    
    def load(self):
        print 'load model',self.name
        if not os.path.exists(self.model_file):
            print 'MODEL HAS NOT BEEN TRAINED or model file not found in ',self.model_file
        self.model = load_model(self.model_file)
        if self.WITH_SCALER:
            self.scaler = joblib.load(self.scaler_file)

        # Load weight to apply to prediction
        if True == os.path.exists(self.weightFileName):
            f = open (self.weightFileName, 'r')
            self.weight = float(f.read())
            f.close()
        


            
    """
    @function      : predict (sample)
    @description   : Return 
                        => Model Weight
                        => Model prediction for 1 sample
    @param samples : Sample to predict 
    @type  samples : array of 2D image shape
    @return        : CnnVgg weight , prediction value
    @rtype         : float, int
    """    
    def predict(self, samples):
        # Apply filter selected 
        if (self.filter):
            samples = self.filter.applyOneSample(samples)

        # get good shape image according to the use case 
        samples = self.shapeRoadCaseImage(samples)

        if showAllImage == True:
            cv2.imshow('NNinput', samples)
            cv2.waitKey(1) & 0xFF 
        
        samples = samples.reshape(1,  IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)
            
        #scale X to the good range
        scaled_values = self.scaler.transform(samples)

        resp = self.model.predict(scaled_values)

        # Display the output of neural network in a nice way
        self.showPrediction(resp)
      
        #return max of the classification prob
        predict = resp.argmax(-1)

        # Display the output of neural network in a nice way
        self.showPrediction(resp)
        
        mean = np.sum(resp)/resp.shape[1]        
        
        #if the probability found is lower than 10 percent above the mean then we trash it
        if (resp[0,predict] < mean*1.10):
            print 'weak or bad prediction : mean, predic',mean,resp[0,predict]
            return (0, 0)
        else:
            if self.name.find('MAIN_NN') > 0:
                #main NN so we return the road use case detected
                return (self.weight, predict)
            else:
                #other NN so we return the angle
                predict = int((predict * self.stepReplay) - MAX_ANGLE)
                return (self.weight, predict)                  
        
    """
    @funtion         : showPrediction (predictionArray)
    @description     : Display Predicted responses in Gray Scale
    @param resp      : Predicted responses
    @type FileName   : Array of shape (1,NN_OUTPUT_NUMBER)
    """   
    def showPrediction(self, predictionArray):
        for value in range(0,len(predictionArray[0,:])):
            #print respNorm[0,value]*255
            if predictionArray[0,value] < 0:
                self.imgNNoutput[:,value*50:value*50+50] = 0
            else:
                self.imgNNoutput[:,value*50:value*50+50] = predictionArray[0,value] * 255
            
        cv2.imshow(self.name, self.imgNNoutput)
        cv2.waitKey(1)
        
        

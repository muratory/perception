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
from keras.optimizers import SGD
from keras.models import load_model

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from commonDeepDriveDefine import *
from nnCommonDefine import NNDeepDriveModel

DNN_OUTPUT_NUMBER = 21



class CnnVggRegressionModel(NNDeepDriveModel):

    WITH_SCALER = True
    WITH_BATCH_NORMALIZATION = False
    VGG_BLOCKS = 2
    CNN_FIRST_LAYER_FILTERS = 32
    DROPOUT_RATE = 0.

    def __init__(self, name,roadUseCase = None):
        # call init
        NNDeepDriveModel.__init__(self,name)
        print 'Create Model',self.name
        self.scaler_file = os.path.join(self.output_folder_base, self.name, 'scaler.pkl')
        self.model_file = os.path.join(self.output_folder_base, self.name, 'model.h5')
        self.weightFileName = os.path.join(self.output_folder_base, self.name, 'weight.txt')

        for fn in [self.scaler_file, self.model_file]:
            if not os.path.exists(os.path.dirname(fn)):
                os.makedirs(os.path.dirname(fn))
        self.NNroadLabel = roadUseCase
        
        self.create()
    """
    @function      : create ()
    @description   : Create Model
    @param         : None
    @return        : None
    """
    def create(self):
        # create the model
        if self.WITH_SCALER:
            self.scaler = StandardScaler()

        self.model = Sequential()

        for block in range(1, self.VGG_BLOCKS + 1):
            self.model.add(Convolution2D(
                self.CNN_FIRST_LAYER_FILTERS * block,
                3, 3,
                input_shape=(1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X),
                border_mode='same',
                activation='tanh',
                dim_ordering='th'
                ))
            if self.WITH_BATCH_NORMALIZATION:
                self.model.add(BatchNormalization(axis=1))
            self.model.add(Convolution2D(
                self.CNN_FIRST_LAYER_FILTERS * block,
                3, 3,
                border_mode='same',
                activation='tanh',
                dim_ordering='th'
                ))
            if self.WITH_BATCH_NORMALIZATION:
                self.model.add(BatchNormalization(axis=1))
            self.model.add(MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                dim_ordering='th'
                ))

        # reshape to a 1D vector
        self.model.add(Flatten())

        # Adding fully connected layers
        self.model.add(Dense(
            output_dim=100,
            activation='tanh',
        ))
        if self.WITH_BATCH_NORMALIZATION:
            self.model.add(BatchNormalization())
        if self.DROPOUT_RATE:
            self.model.add(Dropout(self.DROPOUT_RATE))
        self.model.add(Dense(
            output_dim=100,
            activation='tanh'
        ))
        if self.WITH_BATCH_NORMALIZATION:
            self.model.add(BatchNormalization())
        if self.DROPOUT_RATE:
            self.model.add(Dropout(self.DROPOUT_RATE))

        self.model.add(Dense(output_dim=1, activation='linear'))

        optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0., nesterov=False)
        self.model.compile(loss='mse', optimizer=optimizer)

    """
    @function      : save ()
    @description   : Save
                        => model  in H5  file
                        => Scaler in PKL file
    @param         : None
    @return        : None
    """
    def save(self):
        if self.model is not None:
            print 'save model in ', self.model_file
            self.model.save(self.model_file)
        if self.WITH_SCALER and self.scaler is not None:
            print 'save scaler in ', self.scaler_file
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
        print 'load model', self.name
        if not os.path.exists(self.model_file):
            print 'MODEL HAS NOT BEEN TRAINED or model file not found in ', self.model_file
        self.model = load_model(self.model_file)
        if self.WITH_SCALER:
            self.scaler = joblib.load(self.scaler_file)

        # Load weight to apply to prediction
        if os.path.exists(self.weightFileName):
            f = open(self.weightFileName, 'r')
            self.weight = float(f.read())
            f.close()

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
        (train, Y, indexCorrection) = self.loadTrainingSet(trainingSetFileName,self.NNroadLabel)

        # get good shape image according to the use case 
        train = self.shapeRoadCaseImage(train)

        # Convert arrays as float32 and reshape as a 1D image for scaler
        train = train.reshape(len(train), IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)
        Y = np.asarray(Y, np.float32)

        # scale data
        if self.WITH_SCALER:
            scaled = self.scaler.fit_transform(np.concatenate((train, Y), axis=1))
            Y = scaled[:, -1:]
            train = scaled[:, :-1]

        # Reshape the 1D pixel vector to a 2D image for cnn
        train = train.reshape((train.shape[0], 1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X))

        # Split the training set into two train + cross validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(train[0:indexCorrection], Y[0:indexCorrection], test_size=0.3)
        X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size=0.3)
        print 'train shape = ', X_train.shape

        # add correction frame to the train set
        X_train = np.append(X_train, train[indexCorrection:-1], axis=0)
        Y_train = np.append(Y_train, Y[indexCorrection:-1], axis=0)

        # Train Model
        self.trainModel(X_train, Y_train, X_cv, Y_cv)

        # Evaluate the model performance on the training, cv and test sets
        print "train MSE = {}".format(self.model.evaluate(X_train, Y_train, batch_size=32))
        print "cv MSE    = {}".format(self.model.evaluate(X_cv, Y_cv, batch_size=32))
        print "test MSE  = {}".format(self.model.evaluate(X_test, Y_test, batch_size=32))

        weight = (1 - self.model.evaluate(X_test, Y_test, batch_size=32)) * 100
        print "Weight applied for each prediction : ", weight

        # Save weight to apply to prediction in text file
        f = open(self.weightFileName, 'w')
        f.write(str(weight))
        f.close()

    """
    @function        : trainModel ( train, labels, cv_train, cv_labels)
    @description     : Will train model using training set pass as parameters
                       Will save the model once training is complete
    @param train     : Training set
    @rtype train     : Array of shape (m , NN_INPUT_LAYER)
    @param labels    : Labels for training set
    @rtype labels    : Array of shape (m , NN_OUTPUT_NUMBER)
    @param cv_train  : Cross Validation set
    @rtype cv_train  : Array of shape (0.3*m , NN_INPUT_LAYER)
    @param cv_labels : Labels for Cross Validation set
    @rtype cv_labels : Array of shape (0.3*m , NN_OUTPUT_NUMBER)
    @return          : None
    """
    def trainModel(self, X_train, Y_train, X_cv, Y_cv):
        # set start time
        e1 = cv2.getTickCount()

        print 'Training model ...'
        self.model.fit(
            X_train, Y_train,
            nb_epoch=60,
            batch_size=32,
            shuffle=True,
            validation_data=(X_cv, Y_cv),
            )

        # set end time
        e2 = cv2.getTickCount()
        time = (e2 - e1)/cv2.getTickFrequency()
        print 'Training duration:', time

        # save model
        self.save()

    """
    @function      : predict (sample)
    @description   : Return
                        => Model Weight
                        => Model prediction for 1 sample
    @param samples : Sample to predict
    @type  samples : array of shape (1, IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2)
    @return        : CnnVgg weight , prediction value
    @rtype         : float, int
    """
    def predict(self, samples):

        # get good shape image according to the use case 
        samples = self.shapeRoadCaseImage(samples)
        
        if showAllImage == True:
            cv2.imshow('NNinput', samples)
            cv2.waitKey(1) & 0xFF 
            
        # Convert arrays as float32 and reshape as a 1D image for scaler
        samples = samples.reshape(1, IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)

        
        # scale X
        scaled_values = self.X_scaling(samples, 1)
        # reshape to image for CNN input
        scaled_values = scaled_values.reshape(1, 1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X)
        predict = self.model.predict(scaled_values)
        predict = self.Y_inverse_scaling(predict, samples.shape[1])

        # Display the output of neural network in a nice way
        self.showPrediction(int(predict[0][0]))

        # predict.shape is (1, 1). Only returning the value
        return (self.weight, int(predict[0][0]))

    """
    @funtion       : X_scaling (x , y)
    @description   : Return scale values for X
    @param xvalues : the X values to scale with transform() method
    @type xvalues  : array of shape (m, n)
    @param ysize   : the number of Y outputs (columns)
    @type ysize    : int
    @return        : the X values scaled
    @rtype         : array of shape (m, n)
    """
    def X_scaling(self, xvalues, ysize):
        # add dummy y values
        values = np.concatenate((xvalues, np.zeros((xvalues.shape[0], ysize))), axis=1)
        scaled_values = self.scaler.transform(np.array(values))
        if len(xvalues) == 1:
            scaled_values = scaled_values.reshape(1, -1)
        # remove y scaled values
        return scaled_values[:, :-ysize]

    """
    @funtion       : Y_inverse_scaling (y , x)
    @description   : Return the inverse scale values for Y
    @param yvalues : the Y values to inverse scale with inverse_transform() method
    @type yvalues  : array of shape (m, n)
    @param xsize   : the number of X features (columns)
    @type xsize    : int
    @return        : the Y values inverse scaled
    @rtype         : array of shape (m, n)
    """
    def Y_inverse_scaling(self, yvalues, xsize):
        # add dummy X values
        values = np.concatenate((np.zeros((yvalues.shape[0], xsize)), yvalues), axis=1)
        inv_scaled_values = self.scaler.inverse_transform(values)
        return inv_scaled_values[:, xsize:]

    """
    @funtion         : showPrediction (fileName)
    @description     : Display Predicted responses in Gray Scale
    @param resp      : Predicted responses
    @type FileName   : Array of shape (1,NN_OUTPUT_NUMBER)
    """
    def showPrediction(self, prediction):
        imgNNoutput = np.zeros((50, DNN_OUTPUT_NUMBER*50), dtype=np.uint8)

        # In order to compare value with DNN, convert angle to NN_OUTPUT_NUMBER
        output = int(round((((prediction + MAX_ANGLE) /
                     ((MAX_ANGLE - MIN_ANGLE) / (DNN_OUTPUT_NUMBER - 1)))), 0))
        imgNNoutput[:, output * 50:output * 50 + 50] = 255

        cv2.imshow(self.name, imgNNoutput)
        cv2.waitKey(1)

    
 
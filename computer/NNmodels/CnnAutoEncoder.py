import os

import cv2
import numpy as np

from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.models import Model

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from commonDeepDriveDefine import *
from nnCommonDefine import NNDeepDriveModel

class CnnAutoEncoder(NNDeepDriveModel):

    WITH_SCALER = True
    VGG_BLOCKS = 3
    CNN_FIRST_LAYER_FILTERS = 32
    BATCH_SIZE = 32

    def __init__(self, name):
        # call init
        super(self.__class__, self).__init__(name)
        self.encoder = None
        self.scaler_file = os.path.join(self.output_folder_base, self.name, 'scaler.pkl')
        self.model_file = os.path.join(self.output_folder_base, self.name, 'model_weights.h5')
        self.encoder_file = os.path.join(self.output_folder_base, self.name, 'encoder_weights.h5')

        for fn in [self.scaler_file, self.model_file, self.encoder_file]:
            if not os.path.exists(os.path.dirname(fn)):
                os.makedirs(os.path.dirname(fn))

        self.create()

    """
    @function      : create ()
    @description   : Create Model
    @param         : None
    @return        : None
    """
    def create(self):
        # create the model
        print 'create CNN Auto Encoder', self.name
        if self.WITH_SCALER:
            self.scaler = StandardScaler()

        input_img = Input(shape=(1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X))

        x = input_img
        for block in range(self.VGG_BLOCKS):
            x = Convolution2D(
                self.CNN_FIRST_LAYER_FILTERS * pow(2, block),
                3, 3,
                activation='relu',
                border_mode='same',
                dim_ordering='th')(x)
            x = MaxPooling2D(
                pool_size=(2, 2),
                border_mode='same',
                dim_ordering='th')(x)

        encoded = x

        for block in range(self.VGG_BLOCKS - 1, -1, -1):
            x = Convolution2D(
                self.CNN_FIRST_LAYER_FILTERS * pow(2, block),
                3, 3,
                activation='relu',
                border_mode='same',
                dim_ordering='th')(x)
            x = UpSampling2D(
                size=(2, 2),
                dim_ordering='th')(x)

        decoded = Convolution2D(
            1,
            3, 3,
            activation='sigmoid',
            border_mode='same',
            dim_ordering='th')(x)

        self.encoder = Model(input=input_img, output=encoded)
        self.model = Model(input=input_img, output=decoded)
        self.model.summary()

        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

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
            self.model.save_weights(self.model_file)
        if self.encoder is not None:
            print 'save encoder in ', self.encoder_file
            self.encoder.save_weights(self.encoder_file)
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
        self.model.load_weights(self.model_file)
        self.encoder.load_weights(self.encoder_file)
        if self.WITH_SCALER:
            self.scaler = joblib.load(self.scaler_file)

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
        (train, _, indexCorrection) = self.loadTrainingSet(trainingSetFileName)

        # get good shape image according to the use case
        train = self.shapeRoadCaseImage(train)

        # Convert arrays as float32 and reshape as a 1D image for scaler
        train = train.reshape(len(train), IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)

        # scale data
        if self.WITH_SCALER:
            train = self.scaler.fit_transform(train)

        # Reshape the 1D pixel vector to a 2D image for cnn
        train = train.reshape((train.shape[0], 1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X))

        # Split the training set into two train + cross validation sets
        X_train, X_test = train_test_split(train[0:indexCorrection], test_size=0.3)
        X_train, X_cv = train_test_split(X_train, test_size=0.3)

        # add correction frame to the train set
        X_train = np.append(X_train, train[indexCorrection:-1], axis=0)

        # Train Model
        self.trainModel(X_train, X_cv)

        # Evaluate the model performance on the training, cv and test sets
        print "train MSE = {}".format(self.model.evaluate(X_train, X_train, batch_size=self.BATCH_SIZE))
        print "cv MSE    = {}".format(self.model.evaluate(X_cv, X_cv, batch_size=self.BATCH_SIZE))
        print "test MSE  = {}".format(self.model.evaluate(X_test, X_test, batch_size=self.BATCH_SIZE))

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
    def trainModel(self, X_train, X_cv):
        # set start time
        e1 = cv2.getTickCount()

        print 'Training model ...'
        self.model.fit(
            X_train, X_train,
            nb_epoch=20,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            validation_data=(X_cv, X_cv),
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
        # Convert arrays as float32 and reshape as a 1D image for scaler
        samples = samples.reshape(-1, IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)
        # scale X
        scaled_values = self.scaler.transform(samples)
        # reshape to image for CNN input
        scaled_values = scaled_values.reshape(-1, 1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X)
        predict = self.encoder.predict(scaled_values, batch_size=self.BATCH_SIZE)
        #show output of encoder to debug 
        if showAllImage == True:
            image = self.model.predict(scaled_values)
            image = image.reshape(-1, IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2)
            image = self.scaler.inverse_transform(image)
            for img, sample in zip(image,samples):
                cv2.imshow('encoder output', img.reshape(IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X).astype(np.uint8))
                cv2.imshow('sampleIn', sample.reshape(IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X).astype(np.uint8))
                cv2.waitKey(50) & 0xFF 
                
        return predict

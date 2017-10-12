import os

import cv2
import numpy as np

from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from commonDeepDriveDefine import *

from NNmodels.CnnAutoEncoder import CnnAutoEncoder
from nnCommonDefine import NNDeepDriveModel

# layers size of NN
NN_OUTPUT_NUMBER = 1
NN_HIDDEN_LAYER = 100
DNN_OUTPUT_NUMBER = 21

# temporal parameters for constructing the sequence of images
# number of samples to take for time sequence in RNN
SEQUENCE_LENGTH = 3
# time choosen between two sample for RNN
TIME_INTERVAL = 0.3
# interval in sample for RNN training (should be equal to 3)
SEQUENCE_INTERVAL_TRAIN = int(round(TIME_INTERVAL / FPS_RECORD_TIME))
# interval in sample for prediction  (should be around 9)
SEQUENCE_INTERVAL_PREDICT = int(round(TIME_INTERVAL * VIDEO_FPS))


class RNNRegressionModelWithAutoencoder(NNDeepDriveModel):

    WITH_SCALER = True
    DROPOUT_RATE = 0.35
    DROPOUT_W = 0.
    DROPOUT_U = 0.3

    def __init__(self, name, roadUseCase=None):
        # call init
        NNDeepDriveModel.__init__(self, name)
        # change what could be specific to this model here (called after init)
        self.scaler_file = os.path.join(self.output_folder_base, self.name, 'scaler.pkl')
        self.model_file = os.path.join(self.output_folder_base, self.name, 'model_weights.h5')
        self.weightFileName = os.path.join(self.output_folder_base, self.name, 'weight.txt')
        self.encoder = None
        self.NNroadLabel = roadUseCase

        for fn in [self.scaler_file, self.model_file, self.weightFileName]:
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
        if self.WITH_SCALER:
            self.scaler = StandardScaler()

        # create the model
        print 'create NN with encoder', self.name
        self.numberOutput = NN_OUTPUT_NUMBER
        self.hiddenLayer = NN_HIDDEN_LAYER
        self.imgNNoutput = np.zeros((50, self.numberOutput*50), dtype=np.uint8)

        self.model = Sequential()

        self.auto_encoder = CnnAutoEncoder('CnnAutoEncoder')
        self.auto_encoder.load()
        self.encoder = self.auto_encoder.encoder

        encoder_output_shape = self.encoder.layers[-1].output_shape
        # encoder_output_shape is expected to be (samples, filters, xsize, ysize)
        # then flatten by the temporal transformation
        self.input_dim = np.prod(encoder_output_shape[1:])

        self.model.add(LSTM(input_shape=(SEQUENCE_LENGTH, self.input_dim),
                            output_dim=self.hiddenLayer, activation='tanh',
                            dropout_W=self.DROPOUT_W, dropout_U=self.DROPOUT_U,
                            return_sequences=True))
        self.model.add(Dropout(self.DROPOUT_RATE))
        self.model.add(LSTM(output_dim=self.hiddenLayer, activation='tanh',
                            dropout_W=self.DROPOUT_W, dropout_U=self.DROPOUT_U))
        self.model.add(Dropout(self.DROPOUT_RATE))

        self.model.add(Dense(output_dim=self.numberOutput, activation='linear'))
        optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0., nesterov=False)
        self.model.summary()
        self.model.compile(
            loss='mse',
            optimizer=optimizer)

        # by default we use interval for prediction (that is the interval of video FPS)
        self.sequenceInterval = SEQUENCE_INTERVAL_PREDICT

        # saving area for autoencoder output. used to keep the sequence samples for RNN
        # BUFFER length for samples (should be around 19)
        buffer_samples_prediction = (SEQUENCE_LENGTH - 1) * self.sequenceInterval + 1
        self.predictionSamplesArray = np.zeros(
            ((buffer_samples_prediction,) + encoder_output_shape[1:]), dtype=np.uint8)

    def convertToTemporalData(self, samples, Y=None):
        result = []
        for index in range(len(samples) - (SEQUENCE_LENGTH - 1) * self.sequenceInterval):
            sequence = []
            for sequence_index in range(SEQUENCE_LENGTH):
                sequence.append(samples[index + sequence_index * self.sequenceInterval])
            result.append(sequence)
        result = np.array(result)
        if Y is not None:
            Y = Y[(SEQUENCE_LENGTH - 1) * self.sequenceInterval:]
        return result, Y

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
        # we use interval from the recorded frame
        self.sequenceInterval = SEQUENCE_INTERVAL_TRAIN
        # Load data from NPZ file
        (train, Y, indexCorrection) = self.loadTrainingSet(trainingSetFileName, self.NNroadLabel)
        print "indexCorrection", indexCorrection

        # get good shape image according to the use case
        train = self.shapeRoadCaseImage(train)

        # Reshape for Cnn use
        train = train.reshape((-1, 1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X)).astype(np.float32)
        Y = np.asarray(Y, np.float32)

        # scale data
        if self.WITH_SCALER:
            # only scale Y. X is already scaled by the autoencoder
            Y = self.scaler.fit_transform(Y)

        # autoencode image
        train = self.auto_encoder.predict(train)

        train, Y = self.convertToTemporalData(train, Y)

        train = train.reshape((-1, SEQUENCE_LENGTH, self.input_dim))

        # Split the training set into two train + cross validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            train[0:indexCorrection], Y[0:indexCorrection], test_size=0.3)
        X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size=0.3)

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
        with open(self.weightFileName, 'w') as fd:
            fd.write(str(weight))

    """
    @function        : trainModel ( train, labels, cv_train, cv_labels)
    @description     : Will train model using training set pass as parameters
                       Will save the model once training is complete
    @param X_train     : Training set
    @type X_train     : Array of shape (m , n)
    @param X_cv  : Cross Validation set
    @type X_cv  : Array of shape (0.3*m , n)
    @return          : None
    """
    def trainModel(self, X_train, Y_train, X_cv, Y_cv):
        # set start time
        e1 = cv2.getTickCount()

        print 'Training model ...'
        self.model.fit(
            X_train, Y_train,
            nb_epoch=40,
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

    def save(self):
        if self.model is not None:
            print 'save model in ', self.model_file
            self.model.save_weights(self.model_file)
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
        if self.WITH_SCALER:
            self.scaler = joblib.load(self.scaler_file)

        # Load weight to apply to prediction
        if os.path.exists(self.weightFileName):
            f = open(self.weightFileName, 'r')
            self.weight = float(f.read())
            f.close()

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

        samples = samples.reshape(1, 1, IMAGE_PIXELS_Y / 2, IMAGE_PIXELS_X)

        predict = self.auto_encoder.predict(samples)

        # shift the buffer of sample to let the place for new sample at the end :
        self.predictionSamplesArray = np.roll(self.predictionSamplesArray, -1, axis=0)
        # copy the sample (output of autoencoder) in the free last position
        self.predictionSamplesArray[-1] = np.copy(predict)

        # create the SEQUENCE_LENGTH samples required for RNN
        samples, _ = self.convertToTemporalData(self.predictionSamplesArray)

        samples = samples.reshape((-1, SEQUENCE_LENGTH, self.input_dim))

        predict = self.model.predict(samples)

        predict = self.Y_inverse_scaling(predict, samples.shape[1])

        # Display the output of neural network in a nice way
        self.showPrediction(int(predict[0][0]))

        # predict.shape is (1, 1). Only returning the value
        return (self.weight, int(predict[0][0]))


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
    def Y_inverse_scaling(self, yvalues, _xsize):
        return self.scaler.inverse_transform(yvalues.reshape(-1, 1))

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


    """
    @funtion         : evaluate (fileName)
    @description     : Compute Mean Square Error for full training set
    @param fileName  : List of NPZ file name to evaluate
    @type fileName   : String
    """
    def evaluate(self, fileName):
        # we use interval from the recorded frame
        self.sequenceInterval = SEQUENCE_INTERVAL_TRAIN
        # saving area for autoencoder output. used to keep the sequence samples for RNN
        buffer_samples_prediction = (SEQUENCE_LENGTH - 1) * self.sequenceInterval + 1
        self.predictionSamplesArray = np.zeros(
            ((buffer_samples_prediction,) + self.encoder.layers[-1].output_shape[1:]),
            dtype=np.uint8)
        super(self.__class__, self).evaluate(fileName)

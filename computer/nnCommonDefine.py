import sys
import os
import numpy as np
import glob
import cv2
from math import sqrt,tan

from commonDeepDriveDefine import *
from commonDeepDriveTools import *

# insert NN model path in the path
cwd = os.getcwd()
sys.path.insert(0, cwd + '\\NNmodels')


#neural network used for this work:
DNNclassificationKerasModelIsEnable = True
DNNclassificationModelIsEnable = True
CnnVggRegressionModelIsEnabled = True
CnnAutoEncoderIsEnabled = True
NNRegressionModelWithAutoEncoderIsEnabled = True
RNNRegressionModelWithAutoEncoderIsEnabled = True


class NNDeepDriveModel(object):
    ''' define neural network base class
    '''

    WITH_SCALER = False

    def __init__(self, name):
        self.name = name
        self.scaler = None
        self.model = None
        self.weight = 0
        self.output_folder_base = os.path.join("NNmodels", "modelDirectory")

    def save(self):
        """ save the model to a file """
        pass

    def load(self):
        """ load a model from a file and assign it to self.model """
        pass

    def train(self, trainingSetFileName):
        """ train the model from trainingSetFileName """
        pass

    def predict(self, data):
        """ predict model output from data """
        pass

    """
    @funtion         : evaluate (fileName)
    @description     : Compute Mean Square Error for full training set
    @param fileName  : List of NPZ file name to evaluate
    @type fileName   : String
    """
    def evaluate(self, fileName):
        # Load data from NPZ file        
        (train, Y, index) = self.loadTrainingSet(fileName,self.NNroadLabel)
        predictions = np.zeros(len(train), dtype=np.int)
        mse=0
        
        # Load model using XML file
        self.load()
        #start timer to measure the time of the predictions
        e1 = cv2.getTickCount()
        for image,label,idx in zip(train,Y,range(len(train))):
            w,p = self.predict(image)
            predictions[idx] = p
            mse = mse + sqrt((p - label)*(p - label))
        e2 = cv2.getTickCount()
        t = ((e2 - e1)/cv2.getTickFrequency())/len(train)

        mse = mse / train.shape[0]   

        #remove first vstack element and reshape for display         
        #predictions = predictions.reshape(1,len(predictions))
        true_labels=Y.reshape(1,len(Y))

        print 'Prediction:', predictions
        print 'True labels:', true_labels
        print 'delta',predictions-true_labels
        
              
        print 'MSE = ', mse
        print 'percentage weight of this evaluation =', ((MAX_ANGLE-MIN_ANGLE)-mse)*100/(MAX_ANGLE-MIN_ANGLE)
        print 'percentage weight stored =', self.weight
        print 'average prediction time(ms) =',t*1000


                                
    """
    @function      : loadTrainingSet (fileName)
    @description   : Load training set from file name 
    @param fileName: Name of the training set file name 
    @param NNroadLabel  : Which use case this NN is for. we return only this set
    @rtype         : String
    @return        : whole image trainset from filename(s)
    @return        : whole angle trainset from filename(s)
    @return        : index in trainset where is located the corrective set (filename with CorrectionSet)
    """
    def loadTrainingSet(self, fileName, NNroadLabel = None):
        print 'Loading training data ...',NNroadLabel
        # load training data
        training_datas = glob.glob(fileName)
        train_datas = []
        Y_datas = []
        train_datas_correction = []
        Y_datas_correction = []
        for single_npz in training_datas:
            with np.load(single_npz) as data:
                print 'file = ',single_npz
                if single_npz.find('GPS')>0:
                    gpsDataFile = True
                else: 
                    gpsDataFile = False
                #get datas
                train = data['train']
                steerLabel = data['steerAngle_label_array']
                roadLabel = data['NNroadUseCase_label_array']
                
                #filter out data according to the road use case
                if NNroadLabel != None:
                    #keep only the element in the array that contains the correponding road use case
                    train=train[roadLabel[:,0]==NNroadLabel2Num(NNroadLabel)]
                    train_label=steerLabel[roadLabel[:,0]==NNroadLabel2Num(NNroadLabel)]
                else:
                    #all element are good if there is no filter requested by Road label
                    MainRoadLabel_array = np.zeros(1, dtype=np.uint8)
                    for label in roadLabel:
                        label = num2RoadLabel(label)
                        #1 for intersection case and 0 for no interesection
                        if label == 'RIGHT_TURN' or label == 'LEFT_TURN' or label == 'STRAIGHT':
                            MainRoadLabel_array = np.vstack((MainRoadLabel_array, np.array([1])))
                        else:
                            MainRoadLabel_array = np.vstack((MainRoadLabel_array, np.array([0])))

                    #remove first element 
                    train_label = MainRoadLabel_array[1:,:]
                    
                #look for correction Data that must be added anyway to train set               
                if single_npz.find('CorrectionSet')>=0:
                    #correction set
                    train_datas_correction.append(train)
                    Y_datas_correction.append(train_label)
                else: 
                    #regular train set 
                    train_datas.append(train)
                    Y_datas.append(train_label)
                
                print 'Shape image = ',data['train'].shape
        #concatenate all regular train set
        train = np.concatenate(tuple(train_datas))
        Y = np.concatenate(tuple(Y_datas))

        print 'Shape for regular trainSet'
        print train.shape
        print Y.shape

        #record the index of the regular train set
        index = train.shape[0]
        
        #concatenate now correction set
        if len(train_datas_correction)>0:
            train = np.append(train,np.concatenate(tuple(train_datas_correction)),axis=0)
            Y = np.append(Y,np.concatenate(tuple(Y_datas_correction)),axis=0)
            print 'total shape for Data to be trained'
            print train.shape
            print Y.shape

        return (train, Y, index)


    """
    @function      : shapeRoadCaseImage(imgArray)
    @description   : shape the image for Main NN 
    @param fileName: array of image or image alone
    @rtype         : image array or iamage
    @return        : None
    """         
    def shapeRoadCaseImage(self,imgArray):
        if  self.name.find('MAIN') > 1 :
            #MAin NN detected .. .choose the middle image
            if (imgArray.ndim > 2 ):
                return imgArray[:,IMAGE_PIXELS_Y/4:IMAGE_PIXELS_Y*3/4,:]
            else:
                return imgArray[IMAGE_PIXELS_Y/4:IMAGE_PIXELS_Y*3/4,:]
                
        else:
            #for other NN , just keep lower image
            if (imgArray.ndim > 2 ):
                return imgArray[:,IMAGE_PIXELS_Y/2:IMAGE_PIXELS_Y,:]
            else:
                return imgArray[IMAGE_PIXELS_Y/2:IMAGE_PIXELS_Y,:]
                
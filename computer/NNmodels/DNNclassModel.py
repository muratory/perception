import cv2
import numpy as np


#import matplotlib.pyplot as plt

from commonDeepDriveDefine import *
from nnCommonDefine import NNDeepDriveModel
from Filter import *
from sklearn.model_selection import train_test_split

# layers size of NN
# input image divide by 2 the total image on Y to take only the lowest part
NN_INPUT_LAYER = IMAGE_PIXELS_X * (IMAGE_PIXELS_Y / 2)
# output number is equal to max number of wanted angle + 1 for the zero
NN_OUTPUT_NUMBER = 21
NN_HIDDEN_LAYER = 32



class DNNclassificationModel(NNDeepDriveModel):
    def __init__(self,name, filterType=None, NNroadLabel = None):
        # Call init    
        NNDeepDriveModel.__init__(self, name)
        self.weight       = 0
        self.NNroadLabel = NNroadLabel
        self.weightFileName = os.path.join(self.output_folder_base, self.name, 'weight.txt')
        self.model_file = os.path.join(self.output_folder_base, self.name, 'model.xml')
        self.filter = Filter(filterType)
        
        if not os.path.exists(os.path.dirname(self.model_file)):
            os.makedirs(os.path.dirname(self.model_file))  
        
        #call create model
        self.create()
            
            

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
       

        
    """
    @function      : train (fileName)
    @description   : Will train model using training set pass as parameters
                     70% of training set will be used to train the model
                     30% of training set will be used to test the model 
                     It will compute Mean Square Error and save it as weight for prediction 
    @param fileName: Name of the training set file name 
    @rtype fileName: String
    @return        : None
    """    
    def train(self,trainingSetFileName):
    
         # Load data from NPZ file        
        (train, train_labels, indexCorrection) = self.loadTrainingSet(trainingSetFileName,self.NNroadLabel)

        #prepare Y to be a classigfication output
        eye_array = np.eye(self.numberOutput, dtype=np.uint8)
        Y = np.zeros((1, self.numberOutput), dtype=np.uint8)

        train_set, test_set, train_labels_set, test_labels_set = train_test_split(train, train_labels, test_size=0.3)

        print 'Shape before adding correction =',train_set.shape

        #add to Trainset the correction set
        train_set = np.append(train_set,train[indexCorrection:],axis=0)
        train_labels_set = np.append(train_labels_set,train_labels[indexCorrection:],axis=0)

        print 'Shape after adding correction =',train_set.shape
        
        # Apply filter felected 
        if (self.filter):
            X = self.filter.apply(train_set,len(train_set))
        
        
        # get good shape image according to the use case 
        X = self.shapeRoadCaseImage(X)

        # Convert arrays as float32 and reshape as a 1D image to make X
        X = X.reshape(len(X),IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)
        
        #Create Y classification for MainNN (road label) and other NN (angles)
        if self.name.find('MAIN_NN') > 0:
            # Build training labels from 0 .. N  based on angle_array
            for roadLabel in train_labels_set:
                Y = np.vstack ((Y, eye_array[roadLabel]))
        else:    
            # Build training labels from 0 .. N  based on angle_array
            for angle in train_labels_set:
                angle = int (round((((angle + MAX_ANGLE) / self.stepReplay )),0))
                Y = np.vstack ((Y, eye_array[angle]))       

        #remove first elem added by vstack and change in float
        Y = Y[1:,:]
        Y = np.asarray (Y, np.float32)

         # Train the model 
        self.trainModel(X, Y)

        # Evaluate Model and check prediction rate
        self.computePredictionRate (test_set, test_labels_set)
        print 'weight for test set: %f:' % (self.weight)

        # Save weight based on MSE 
        f = open (self.weightFileName, 'w')
        f.write(str(self.weight))
        f.close()  

    """
    @function           : trainModel (fileName)
    @description        : Will train model using training set pass as parameters
                          Will save the model once training is complete 
    @param train        : Training set
    @rtype train        : Array of shape (m , NN_INPUT_LAYER)
    @param train_labels : Labels for training set
    @rtype train_labels : Array of shape (m , NN_OUTPUT_NUMBER)    
    @return             : None
    """    
    def trainModel(self, train, train_labels):
        # Set start time
        e1 = cv2.getTickCount()

        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
        params = dict(term_crit = criteria,
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )

        print 'Training MLP ...'
        print train.shape
        print train_labels.shape
        num_iter = self.model.train(train, train_labels, None, params = params)

        # Set end time
        e2 = cv2.getTickCount()
        time = (e2 - e1)/cv2.getTickFrequency()
        print 'Training duration:', time

        # Save param
        self.model.save(self.model_file)

        print 'Ran for %d iterations' % num_iter
        
    """
    @function      : load ()
    @description   : Load model from XML file and weight from txt file
    @param         : None 
    @return        : None
    """    
    def load(self):
        print 'load model ' , self.name
        self.model.load(self.model_file)
        
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
        # Apply filter felected 
        if (self.filter):
            samples = self.filter.applyOneSample(samples)

        # get good shape image according to the use case 
        samples = self.shapeRoadCaseImage(samples)
        '''
        if self.name.find('MAIN_NN') > 0:
            cv2.imshow('Main', samples)
            cv2.waitKey(1)
        ''' 
        #convert into float and 1D vector for nn 
        samples = samples.reshape(1,  IMAGE_PIXELS_X * IMAGE_PIXELS_Y / 2).astype(np.float32)

        ret, resp = self.model.predict(samples)
        
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
    @funtion          : computePredictionRate (test_set, labels_set)
    @description      : Predict response for full training set 
                        Store in self.weight the percentage of correct answer
    @param test_set   : Full Test set
    @type test_set    : Array of shape (m, NN_INPUT_LAYER)
    @param labels_set : Labels for test_set
    @type labels_set  : Array of shape (m, NN_OUTPUT_NUMBER)
    """
    def computePredictionRate(self, test_set, labels_set):      
        # Check Prediction all trained image_1D using for the current model
        print 'Testing...'
        mse=0
        for image,angle in zip(test_set,labels_set):
            w,p = self.predict(image)
            mse = mse + sqrt((p - angle)*(p - angle))

        mse = mse / test_set.shape[0]

        self.weight = ((MAX_ANGLE-MIN_ANGLE)-mse)*100/(MAX_ANGLE-MIN_ANGLE)

        cv2.destroyAllWindows()

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
                self.imgNNoutput[:,value*50:value*50+50] = predictionArray[0,value] * 255 / 1.715
            
        cv2.imshow(self.name, self.imgNNoutput)
        cv2.waitKey(1)
        
        # Plot results
        #print "predictionArray = " , predictionArray
        #plt.plot(predictionArray[0,:], 'ro')
        #plt.axis([0, 20, -2, 2])
        #plt.show()
            



        
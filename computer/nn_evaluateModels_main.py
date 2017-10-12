from commonDeepDriveDefine import *
from NNmodels.DNNclassModel import DNNclassificationModel
from NNmodels.CnnVggRegressionModel import CnnVggRegressionModel
from NNmodels.DNNclassificationKerasModel import DnnClassificationKerasModel
from NNmodels.DNNclassificationKerasModelMainNN import DnnClassificationKerasModelMainNN
from NNmodels.DNNclassificationModelMainNN import DNNclassificationModelMainNN
from NNmodels.CnnAutoEncoder import CnnAutoEncoder
from NNmodels.NNRegressionModelWithAutoencoder import NNRegressionModelWithAutoencoder
from NNmodels.RNNRegressionModelWithAutoencoder import RNNRegressionModelWithAutoencoder

# List of models for CAM1
modelList = []

# Add below all model you want to train
if DNNclassificationModelIsEnable:
    #modelList.append(DNNclassificationModel('DNNclassificationModel'))
    #modelList.append(DNNclassificationModel('DNNclassificationModel_CANNY','CANNY'))
    #modelList.append(DNNclassificationModel('CANNY_DNNclassificationModel_CAM2','CANNY'))
    #modelList.append(DNNclassificationModelMainNN('DNNclassificationModel_CANNY_MAIN_NN','CANNY'))
    #modelList.append(DNNclassificationModel('DNNClassification_CANNY_IDLE','CANNY','IDLE')) 
    pass
    
if CnnVggRegressionModelIsEnabled:
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_RIGHT_TURN','RIGHT_TURN'))
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_LEFT_TURN','LEFT_TURN'))
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_IDLE','IDLE'))
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_STRAIGHT','STRAIGHT'))
    pass
    
if DNNclassificationKerasModelIsEnable:
    #modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_RIGHT_TURN','CANNY','RIGHT_TURN'))
    #modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_LEFT_TURN','CANNY','LEFT_TURN'))
    #modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_IDLE','CANNY','IDLE'))
    #modelList.append(DnnClassificationKerasModelMainNN('DNNKerasClassification_CANNY_MAIN_NN','CANNY'))
    pass
    
if NNRegressionModelWithAutoEncoderIsEnabled:
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_IDLE','IDLE'))
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_STRAIGHT','STRAIGHT'))
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_LEFT_TURN','LEFT_TURN'))
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_RIGHT_TURN','RIGHT_TURN'))
    pass
    
if RNNRegressionModelWithAutoEncoderIsEnabled:
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_IDLE', 'IDLE'))
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_STRAIGHT', 'STRAIGHT'))
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_LEFT', 'LEFT_TURN'))
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_RIGHT', 'RIGHT_TURN'))
    pass


    
# Evaluate all model in the list of model with data coming from the path name below
for model in modelList :
    # Evaluate modelfor CAM1 with data from the directory passed in parameter
    print 'Evaluate model',model.name
    model.evaluate('training_data/*.npz')



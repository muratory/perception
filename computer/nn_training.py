from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from nnCommonDefine import *


from NNmodels.DNNclassModel import DNNclassificationModel
from NNmodels.CnnVggRegressionModel import CnnVggRegressionModel
from NNmodels.DNNclassificationKerasModel import DnnClassificationKerasModel
from NNmodels.DNNclassificationKerasModelMainNN import DnnClassificationKerasModelMainNN
from NNmodels.CnnAutoEncoder import CnnAutoEncoder
from NNmodels.NNRegressionModelWithAutoencoder import NNRegressionModelWithAutoencoder
from NNmodels.RNNRegressionModelWithAutoencoder import RNNRegressionModelWithAutoencoder

modelList = []
 

# Add below all model you want to train
if DNNclassificationKerasModelIsEnable:
    #modelList.append(DnnClassificationKerasModelMainNN('DNNKerasClassification_CANNY_MAIN_NN','CANNY'))
    pass


if DNNclassificationModelIsEnable:
    #modelList.append(DNNclassificationModelMainNN('DNNclassificationModel_CANNY_MAIN_NN','CANNY'))
    pass

#NOTE : no CnnVgg classification model availabel yet


#create one neural network for each roadLabel
if DNNclassificationKerasModelIsEnable:
    #modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_RIGHT_TURN','CANNY','RIGHT_TURN'))
    #modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_LEFT_TURN','CANNY','LEFT_TURN'))
    modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_IDLE','CANNY','IDLE'))
    #modelList.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_STRAIGHT','CANNY','STRAIGHT'))
    pass

if CnnVggRegressionModelIsEnabled:
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_RIGHT_TURN','RIGHT_TURN'))
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_LEFT_TURN','LEFT_TURN'))
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_IDLE','IDLE'))
    #modelList.append(CnnVggRegressionModel('CnnVggRegressionModel_STRAIGHT','STRAIGHT'))
    pass

if CnnAutoEncoderIsEnabled:
    #modelList.append(CnnAutoEncoder('CnnAutoEncoder'))
    pass

if NNRegressionModelWithAutoEncoderIsEnabled:
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_IDLE', 'IDLE'))
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_STRAIGHT','STRAIGHT'))
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_LEFT_TURN', 'LEFT_TURN'))
    #modelList.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_RIGHT_TURN','RIGHT_TURN'))
    pass

if RNNRegressionModelWithAutoEncoderIsEnabled:
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_IDLE', 'IDLE'))
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_STRAIGHT', 'STRAIGHT'))
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_LEFT_TURN', 'LEFT_TURN'))
    #modelList.append(RNNRegressionModelWithAutoencoder('RNNRegressionModelWithAutoencoder_RIGHT_TURN', 'RIGHT_TURN'))
    pass

    
#Please note lineDetection model does not need  etraining 
    
    
# Train all model in the list of model with data coming from the path name below
for model in modelList:
    #train mode1 with data from the directory passed in parameter
    print 'Train model ', model.name
    model.train('training_data/*.npz')

# Evaluate all model in the list of model with data coming from the path name below
for model in modelList:
    # evaluate model with data from the directory passed in parameter
    print 'Evaluate model', model.name
    model.evaluate('training_data/*.npz')

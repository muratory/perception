import cv2
import numpy as np
import threading
import Queue
import time
from scipy import stats

#libraries from deep drive
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from nnCommonDefine import *

from KeyboardThread import *
from VideoThread import *
from pathControlClientThread import *



from NNmodels.DNNclassModel import DNNclassificationModel
from NNmodels.CnnVggRegressionModel import CnnVggRegressionModel
from NNmodels.DNNclassificationKerasModel import DnnClassificationKerasModel
from NNmodels.DNNclassificationKerasModelMainNN import DnnClassificationKerasModelMainNN
from NNmodels.CnnAutoEncoder import CnnAutoEncoder
from NNmodels.NNRegressionModelWithAutoencoder import NNRegressionModelWithAutoencoder
from NNmodels.RNNRegressionModelWithAutoencoder import RNNRegressionModelWithAutoencoder



####################################### Deep Drive Thread ##############################


class nnMain(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)

        # create Video Stream Client Thread
        self.sctVideoStream = VideoThread()
        self.sctVideoStream.name = 'VideoSocketThread'
        self.sctVideoStream.start()

        # create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'nn_main_Kb'
        self.keyboardThread.start()
        
        #create Gps Thread
        self.srvNn = serverThread()
        self.srvNn.name = 'srvNnThread'
        self.srvNn.start()

        #create pathControlCommand client to receive the command such as SPEED for car control 
        self.sctPathControlCommandClientThread = pathControlCommandClientThread()
        self.sctPathControlCommandClientThread.name = 'sctPathControlCommandClientThread'
        self.sctPathControlCommandClientThread.start()
        

    def ConnectClient(self):
        # loop until all client connected
        videoClientConnected = False
        steerClientConnected = False
        nnSteeringConnected = False    
        pathControlCommandConnected = False  
        

        # launch connection thread for all client
        if videoClientEnable == True:
            self.sctVideoStream.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + CAR_IP + ':' +
                str(PORT_VIDEO_NN_SERVER) + '/?action=stream'))
            
        if nnSteeringEnable == True:
            self.srvNn.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_NN_STEERING_SERVER))
            
        if pathControlCommandEnable == True:
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_PATH_CONTROL_COMMAND_SERVER))
            

        while ((videoClientConnected != videoClientEnable) or
               (pathControlCommandConnected != pathControlCommandEnable) or
               (nnSteeringConnected != nnSteeringEnable)):

            
            if (videoClientConnected != videoClientEnable):
                try:
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        videoClientConnected = True
                        print 'Video stream server connected'
                except Queue.Empty:
                    print 'Video Client not connected'

            if (nnSteeringConnected != nnSteeringEnable):
                try:
                    reply = self.srvNn.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        nnSteeringConnected = True
                        print 'Neural Network server connected'
                except Queue.Empty:
                    print 'Neural Network Steer not ready'


            if (pathControlCommandConnected != pathControlCommandEnable):
                try:
                    reply = self.sctPathControlCommandClientThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        pathControlCommandConnected=True
                        print 'pathControl Command Client connected'
                except Queue.Empty:
                    print 'pathControl Command Client not connected'
            
            
            try:
                reply = self.keyboardThread.reply_q.get(False)
                if reply.type == ClientReply.SUCCESS:
                    if reply.data == 'exit':
                        return False
            except Queue.Empty:
                time.sleep(0.5)
                pass
            
        #otherwise it is ok and all is connected
        return True
            

    def run(self):
        
        
        modelListMAIN = []
        modelListNNRight = []
        modelListNNLeft = []
        modelListNNIdle = []
        modelListNNStraight = []
        
        pathControlCommandLabel = 'NONE'
        
        keyBoardOverride = 0
        lastKeyPressed = 0

        forceNNroadLabel = 'NONE'
        mainSelectionByGpsps = False
        
        lastSteerControlTime = time.time()
        
        if gpsEnable == True and pathControlCommandEnable == True:
            mainSelectionByGpsps = True
        else:
            forceNNroadLabel = 'IDLE'

        
        print 'LOADING NEURAL NETWORK. PLEASE WAIT'
        
        # create all neural network model you need and add them to list for prediction
        if DNNclassificationModelIsEnable:
            #modelListMAIN.append(DnnClassificationKerasModelMainNN('DNNKerasClassification_CANNY_MAIN_NN','CANNY'))
            #modelListNNRight.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_RIGHT_TURN','CANNY','RIGHT_TURN'))
            #modelListNNLeft.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_LEFT_TURN','CANNY','LEFT_TURN'))
            #modelListNNIdle.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_IDLE','CANNY','IDLE'))
            #modelListNNStraight.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_STRAIGHT','CANNY','STRAIGHT'))
            pass
            
        if CnnVggRegressionModelIsEnabled:
            # modelListNNRight.append(CnnVggRegressionModel('CnnVggRegressionModel_RIGHT_TURN','RIGHT_TURN'))
            # modelListNNLeft.append(CnnVggRegressionModel('CnnVggRegressionModel_LEFT_TURN','LEFT_TURN'))
            # modelListNNIdle.append(CnnVggRegressionModel('CnnVggRegressionModel_IDLE','IDLE'))
            # modelListNNStraight.append(CnnVggRegressionModel('CnnVggRegressionModel_STRAIGHT','STRAIGHT'))
            pass
            
        if DNNclassificationKerasModelIsEnable:
            # modelListMAIN.append(DnnClassificationKerasModelMainNN('DNNKerasClassification_CANNY_MAIN_NN','CANNY'))
            # modelListNNRight.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_RIGHT_TURN','CANNY','RIGHT_TURN'))
            # modelListNNLeft.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_LEFT_TURN','CANNY','LEFT_TURN'))
            #modelListNNIdle.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_IDLE','CANNY','IDLE'))
            #modelListNNStraight.append(DnnClassificationKerasModel('DNNKerasClassification_CANNY_STRAIGHT','CANNY','STRAIGHT'))
            pass
            
        if NNRegressionModelWithAutoEncoderIsEnabled:
            modelListNNIdle.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_IDLE','IDLE'))
            modelListNNStraight.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_STRAIGHT','STRAIGHT'))
            #modelListNNLeft.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_LEFT_TURN','LEFT_TURN'))
            modelListNNRight.append(NNRegressionModelWithAutoencoder('NNRegressionModelWithAutoencoder_RIGHT_TURN','RIGHT_TURN'))
            pass

        #init dummy image to a value
        dummyImage = np.zeros((IMAGE_PIXELS_Y, IMAGE_PIXELS_X), dtype=np.uint8)
        
        # load all model
        for model in modelListMAIN:
            model.load()
	    model.predict(dummyImage)
        
        for model in modelListNNRight:
            model.load()
	    model.predict(dummyImage)
            
        for model in modelListNNLeft:
            model.load()
	    model.predict(dummyImage)
            
        for model in modelListNNIdle:
            model.load()
	    model.predict(dummyImage)
            
        for model in modelListNNStraight:
            model.load()
	    model.predict(dummyImage)
        
        
        print 'NEURAL NETWORK LOADED'
        
        # init running prediction average values with null angle. MANDATORY for the first sample
        predictionValuesRoadUseCase = np.zeros(NB_SAMPLE_RUNNING_AVERAGE_PREDICTION_MAIN_NN, dtype=np.int)
        predictionValuesSteeringAngle = np.zeros(NB_SAMPLE_RUNNING_AVERAGE_PREDICTION, dtype=np.int)
        predictionIndexRoadSelectionNN = 0
        predictionIndexSteeringAngle = 0
        
        lastAngleSent = 0


        # initial steer command set to stop
        try:

            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
                        # start receiver thread client to receive continuously data
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))

            #receive command from path cotronl such as IDLE ... 
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))



            print 'Start Main Thread for Neural Network Prediction'
            
            while True:
                ############################# Manage IMAGE for Deep neural network to extract Steer Command ###############
                try:
                    # try to see if image ready for CAM1
                    replyVideo = self.sctVideoStream.reply_q.get(False)
                    if replyVideo.type == ClientReply.SUCCESS:

                        # print length as debug
                        # print 'length =' + str(len(self.sctVideoStream.lastImage))

                        # decode jpg into array
                        i = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),-1)

                        #keep only gray image for NN
                        image2nn = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
 
                        # for all model , make a prediction and average it
                        # (could be other processing here)
                        NNRoaduseCase = 0
                        totalWeight   = 0

                        #start timer to measure the time of the predictions
                        e1 = cv2.getTickCount()
                        
                        if forceNNroadLabel != 'NONE':
                            NNRoaduseCase = forceNNroadLabel
                        else:
                            if mainSelectionByGpsps == True:
                                #gps2MainRoadLabel always return STRAIGHT in case of interesction
                                NNRoaduseCase = pathControlCommandLabel
                            else:
                                NNuseCasePrediction = 0
                                totalWeight = 0
                                #no gps .. then use Main NN to elaborate the road use case
                                for model in modelListMAIN:
                                    # make prediction
                                    (weight, prediction) = model.predict(image2nn)
                                    NNuseCasePrediction += prediction*weight
                                    totalWeight    += weight
                                    #print 'prediction model ',model.name,' = ',prediction

                                
                                if len(modelListMAIN) > 0  and (totalWeight != 0):
                                    NNuseCasePrediction = NNuseCasePrediction/totalWeight
                                else:
                                    print 'ERROR model list is empty ',len(modelListMAIN), ' ',totalWeight

                                if totalWeight != 0:
                                    # fill averaged prediction table
                                    predictionValuesRoadUseCase[predictionIndexRoadSelectionNN] = round(NNuseCasePrediction)
                                    predictionIndexRoadSelectionNN += 1
                                    if predictionIndexRoadSelectionNN >= NB_SAMPLE_RUNNING_AVERAGE_PREDICTION_MAIN_NN:
                                        predictionIndexRoadSelectionNN = 0
                                    
                                # Compute the averaged prediction of the best road use case detected
                                # get the most common element in table
                                NNuseCasePrediction = stats.mode(predictionValuesRoadUseCase)[0]
                                if NNuseCasePrediction == 1:
                                    NNRoaduseCase = 'STRAIGHT'
                                else:
                                    NNRoaduseCase = 'IDLE'

        
                        
                        #select the best NN to use according to usecase detected
                        if NNRoaduseCase == 'RIGHT_TURN':
                            modelList = modelListNNRight
                        elif NNRoaduseCase == 'LEFT_TURN':
                            modelList = modelListNNLeft
                        elif NNRoaduseCase == 'STRAIGHT':
                            modelList = modelListNNStraight
                        else:
                            modelList = modelListNNIdle

                            
                        #now getting valur for steering 
                        NNsteerCommand = 0
                        totalWeight    = 0
                        for model in modelList:
                            # make prediction
                            (weight, prediction) = model.predict(image2nn)
                            NNsteerCommand += prediction*weight
                            totalWeight    += weight
                            # print 'prediction model ',model.name,' = ',prediction #, "\tweight=",weight


                        if len(modelList) > 0  and (totalWeight != 0):
                            NNsteerCommand = NNsteerCommand/totalWeight
                        else:
                            #print 'ERROR model list is empty ',len(modelList), ' ',totalWeight
                            pass

                        if totalWeight != 0:
                            # fill average angle table based on prediction
                            predictionValuesSteeringAngle[predictionIndexSteeringAngle] = round(NNsteerCommand)
                            #print 'Prediction Average = ', + self.predictionValuesToAverage
                            predictionIndexSteeringAngle += 1
                            if predictionIndexSteeringAngle >= NB_SAMPLE_RUNNING_AVERAGE_PREDICTION:
                                predictionIndexSteeringAngle = 0                                
                                
                        e2 = cv2.getTickCount()
                        t = (e2 - e1)/cv2.getTickFrequency()

                        if (t > STEERING_PREDICTION_SAMPLING_TIME):
                            print 'WARNING : too much time to predict = ', t
                            cv2.putText(i, 'PREDICT TIME TOO HIGH=' + str(int(t*1000)), (0,IMAGE_PIXELS_Y/2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        else:
                            #we have time to display some stuff
                            #write speed
                            cv2.putText(i, 'RoadUseCase = ' + NNRoaduseCase, (0,IMAGE_PIXELS_Y/2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1) 
                            cv2.putText(i, 'PredicTime  = ' + str(int(t*1000)), (0,IMAGE_PIXELS_Y/2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                            cv2.putText(i, 'SteerAngle  = ' + str(lastAngleSent), (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                            
                            if forceNNroadLabel != 'NONE':
                                cv2.putText(i, 'MANUAL SELECTION', (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            else :
                                if mainSelectionByGpsps == True:
                                    cv2.putText(i, 'GPS SELECTION', (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                                else:
                                    cv2.putText(i, 'NN SELECTION', (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            
                            showLabel(lastAngleSent,'NNvision', i)
                            cv2.imshow('NNvision', i)
                            # check if we want to stop autonomous driving
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    else:
                        print 'Error getting image :' + str(replyVideo.data)
                        break
                        
                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass


                #############################get Command from PathControlCommand server   ###############
                try:
                    # try to see if data ready
                    reply = self.sctPathControlCommandClientThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        strCommand = str(reply.data)
                        #filter only interesting command for neural network
                        if strCommand == 'IDLE' or strCommand == 'LEFT' or strCommand == 'RIGHT' or strCommand == 'STRAIGHT' :
                            pathControlCommandLabel = strCommand
                            print 'receive label ',pathControlCommandLabel,' t=', time.time()

                    else:
                        print 'Error getting path control command :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass


                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        # new keyboard input found
                        keyPressed = reply.data
                        strText = keyPressed

                        if keyPressed == 'exit':
                            turn_angle = 0
                            # get out of the loop
                            break
                        
                        elif keyPressed == 'help':
                            strText = 'Key:Q,R,I,L,S,G(ps)'
                            
                        elif (keyPressed == 'RIGHT_TURN') or (keyPressed == 'IDLE') or (keyPressed == 'LEFT_TURN') or (keyPressed == 'STRAIGHT'):
                            forceNNroadLabel = keyPressed  
                            
                        elif (keyPressed == 'GPS'):
                            forceNNroadLabel = 'NONE'
                            mainSelectionByGpsps = True
                        
                        elif keyPressed == 'none':
                            #no handle of no keyboard pressed
                            strText=''
                            
                        else:
                            # key not known display error
                            self.keyboardThread.displayText()
                            strText=''

                        if strText != '':
                            self.keyboardThread.displayText(strText)

                        # record lastkey that can be use for consecutive command action
                        lastKeyPressed = keyPressed
                                                      

                    else:
                        print 'Error getting keyboard input :' + str(reply.data)
                        break
                except Queue.Empty:
                    # queue empty most of the time because keyboard not hit
                    pass


                ############### Compute/send Angle to the server####################
                # send control command according to sampling dedicated for it
                timeNow = time.time()
                if timeNow > (lastSteerControlTime + STEERING_PREDICTION_SAMPLING_TIME):
                    prediction = np.sum(predictionValuesSteeringAngle, dtype=int) / NB_SAMPLE_RUNNING_AVERAGE_PREDICTION
                    #print 'NN prediction = ' + str(predictionValuesSteeringAngle) + ' , average_value = ' + str(prediction)
                    #print 'UseCasePredict  = ' + str(predictionValuesRoadUseCase) + ' , Best_Value = ' + NNRoaduseCase )
                    
                    # Send command
                    self.srvNn.cmd_q.put(ClientCommand(ClientCommand.SEND, prediction))
                    lastAngleSent = prediction
                    lastSteerControlTime = timeNow
                    
                    
        finally:
            print 'ending Neural Network main'
            # stop and close all client and close them
            self.srvNn.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvNn.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            # and make sure all of them ended properly
            self.srvNn.join()
            self.sctVideoStream.join()
            self.keyboardThread.join()
            self.sctPathControlCommandClientThread.join()
            print ' Neural Network main Done'
            
            






if __name__ == '__main__':
    # create Deep drive thread and strt
    nnMain = nnMain()
    nnMain.name = 'nnMain'

    # start
    nnMain.start()

    nnMain.join()
    print 'end'

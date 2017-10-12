import cv2
import numpy as np
import threading
import Queue
import time
from scipy import stats

from RoadLaneDetection.road_lane_detector import PID_process
from roadLineDetectionModel import LineDetectionModel

#libraries from deep drive
from commonDeepDriveDefine import *
from commonDeepDriveTools import *

from KeyboardThread import *
from VideoThread import *
from pathControlClientThread import *

####################################### Deep Drive Thread ##############################


class roadLine(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)

        # create Video Stream Client Thread
        self.sctVideoStream = VideoThread()
        self.sctVideoStream.name = 'VideoSocketThread'
        self.sctVideoStream.start()

        # create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'roadLine_main_Kb'
        self.keyboardThread.start()
        
        #create Gps Thread
        self.srvRoadLine = serverThread()
        self.srvRoadLine.name = 'srvRoadLineThread'
        self.srvRoadLine.start()

        #create pathControlCommand client to receive the command such as SPEED for car control 
        self.sctPathControlCommandClientThread = pathControlCommandClientThread()
        self.sctPathControlCommandClientThread.name = 'sctPathControlCommandClientThread_RL'
        self.sctPathControlCommandClientThread.start()

    def ConnectClient(self):
        # loop until all client connected
        videoClientConnected = False
        steerClientConnected = False
        roadLineSteeringConnected = False    
        pathControlCommandConnected = False  
        

        # launch connection thread for all client
        if videoClientEnable == True:
            self.sctVideoStream.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + CAR_IP + ':' +
                str(PORT_VIDEO_NN_SERVER) + '/?action=stream'))
            
        if roadLineSteeringEnable == True:
            self.srvRoadLine.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_ROADLINE_STEERING_SERVER))
            
        if pathControlCommandEnable == True:
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_PATH_CONTROL_COMMAND_SERVER))

        while ((videoClientConnected != videoClientEnable) or
               (pathControlCommandConnected != pathControlCommandEnable) or
               (roadLineSteeringConnected != roadLineSteeringEnable)):

            # wait for .5 second before to check
            time.sleep(0.5)

            if (videoClientConnected != videoClientEnable):
                try:
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        videoClientConnected = True
                        print 'Video stream server connected'
                except Queue.Empty:
                    print 'Video Client not connected'

            if (roadLineSteeringConnected != roadLineSteeringEnable):
                try:
                    reply = self.srvRoadLine.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        roadLineSteeringConnected = True
                        print 'Road line steering server connected'
                except Queue.Empty:
                    print 'Road line steering server not ready'


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

    def run(self):

        pathControlCommandLabel = 'NONE'
        
        keyBoardOverride = 0
        lastKeyPressed = 0

        forceRoadLineLabel = 'NONE'
        mainSelectionByGpsps = False
        
        lastSteerControlTime = time.time()
        
        if gpsEnable == True and pathControlCommandEnable == True:
            mainSelectionByGpsps = True
        else:
            forceRoadLineLabel = 'IDLE'
    
        #get LineDetect model
        roadLineModel = LineDetectionModel('LineDetectionModel')

        
        predictionValuesSteeringAngle = np.zeros(NB_SAMPLE_RUNNING_AVERAGE_PREDICTION, dtype=np.int)
        predictionIndexSteeringAngle = 0
        
        lastAngleSent = 0


        # initial steer command set to stop
        try:
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread for Road Line detect estimate'

            # start receiver thread client to receive continuously data
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            #receive command from path cotronl such as IDLE ... 
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            while True:
                ############################# Manage IMAGE for Deep neural network to extract Steer Command ###############
                try:
                    # try to see if image ready for CAM1
                    replyVideo = self.sctVideoStream.reply_q.get(False)
                    if replyVideo.type == ClientReply.SUCCESS:

                        # print length as debug
                        # print 'length =' + str(len(self.sctVideoStream.lastImage))

                        # decode jpg into array
                        image2RoadLine = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),-1)

 
                        # for all model , make a prediction and average it
                        # (could be other processing here)
                        roadLineUseCase = 0
                        totalWeight   = 0

                        #start timer to measure the time of the predictions
                        e1 = cv2.getTickCount()
                        
                        if forceRoadLineLabel != 'NONE':
                            roadLineUseCase = forceRoadLineLabel
                        else:
                            if mainSelectionByGpsps == True:
                                #gps2MainRoadLabel always return STRAIGHT in case of interesction
                                roadLineUseCase = pathControlCommandLabel
                            else:
                                forceRoadLineLabel = 'IDLE'
                                roadLineUseCase = forceRoadLineLabel
                        
                        
                            
                        #now getting value for steering 
                        (roadLineOffset, roadLineSteerCommand) = roadLineModel.predict(image2RoadLine,roadLineUseCase)
                        # print 'roadLineSteerCommand = ',roadLineSteerCommand,
                        predictionValuesSteeringAngle[predictionIndexSteeringAngle] = round(roadLineSteerCommand)

                        # Steer from PID based on offset
                        if roadLineOffset != 999 :
                            # pidSteerCommand = PID_process (roadLineOffset)
                            # print 'roadLineOffset= ',roadLineOffset,'\troadLineSteerCommand= ',roadLineSteerCommand,'\tpidSteerCommand= ',pidSteerCommand
                            # predictionValuesSteeringAngle[predictionIndexSteeringAngle] = round(pidSteerCommand)
                            pass

                        # fill average angle table based on prediction
                        # predictionValuesSteeringAngle[predictionIndexSteeringAngle] = round(pidSteerCommand)
                        #print 'Prediction Average = ', + self.predictionValuesToAverage
                        predictionIndexSteeringAngle += 1
                        if predictionIndexSteeringAngle >= NB_SAMPLE_RUNNING_AVERAGE_PREDICTION:
                            predictionIndexSteeringAngle = 0                                

                        e2 = cv2.getTickCount()
                        t = (e2 - e1)/cv2.getTickFrequency()

                        if (t > STEERING_PREDICTION_SAMPLING_TIME):
                            print 'WARNING : too much time to predict = ', t
                            cv2.putText(image2RoadLine, 'PREDICT TIME TOO HIGH=' + str(int(t*1000)), (0,IMAGE_PIXELS_Y/2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        else:
                            #we have time to display some stuff
                            #write speed
                            cv2.putText(image2RoadLine, 'RoadUseCase = ' + roadLineUseCase, (0,IMAGE_PIXELS_Y/2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1) 
                            cv2.putText(image2RoadLine, 'PredicTime  = ' + str(int(t*1000)), (0,IMAGE_PIXELS_Y/2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                            cv2.putText(image2RoadLine, 'SteerAngle  = ' + str(lastAngleSent), (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                            
                            if forceRoadLineLabel != 'NONE':
                                cv2.putText(image2RoadLine, 'MANUAL SELECTION', (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            else :
                                if mainSelectionByGpsps == True:
                                    cv2.putText(image2RoadLine, 'GPS SELECTION', (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            
                            showLabel(lastAngleSent,'roadLinevision', image2RoadLine)
                            cv2.imshow('roadLinevision', image2RoadLine)
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
                        if strCommand == 'IDLE' or strCommand == 'LEFT_TURN' or strCommand == 'RIGHT_TURN' or strCommand == 'STRAIGHT' :
                            pathControlCommandLabel = strCommand

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
                            forceRoadLineLabel = keyPressed  
                            
                        elif (keyPressed == 'GPS'):
                            forceRoadLineLabel = 'NONE'
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
                    #print 'UseCasePredict  = ' + str(predictionValuesRoadUseCase) + ' , Best_Value = ' + roadLineUseCase )
                    
                    # Send angle to the server and all client connected
                    if lastAngleSent != prediction:
                        self.srvRoadLine.cmd_q.put(ClientCommand(ClientCommand.SEND, prediction))
                        lastAngleSent = prediction
                    lastSteerControlTime = timeNow
                    
                    
        finally:
            print 'ending road Line detection main'

            # stop and close all client and close them
            self.srvRoadLine.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvRoadLine.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            # and make sure all of them ended properly
            self.srvRoadLine.join()
            self.sctVideoStream.join()
            self.keyboardThread.join()
            self.sctPathControlCommandClientThread.join()
            print ' road Line detection main Done'
            


if __name__ == '__main__':
    # create Deep drive thread and strt
    print "Create RoadLine Main"
    roadLine = roadLine()
    roadLine.name = 'nnMain'

    # start
    print "Start RoadLine Main"
    roadLine.start()

    roadLine.join()
    print 'end'

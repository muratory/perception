import cv2
import numpy as np
import threading
import Queue
import time

#libraries from deep drive
from commonDeepDriveDefine import *
from KeyboardThread import *
from SteerThread import *
from VideoThread import *
from SensorThread import *
from pathControlClientThread import *
from steerClientThread import *
from gpsClientThread import *


####################################### Deep Drive Thread ##############################


class DeepDriveThread(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)

        # create Video Stream Client Thread
        self.sctVideoCarStream = VideoThread()
        self.sctVideoCarStream.name = 'sctVideoCarStream'
        self.sctVideoCarStream.start()

        # create Steer Client Thread
        self.sctCarSteering = SteerThread()
        self.sctCarSteering.name = 'SteerThread'
        self.sctCarSteering.start()

        # create Sensor Client Thread
        self.sctCarSensor = SensorThread()
        self.sctCarSensor.name = 'SensorThread'
        self.sctCarSensor.start()

        # create Keyboard Thread
        self.keyboardThread = keyboardThread(1800,800)
        self.keyboardThread.name = 'car_main_kb'
        self.keyboardThread.start()
        
        #create path control steering client thread
        self.sctPathControlSteeringClientThread = steerClientThread()
        self.sctPathControlSteeringClientThread.name = 'sctPathControlSteeringClientThread'
        self.sctPathControlSteeringClientThread.start()
        
        #create pathControlCommand client to receive the command such as SPEED for car control 
        self.sctPathControlCommandClientThread = pathControlCommandClientThread()
        self.sctPathControlCommandClientThread.name = 'sctPathControlCommandClientThread'
        self.sctPathControlCommandClientThread.start()
        
        #create neural network prediction Client Thread
        self.sctNnSteerThread = steerClientThread()
        self.sctNnSteerThread.name = 'nnSteeringClientThread'
        self.sctNnSteerThread.start()

        #create road Line Client Thread
        self.sctRoadLineSteerThread = steerClientThread()
        self.sctRoadLineSteerThread.name = 'roadLineSteeringClientThread'
        self.sctRoadLineSteerThread.start()

        #create Gps Thread
        self.sctGps = gpsClientThread()
        self.sctGps.name = 'GpsClientThread'
        self.sctGps.start()



    def ConnectClient(self):
        # loop until all client connected
        videoCarClientConnected = False
        steerClientConnected = False
        sensorClientConnected = False
        pathControlSteeringConnected = False
        nnSteeringConnected = False
        roadLineSteeringConnected = False
        pathControlCommandConnected = False
        gpsClientConnected = False

        # launch connection thread for all client
        if videoCarClientEnable == True:
            self.sctVideoCarStream.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + CAR_IP + ':' +
                str(PORT_VIDEO_CAR_SERVER) + '/?action=stream'))
            
        if steerClientEnable == True :
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_STEER_SERVER))
            
        if sensorClientEnable == True:
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_SENSOR_SERVER))
            
        if pathControlSteeringEnable == True:
            self.sctPathControlSteeringClientThread.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_PATH_CONTROL_STEERING_SERVER))
            
        if pathControlCommandEnable == True:
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_PATH_CONTROL_COMMAND_SERVER))
            
        if nnSteeringEnable == True:
            self.sctNnSteerThread.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_NN_SERVER))

        if roadLineSteeringEnable == True:
            self.sctRoadLineSteerThread.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_ROADLINE_SERVER))

        if gpsEnable == True:
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_GPS_FIX_SERVER))
            
        while ((videoCarClientConnected != videoCarClientEnable) or
                (steerClientConnected != steerClientEnable) or
                (sensorClientConnected != sensorClientEnable) or
                (pathControlSteeringConnected != pathControlSteeringEnable) or
                (pathControlCommandConnected != pathControlCommandEnable) or
                (gpsClientConnected != gpsEnable) or
                (roadLineSteeringConnected != roadLineSteeringEnable) or
                (nnSteeringConnected != nnSteeringEnable)):

            # wait for .5 second before to check
            time.sleep(0.5)

            if (videoCarClientConnected != videoCarClientEnable):
                try:
                    reply = self.sctVideoCarStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        videoCarClientConnected = True
                        print 'Video stream server connected'
                except Queue.Empty:
                    print 'Video Client not connected'

            if (steerClientConnected != steerClientEnable):
                try:
                    reply = self.sctCarSteering.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        steerClientConnected = True
                        print 'Steer server connected'
                except Queue.Empty:
                    print 'Steer Client not connected'

            if (sensorClientConnected != sensorClientEnable):
                try:
                    reply = self.sctCarSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sensorClientConnected = True
                        print 'Sensor server connected'
                except Queue.Empty:
                    print 'Sensor Client not connected'
                    
            if (pathControlSteeringConnected != pathControlSteeringEnable):
                try:
                    reply = self.sctPathControlSteeringClientThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        pathControlSteeringConnected=True
                        print 'pathControl Steering Client not connected'
                except Queue.Empty:
                    print 'pathControl Steering Client connected' 
            
                    
            if (pathControlCommandConnected != pathControlCommandEnable):
                try:
                    reply = self.sctPathControlCommandClientThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        pathControlCommandConnected=True
                        print 'pathControl Command Client connected'
                except Queue.Empty:
                    print 'pathControl Command Client not connected' 


            if (nnSteeringConnected != nnSteeringEnable):
                try:
                    reply = self.sctNnSteerThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        nnSteeringConnected=True
                        print 'NN steering server connected'
                except Queue.Empty:
                    print 'NN steering not connected'
                    
            if (roadLineSteeringConnected != roadLineSteeringEnable):
                try:
                    reply = self.sctRoadLineSteerThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        roadLineSteeringConnected=True
                        print 'RoadLine steering server connected'
                except Queue.Empty:
                    print 'RoadLine steering not connected'
                    
            if (gpsClientConnected != gpsEnable):
                try:
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        gpsClientConnected=True
                        print 'Gps server connected'
                except Queue.Empty:
                    print 'Gps Client not connected' 
                    
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
        inputAngle = 0
        totalRecordTime = 0
        lastKeyPressed = 0
        lastSteerKeyboardAngle = 0
        total_frame = 0
        saved_frame = 0
        recordTime = 0
        record = 0
        startCar = 0
        turn_angle=0
        pathControlSpeed = 0
        pathControlCommand = 'None'
        pwmSpeed = 0
        gpsSpeed = 0
        lastpwmSpeed = 0
        offsetSpeed = 0
        steerAngleUpdate = False
        gpsSpeedControl = False
        

        # init timing
        lastSteerControlTime = time.time()
        lastSteerKeyboardTime = time.time()
        lastPathTime = time.time()

 
        #different Steering Angle coming from different module :
        steerKeyboardAngle = 0 #angle from keyboard when forced by manual intervention
        steerPathAngle = 0 #angle coming from Position module and graph slam ...
        steerNnAngle = 0       #angle coming from Neural network prediction module
        steerRoadLineAngle = 0       #angle coming from roadLine prediction module
        inputAngle = 'None' #where comes from Angle steerKeyboardAngle,steerPathAngle,steerNnAngle, mix
        lastInputAngle = 'None' #record the last Angle seed 
        
        if gpsEnable == False:
            pathControlCommand = 'IDLE'
            pathControlSpeed = INITIAL_CAR_SPEED
            pwmSpeed = 0
        else:
            gpsSpeedControl = True           
        
        # Angle command sent to The car 
        self.steerAngleCommand = 0
        
        #array for correction record when pushing space
        image_array_correction = np.zeros((1,IMAGE_PIXELS_Y, IMAGE_PIXELS_X,3), dtype=np.uint8)            
        steerAngle_array_correction = np.zeros(1, dtype=np.uint8)
        NNroadLabel_array_correction = np.zeros(1, dtype=np.uint8)

        IerrorSpeed = 0
        DerrorSpeed = 0
        errorSpeed = 0
        lastErrorSpeed = 0
        
        cv2.namedWindow('CarVision')
        #display the car vision and info
        cv2.moveWindow('CarVision', 1300,700)
        
        # initial steer command set to stop
        try:
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread and sub Thread for Car Driver'
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            self.sctPathControlSteeringClientThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))
            self.sctNnSteerThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))
            self.sctRoadLineSteerThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, VEHICULE_NAME))


            # start car to be able to see additioanl data
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED', pwmSpeed)))
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE', 0)))

            lastFrameTime    = 0
            
            while True:
                ############################# Manage IMAGE from car Camera ###############
                try:
                    # try to see if image ready for car vision
                    replyVideo = self.sctVideoCarStream.reply_q.get(False)
                    if replyVideo.type == ClientReply.SUCCESS:

                        # decode jpg into array
                        i = cv2.imdecode(np.fromstring(self.sctVideoCarStream.lastImage, dtype=np.uint8),-1)
                        image2nn = i.copy()
                        
                        #we have time to display some stuff
                        #write speed
                        cv2.putText(i, 'Input Angle  = ' + inputAngle, (0,IMAGE_PIXELS_Y/2 + 0), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'SteerAngle  = ' + str(self.steerAngleCommand), (0,IMAGE_PIXELS_Y/2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'speed Command     = ' + str(pathControlSpeed), (0,IMAGE_PIXELS_Y/2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'speed Measure     = ' + str(gpsSpeed), (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'Speed PWM         = ' + str(pwmSpeed), (0,IMAGE_PIXELS_Y/2 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'Command = ' + str(pathControlCommand), (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        if gpsSpeedControl == True:
                            cv2.putText(i, 'GPS speed control ENABLE' , (0,IMAGE_PIXELS_Y/2 + 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        else:
                            cv2.putText(i, 'GPS speed control DISABLE' , (0,IMAGE_PIXELS_Y/2 + 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            
                            
                        showLabel(self.steerAngleCommand,'CarVision', i)
                        
                        #display the car vision and info
                        cv2.imshow('CarVision', i)
                        
                        # check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print 'Error getting image :' + str(replyVideo.data)
                        break
                    
                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass
                
                #############################Handle position angle from NN  ###############
                try:
                    # try to see if data ready
                    reply = self.sctNnSteerThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        print 'NN module provide Angle =',str(reply.data)
                        steerNnAngle = int(reply.data)
                        steerAngleUpdate = True
                    else:
                        print 'Error getting NN module angle :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass


                #############################Handle position angle from NN  ###############
                try:
                    # try to see if data ready
                    reply = self.sctRoadLineSteerThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        print 'RoadLine module provide Angle =',str(reply.data)
                        steerRoadLineAngle = int(reply.data)
                        steerAngleUpdate = True
                    else:
                        print 'Error getting Road Line module angle :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass

                #############################Handle position angle coming from Path control ###############
                #check now if the slamGraph Thread posted a new steeringAngle
                try:
                    # try to see if data ready
                    reply = self.sctPathControlSteeringClientThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #print 'Path control module provide Angle =',str(reply.data)
                        steerPathAngle = int(reply.data)
                        steerAngleUpdate = True
                    else:
                        print 'Error getting Path control data :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass

                #############################get Command from PathControlCommand server   ###############
                try:
                    # try to see if data ready
                    reply = self.sctPathControlCommandClientThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        strCommand = str(reply.data)
                        #print 'Command receive',strCommand
                        if strCommand.find('SPEED') >= 0:
                            speedList = strCommand.split('=')
                            #keep only the speed
                            print 'Speed Command receive speed = ',speedList[1]
                            #convert into int
                            pathControlSpeed = int(speedList[1])

                        else:
                            #this is another command that need to be handled differently
                            pathControlCommand = strCommand

                    else:
                        print 'Error getting path control command :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass



                ############################# regulate car speed with Gps speed measure ###############
                try:

                    # try to see if image ready
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                            #fix has been receive . process it
                            vPosX, vPosY, vSpeed, vOrient, vName = reply.data.split(',')
                            #print 'Received FIX for vehicle ',vName, ' x=', vPosX, ' y=', vPosY
                            if (vName == VEHICULE_NAME):
                                ############### SEND value to Graph slam ############
                                gpsPosition = (int(vPosX),int(vPosY))  
                                gpsSpeed = int(vSpeed)  
                                
                                #compute diff between expected and real
                                errorSpeed = pathControlSpeed-gpsSpeed
                                
                                #reset to pathControlSpeed value when no move
                                if startCar == 0 or gpsSpeedControl == False:
                                    #no control of PID, just compute pwmSpeed
                                    pwmSpeed = int(((pathControlSpeed + offsetSpeed) * MAX_CAR_SPEED_COMMAND)/ MAX_CAR_SPEED )
                                    IerrorSpeed = 0
                                    DerrorSpeed = 0
                                    errorSpeed = 0
                                    
                                else:                      

                                    IerrorSpeed = IerrorSpeed + errorSpeed

                                                         
                                    DerrorSpeed = errorSpeed - lastErrorSpeed
                                    
                                                                      
                                    #add portion of error
                                    newPathControlSpeed = errorSpeed*P_SPEED_REGULATION + IerrorSpeed * I_SPEED_REGULATION + DerrorSpeed * D_SPEED_REGULATION
    
                                    pwmSpeed = int((newPathControlSpeed * MAX_CAR_SPEED_COMMAND)/ MAX_CAR_SPEED )
                                    
                                    lastErrorSpeed = errorSpeed

                    else:
                        print 'Error getting Gps value :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass      
                
                
                 
                ############################# Get Sensor value ###############
                try:
                    # try to see if image ready
                    reply = self.sctCarSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        if (reply.data < STOP_DISTANCE):
                            print 'sensor value = ' + str(reply.data)
                            if (inputAngle == 0) :
                                self.sctCarSteering.cmd_q.put(ClientCommand(
                                    ClientCommand.SEND, ('STEER_COMMAND', 'stop')))
                        elif startCar == 1 :
                            #move forward the car only if we ordered to start it previously 
                            self.sctCarSteering.cmd_q.put(ClientCommand(
                                ClientCommand.SEND, ('STEER_COMMAND', 'forward')))

                    else:
                        print 'Error getting Sensor :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because image not ready
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
                            sourceAngle = 0
                            turn_angle = 0
                            # get out of the loop
                            break
                        elif keyPressed == 'help':
                            strText = 'Key:Q,Arrow,+,-,N,P,R,G,space'

                        elif keyPressed == 'right':
                            inputAngle = 'steerKeyboardAngle'
                            turn_angle = STEP_CAPTURE * 2


                        elif keyPressed == 'left':
                            inputAngle = 'steerKeyboardAngle'
                            turn_angle = -STEP_CAPTURE * 2

                        elif keyPressed == 'PATH_CONTROL':
                            inputAngle = 'steerPathAngle'
                            if lastInputAngle == 'steerPathAngle':
                                inputAngle = 'None'
                            else:
                                lastInputAngle = inputAngle
                            
                        elif keyPressed == 'NN_CONTROL':
                            inputAngle = 'steerNnAngle'
                            if lastInputAngle == 'steerNnAngle':
                                inputAngle = 'None'
                            else:
                                lastInputAngle = inputAngle
                            
                        #overide of Right turn key
                        elif keyPressed == 'RIGHT_TURN':
                            strText='ROADLINE_CONTROL'
                            inputAngle = 'steerRoadLineAngle'
                            if lastInputAngle == 'steerRoadLineAngle':
                                inputAngle = 'None'
                            else:
                                lastInputAngle = inputAngle

                        elif keyPressed == 'up':
                            if (self.sctCarSteering.steerCommand == 'backward') :
                                # stop if we were in back before 
                                self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                                #but no record since we are stop
                            else:
                                #in all other case, go forward and record
                                self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))
                                startCar = 1

                        elif keyPressed == 'down':
                            startCar = 0
                            if (self.sctCarSteering.steerCommand == 'stop') :
                                # go back if we were stop before 
                                self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','backward')))
                            else:
                                #in all other case, just stop
                                self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))   
                            
                        elif keyPressed == 'minus':
                            offsetSpeed -= 100
                            
                        elif keyPressed == 'plus':
                            offsetSpeed += 100

                        elif keyPressed == 'space':
                            if record==1:
                                record = 0
                            else:
                                record = 1
                            print "Record set to ",record
                        
                        elif keyPressed == 'GPS':
                            if gpsSpeedControl == True:
                                gpsSpeedControl = False
                            else:
                                gpsSpeedControl = True
                            
                        elif keyPressed == 'none':
                            if inputAngle == 'steerKeyboardAngle' :
                                #when stopping keyboard control, come back to last angle control
                                inputAngle = lastInputAngle
                            strText=''
                            turn_angle = 0
                            
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

                
                # get time and manage steer from keyboard
                timeNow = time.time()
                if timeNow > (lastSteerKeyboardTime + STEERING_KEYBOARD_SAMPLING_TIME):
                    # it s time to update steer command
                    steerKeyboardAngle += turn_angle
                    if lastKeyPressed == 'right' or lastKeyPressed == 'left':
                        steerAngleUpdate = True
                    lastSteerKeyboardTime = timeNow

                ############### Control the Car with all the input we can have ####################
                if steerAngleUpdate == True :
                    # send control command  (sampling is done in each module 
                    if inputAngle == 'steerKeyboardAngle' :
                        self.steerAngleCommand = steerKeyboardAngle
                    elif inputAngle == 'steerPathAngle' :
                        self.steerAngleCommand = steerPathAngle
                    elif inputAngle == 'steerNnAngle':
                        self.steerAngleCommand = steerNnAngle
                    elif inputAngle == 'steerRoadLineAngle':
                        self.steerAngleCommand = steerRoadLineAngle
    
                        
                    if self.steerAngleCommand < MIN_ANGLE:
                        self.steerAngleCommand = MIN_ANGLE
                    elif self.steerAngleCommand > MAX_ANGLE:
                        self.steerAngleCommand = MAX_ANGLE
                                            
                    print ('Angle =',self.steerAngleCommand)
                    
                    # Send command
                    self.sctCarSteering.cmd_q.put(ClientCommand(
                        ClientCommand.SEND, ('STEER_ANGLE', self.steerAngleCommand)))
                    
                    steerAngleUpdate = False
    
                    # reset the steerKeyboard angle to the latest angle
                    # to start from it if correction needed
                    steerKeyboardAngle = self.steerAngleCommand
                               
                ############### See if we want to record something ####################                        
                        
                #See now if we have to record or not the frame into vstack memory
                timeNow = time.time()
                if (record == 1):
                    #start recording time if all condition are met
                    if recordTime == 0:
                        if pathControlCommand != 'NONE':
                            recordTime = time.time()
                        else:
                            print 'WARNING : to record please select a road use case with l,r,i,c...'
                            record = 0
                            continue

                    
                    #check if this is time to record a frame
                    if  timeNow > (lastFrameTime + FPS_RECORD_TIME):
                        #record image and labels
                        saved_frame += 1          
                        angle = self.steerAngleCommand

                        #expand dimension of image to allow the stack later
                        temp_array = np.expand_dims(image2nn, axis=0)
                        #record for main NN which select the road use case one 
                        image_array_correction = np.vstack((image_array_correction, temp_array))
                        NNroadLabel_array_correction = np.vstack((NNroadLabel_array_correction, np.array([NNroadLabel2Num(pathControlCommand)])))
                        #no need to save angle for this NN but just in case
                        steerAngle_array_correction = np.vstack((steerAngle_array_correction, np.array([angle])))

                        lastFrameTime = timeNow
                else:
                    #record the time if recorTime exist
                    if recordTime != 0:
                        totalRecordTime += (timeNow - recordTime)
                        recordTime = 0
                        
                        
                #############################################in any case send speed if new one exist
                
                if gpsEnable == False:
                    pwmSpeed = int((pathControlSpeed * MAX_CAR_SPEED_COMMAND)/ MAX_CAR_SPEED )
                
                #boundaries check
                if pwmSpeed > MAX_CAR_SPEED_COMMAND:
                    pwmSpeed = MAX_CAR_SPEED_COMMAND
                elif pwmSpeed < 0 :
                    pwmSpeed = 0                 

                #send the speed to car
                if pwmSpeed != lastpwmSpeed :
                    self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED', pwmSpeed)))
                    lastpwmSpeed = pwmSpeed
                        
        finally:
            if totalRecordTime !=0 or record==1:            
                # Build file name based on date/time
                timestr  = time.strftime("%Y%m%d-%H%M%S")
                fileName_one = 'training_data/CorrectionSet_ONE_' + timestr
                #fileName_two = self.dataFolder + '/trainingSet_TWO' + timestr
                
                if len(image_array_correction) > 1:
                    # save training images for NN one (main NN to select road case)
                    image_array_correction = image_array_correction[1:, :]
                    steerAngle_array_correction = steerAngle_array_correction[1:, :]
                    NNroadLabel_array_correction = NNroadLabel_array_correction[1:, :]
                    print 'road Label distribution ONE'
                    print np.histogram(NNroadLabel_array_correction,bins=range(0,5))
                    print 'image shape       =', image_array_correction.shape
                    print 'steer angle shape =', steerAngle_array_correction.shape
                    print 'road label shape  =', NNroadLabel_array_correction.shape
                    # save training data as a numpy file
                    np.savez(fileName_one, train=image_array_correction, steerAngle_label_array=steerAngle_array_correction, NNroadUseCase_label_array=NNroadLabel_array_correction)
                    print "File Name :",fileName_one

                print 'Total frame:', total_frame
                print 'Saved frame:', saved_frame , ' in ', totalRecordTime, ' seconds'
                print 'Dropped frame', total_frame - saved_frame            

            print 'ending Deep Driver'

            # stop and close all client and close them
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND', 'stop')))
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND', 'home')))
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctCarSteering.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctPathControlSteeringClientThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctPathControlSteeringClientThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctPathControlCommandClientThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctNnSteerThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctNnSteerThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctRoadLineSteerThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctRoadLineSteerThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            # and make sure all of them ended properly
            self.sctVideoCarStream.join()
            self.sctCarSteering.join()
            self.sctCarSensor.join()
            self.sctPathControlSteeringClientThread.join()
            self.sctPathControlCommandClientThread.join()
            self.keyboardThread.join()
            self.sctNnSteerThread.join()
            self.sctRoadLineSteerThread.join()
            self.sctGps.join()
            print 'Deep Driver Done'

if __name__ == '__main__':
    # create Deep drive thread and strt
    DDriveThread = DeepDriveThread()
    DDriveThread.name = 'DDriveThread'

    # start
    DDriveThread.start()

    DDriveThread.join()
    print 'end'

import cv2
import numpy as np
import threading
import Queue
import time

#libraries from deep drive
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from KeyboardThread import *
from gpsClientThread import *
from perceptionClientThread import *
from graphSlamThread import *
from pathControlClientThread import *



####################################### Deep Drive Thread ##############################

class pathControl(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)


        # create Keyboard Thread
        self.keyboardThread = keyboardThread(1960,0)
        self.keyboardThread.name = 'pathControl_Kb'
        self.keyboardThread.start()
        
        #create GPS path control server to provide steer Angle based Graphslam and GPS pos
        self.srvPathControlSteering = serverThread()
        self.srvPathControlSteering.name = 'srvPathControlSteering'
        self.srvPathControlSteering.start()
        
        #server which provide moveOrder to the car such as :
        #FORWARD, STOP, SPEED=XXX, IDLE, STRAIGHT, LEFT, RIGHT, CHANGE_RIGHT, CHANGE_LEFT
        self.srvPathControlCommand = serverThread()
        self.srvPathControlCommand.name = 'srvPathControlCommand'
        self.srvPathControlCommand.start()

        
        #create Gps Thread that receive image from camera and detect cars
        self.sctGps = gpsClientThread()
        self.sctGps.name = 'GpsClientPathControl'
        self.sctGps.start()
        
        #create Gps Thread that receive image from camera and detect cars
        self.sctPerception = perceptionClientThread()
        self.sctPerception.name = 'PerceptionClientPathControl'
        self.sctPerception.start()

        #create GraphSlam Thread to compute slam
        self.graphSlamThread = graphSlamThread()
        self.graphSlamThread.name = 'GraphSlamThread'
        self.graphSlamThread.start()
        


    def ConnectClient(self):
        # loop until all client connected
        sctGpsConnected = False
        sctPerceptionConnected = False
        gpsServerFixConnected = False
        PathControlCommandServerConnected = False
        pathControlSteeringServerConnected = False
               

        # launch connection thread for all client                    
        if gpsEnable == True:
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_GPS_FIX_SERVER))
            
        if perceptionEnable == True:
            self.sctPerception.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_PERCEPTION_SERVER))
            
                
        if pathControlSteeringEnable == True:
            self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_PATH_CONTROL_STEERING_SERVER))
            
        if pathControlCommandEnable == True:
            self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_PATH_CONTROL_COMMAND_SERVER))


        while ((sctGpsConnected != gpsEnable) or
                (sctPerceptionConnected != perceptionEnable)):
            # wait for .5 second before to check
            time.sleep(0.5)
                    

            if (sctGpsConnected != gpsEnable):
                try:
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sctGpsConnected=True
                        print 'Gps fix client connected'
                except Queue.Empty:
                    print 'Gps fix Client not connected' 
                    

            if (sctPerceptionConnected != perceptionEnable):
                try:
                    reply = self.sctPerception.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sctPerceptionConnected=True
                        print 'Perception  client connected'
                except Queue.Empty:
                    print 'Perception  Client not connected' 
                    
            try:
                reply = self.keyboardThread.reply_q.get(False)
                if reply.type == ClientReply.SUCCESS:
                    if reply.data == 'exit':
                        return False
            except Queue.Empty:
                time.sleep(0.5)
                pass
                    

    def run(self):
        lastKeyPressed = 0
        gpsPosition = (0,0)
        gpsSpeed=0
        #path control command
        pathControlCommand = 'IDLE'
        lastPastControlCommand = 'NONE'
        
        #path control speed
        initialCarSpeed = INITIAL_CAR_SPEED
        pathControlSpeed = initialCarSpeed
        lastPastControlSpeed = 0
        lastCommandTime = time.time()
        lastVideoTime = time.time()
        lastnoObjectTime=time.time()

        objectName=''
        distObj=0
        
        speedBeforeStop = 0 #0 means no speed before stop
        noObjCounter = 0

        if graphSlamEnable:
            cv2.namedWindow('GraphSlam')
            
        cv2.namedWindow('PathControlVision')
        cv2.moveWindow('PathControlVision', 1300, 0)
 
        steerPathAngle = 0 #angle coming from Position module and graph slam ...
        
        stopState = 'None'
        lastStopState = stopState
        # initial steer command set to stop
        try:
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread and sub Thread for path Control '
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            self.sctPerception.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))

            

            while True:
                ################# Manage IMAGE from car Camera###############
                timeNow = time.time()
                if timeNow > lastVideoTime + 0.1:
                    
                    
                    # get image
                    i = cv2.imread('frame.jpg')
                    
                    if i != None:
                        lastVideoTime = timeNow

                        cv2.putText(i, 'Path SteerAngle  = ' + str(steerPathAngle), (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'GPS position =' + str(gpsPosition), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'pathControlCommand =' + lastPastControlCommand, (0,45), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'pathControlSpeed =' + str(lastPastControlSpeed), (0,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1) 
                        cv2.putText(i, 'GPS speed mm/s  =' + str(gpsSpeed), (0,75), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        if objectName !='' :
                            cv2.putText(i, 'Object  =' +objectName+', '+ str(distObj), (0,90), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                            objectName = ''
                        
                        #show green line steering angle
                        showLabel(steerPathAngle,'PathControlVision', i)
                        
                        
                        #display the car vision and info
                        cv2.imshow('PathControlVision', i)
                            
                        # check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                

                ################# Handle fix from GPS ###############
                try:
                    #check if car gps fix has been detected
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #fix has been receive . process it
                        vPosX, vPosY, vGpsSpeed, vOrient, vName = reply.data.split(',')
                        #print 'Received FIX for vehicle ',vName, ' x=', vPosX, ' y=', vPosY
                        if (vName == VEHICULE_NAME):
                            ############### SEND value to Graph slam ############
                            gpsPosition = (int(vPosX),int(vPosY),int(vGpsSpeed),float(vOrient))
                            gpsSpeed = int(vGpsSpeed)
                            
                            #send GPS info to slam if feqture enable
                            if int(vGpsSpeed) > 50:
                                if graphSlamEnable == True:
                                    self.graphSlamThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, gpsPosition))

                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass
                
                
                 ################# Handle object recievde from perception ###############
                timeNow = time.time() 
                try:
                    #check if car gps fix has been detected
                    reply = self.sctPerception.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #Object  has been receive . process it
                        objectName,distObj = reply.data.split(',')
                        print 'Received object',objectName,distObj
                        
                        #we received an object so reset the noObjCounter
                        noObjCounter=0
                        lastnoObjectTime = timeNow

                        #take action depending on object detected
                        if (objectName == 'stop'):
                            if (int(distObj) <= 30 ):
                                #only stop the car if we are not already in a procedure to stop it
                                if stopState == 'None':
                                    stopState = 'firstEntry'
                                    #reset speed for sure and wait at least 4 s
                                    speedBeforeStop = pathControlSpeed
                                    pathControlSpeed = 0
                                    lastStopTime = timeNow 
                                           
                except Queue.Empty:
                    #noObjCounter is incremented every second 
                    if timeNow > lastnoObjectTime + 1.0:
                        noObjCounter+=1
                        lastnoObjectTime=timeNow

                #manage the state for speed control:
                if stopState=='firstEntry':
                    #wait for 4 s before to restart wathever state 
                    if timeNow > lastStopTime + 4.0:
                        #we do the stop during 4 second and then restart the car
                        pathControlSpeed = speedBeforeStop
                        #disable stop in orde to let the car possibility to start
                        stopState='disableStop'
                        #reset last time to start from it
                        lastStopTime = timeNow
                        
                    elif objectName == 'car':
                        #if car detected restart for 4 second the counter
                        lastStopTime = timeNow
                
                #in this state we stay 4 second without allowing to stop
                if stopState=='disableStop':
                    if timeNow > lastStopTime + 4.0:
                        #during this extra time we do avoid to do the stop again even if detected 
                        lastStopTime = timeNow
                        stopState = 'None'

                    
                        
                if lastStopState != stopState:
                    print 'stopstat= ',stopState
                    lastStopState=stopState
                    
                    
                #############################Handle gpraphSlam ###############
                #check now if the slamGraph Thread posted a new steeringAngle
                try:
                    # try to see if data ready
                    reply = self.graphSlamThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        steerPathAngle = int(reply.data)
                        print 'receive Angle from graphSlamThread = ' + str(steerPathAngle)
                         
                        # Send Angle to server
                        self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, steerPathAngle))
                                                            
                        #show graphslam result in main thread
                        if self.graphSlamThread.myUserInterface.GraphSlamImg != None:
                            cv2.imshow('GraphSlam',self.graphSlamThread.myUserInterface.GraphSlamImg)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    else:
                        print 'Error getting GraphSlam data :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass


                ############################# Handle pathControl Command 5speed + control ####################################     
                    
                ############################# Handle pathControl Speed ####################################
                #no tuning of speed for the moment, just return INITIAL one
                
                if pathControlSpeed < 0:
                    pathControlSpeed = 0
                elif pathControlSpeed > MAX_CAR_SPEED:
                    pathControlSpeed = MAX_CAR_SPEED

                if pathControlSpeed != lastPastControlSpeed:
                    lastPastControlSpeed = pathControlSpeed
                    print 'send speed =',pathControlSpeed
                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, 'SPEED='+str(pathControlSpeed)))
 
                                    
                ################## first send the control command if necessary #####################
                # for the moment only set pathControlCommand to say IDLE or STRAIGHT depending
                pathControlCommand = gps2MainRoadLabel(gpsPosition)
                             
                if pathControlCommand != lastPastControlCommand:
                    lastPastControlCommand = pathControlCommand
                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, pathControlCommand))

            
            
             

                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        # new keyboard input found
                        keyPressed = reply.data
                        strText = keyPressed

                        if keyPressed == 'exit':
                            # get out of the loop
                            break
                        
                        elif keyPressed == 'help':
                            strText = 'Key:Q,P(lot),+,-'
                        
                        elif keyPressed == 'minus':
                            pathControlSpeed -= 50
                            initialCarSpeed -=50
                            
                        elif keyPressed == 'plus':
                            pathControlSpeed += 50
                            initialCarSpeed+=50
                            
                        elif keyPressed == 'PATH_CONTROL':
                            strText='Plot Slam'
                            #request with send to plot the graph
                            self.graphSlamThread.cmd_q.put(ClientCommand(ClientCommand.SEND, ''))
                            
                        elif keyPressed == 'none':
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



                ######################## make sure to send command on all client because client could be new 
                # make sure to send the command at least every second
                timeNow = time.time()
                if timeNow > (lastCommandTime + 1.0):

                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, pathControlCommand))
                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, 'SPEED='+str(pathControlSpeed)))
                    self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, steerPathAngle))
                    
                    lastCommandTime = timeNow


                       
        finally:
            print 'ending path control Thread'

            # stop and close all client and close them
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctPerception.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctPerception.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.graphSlamThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
                                                       
            # and make sure all of them ended properly
            self.keyboardThread.join()
            self.srvPathControlSteering.join()
            self.sctGps.join()
            self.sctPerception.join()
            self.srvPathControlCommand.join()
            self.graphSlamThread.join()
            print 'Path control Done'

if __name__ == '__main__':

    pathControl = pathControl()
    pathControl.name = 'pathControl'

    # start
    pathControl.start()

    pathControl.join()
    print 'end'

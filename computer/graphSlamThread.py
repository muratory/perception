import threading
import Queue
import struct
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from graph_slam import *
import time


class graphSlamThread(commonThread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(graphSlamThread, self).__init__()
        self.steeringAngle = 0

        if FUNCTIONAL_MODE == MODE_REAL:
            self.gpsFromDeepDrive = True   # Set it to TRUE to run in real Deep Drive Hard envt.
        else:
            self.gpsFromDeepDrive = False    # Set it to TRUE to run in real Deep Drive Hard envt.

        self.initialized = False
        
        
    def _handle_SEND(self, cmd):
        #refresh graphslam matplotlib
        plt.pause(0.5)

 
    def _handle_RECEIVE(self, cmd):
        newPosition = cmd.data
        print "handle New Position in graphSlam",str(newPosition)

        steeringAngle = self.computeNextSteeringAngle(newPosition)
        #send in reply queue the computation
        self.reply_q.put(self._success_reply(steeringAngle))


    def computeNextSteeringAngle(self,newPosSpeedOrient):
        global sensinggpsx
        global sensinggpsy

        newPosition = (newPosSpeedOrient[0],newPosSpeedOrient[1])
        newSpeed = newPosSpeedOrient[2]
        orientation = newPosSpeedOrient[3]

        if self.initialized == False:
            print "Graph Slam starting..."
            
            self.N = 0
            self.err = 0.0
            self.previoussteeringRadian = 0.0
            self.speed = 0.0
            self.prev_speed = 0.0
            self.prev_measurement = [0.,0.]
            self.lastPosTable = []

            # DEBUG
            if self.gpsFromDeepDrive == False:
                newPosition = [424, 104]  # c'est le pt 10


            # Init world and setting, starting with current robot position given by GPS
            self.myUserInterface, self.mypath = graph_slam_init(self.gpsFromDeepDrive, newPosition)

            self.myrobot, self.filter, self.myslam = run_init(self.myUserInterface,
                                                              self.mypath.spath,
                                                              self.myUserInterface.steering_noise,
                                                              self.myUserInterface.distance_noise,
                                                              self.myUserInterface.measurement_noise,
                                                              self.myUserInterface.init,
                                                              self.myUserInterface.initorient)

            self.initialized = True
            self.speed = self.myUserInterface.speed
            self.prev_speed = 0 # 1st call must be with speed 0 !!!
 
        
        sensinggpsx.append(newPosition[0]/10.0)
        sensinggpsy.append(newPosition[1]/10.0)
        '''
        #compensate GPS lag or other known GPS problem with projection of the distance run by the robot
        #compute the distance run during the lag of GPS 
        distanceGpsLag = newSpeed*GPS_LAG_TIME
        #we need to advance the position of the GPS by this distance :
        xGpsCorrected = int(newPosition[0] + cos(orientation)*distanceGpsLag)
        yGpsCorrected = int(newPosition[1] + sin(orientation)*distanceGpsLag)
        newPosition = (xGpsCorrected,yGpsCorrected)
        
        '''
        
        # Sense ! from Deep Drive GPS or from Simulation
        if self.gpsFromDeepDrive == True:
            measurements = [float(newPosition[0])/float(SHRINK_FACTOR), float(newPosition[1])/float(SHRINK_FACTOR), orientation]

        else:
            # Sense: My robot Sense its environment
            measurements = self.myrobot.sense(self.myUserInterface.grid, self.myUserInterface.sensing_cfg)

        if self.gpsFromDeepDrive == True:
        # Estimate Speed
            self.lastPosTable, self.prev_speed = self.myrobot.speed_estimator(self.lastPosTable, measurements)
            print '-----Estimated speed =', self.prev_speed
        else:
            # Speed is constant and is given by self.myUserInterface.speed
            pass
        
        # Estimate new steering...
        self.myrobot, \
        self.filter, \
        self.myslam, \
        steeringRadian = run_asfunction(self.myUserInterface,
                                        self.mypath.spath,
                                        self.myrobot,
                                        self.myslam,
                                        self.filter,
                                        self.N,
                                        measurements,
                                        self.prev_speed,
                                        self.previoussteeringRadian)




        # Simulate the move of my robot
        if self.gpsFromDeepDrive == False:
           self.myrobot = self.myrobot.move(steeringRadian, self.myUserInterface.speed, True)  # move my robot accordingly considering noise
        self.previoussteeringRadian = steeringRadian

        self.err += (self.myrobot.cte ** 2)
        self.N += 1

            
        self.myUserInterface.log_along_run(self.myrobot, self.myslam, self.myrobot.estimate)
        # Log and plot along the run...
        self.myUserInterface.plot_track(self.myrobot, self.myslam, self.myrobot.estimate)


        if self.myrobot.check_goal(self.myUserInterface.goal,self.myUserInterface.goal_distance) == True:
            
            # Final Plot.
            # ---------------
            self.myUserInterface.textual_final_plot(self.myrobot, self.myslam, self.N, self.err, 0)
            '''
            self.myUserInterface.plot_track(self.myrobot, self.myslam, self.myrobot.estimate)
            '''
            self.initialized = False


        print 'Step:', self.N, 'Steering Angle-Deg computed ', degrees(self.steeringAngle)
        
        #Convert steering to degrees
        self.steeringAngle = degrees(steeringRadian)
        
        return self.steeringAngle
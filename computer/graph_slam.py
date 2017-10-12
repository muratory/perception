# ------------------------------------------------------------------------------
#              Simulated Robot - Python 2.7xxx
# -----------------------------------------------------------------------------
#  Most of the source comes from the following Udacity Training
#
#       Artificial Intelligence for Robotics - Programming a Robotic Car
#       https://classroom.udacity.com/courses/cs373/lessons/48739381/concepts/487350240923#
#
# Adapted by Alain Vallauri, alain.vallauri@intel.com,
#   Integrated Landmark sensing
#   Integrated the Online Graph SLAM
#   Added a complete UI to parametrize the simulation and plot
#
# -----------------------------------------------------------------------------
# -------------  Code content  --------------------------
# main: this is our main routine, 
#       Creates an empty path plan from the grid
#       Search the path to the goal using A* 
#       smooth the path
#       run, so perform the move, see details below
#
# Class plan: motion planning
#   methods:
#       init > creates an empty plan
#       make_heuristic > make heuristic function for a grid
#       astar > A* for searching a path to the goal
#       smooth > smoothing function to smooth the path
#
# Class robot: simulated robot
#   methods
#       init > creates robot and initializes location/orientation to 0, 0, 0
#       set > sets a robot coordinate
#       set_noise > sets the noise parameters
#       check > checks of the robot pose collides with an obstacle, or is too far outside the plane
#       move > !!!! CELLE LA JE LA CONNAIS PAS !!!!!
#       sense > similar to GPS in a car but with substancial noise. NO LANDMARK !!!!!!!
#       measurement_prob >
#
# Class particles: particle filter (position estimation)
#       init > creates particle set with given initial position
#       get_position > extract position from a particle set as an average of all alive particles
#       move > motion of the particles
#       sense > sensing and resampling
#
# Control program
#   run: runs control program for the robot, so perform the move
#       loop until goal is reached
#         Estimate current position as the average of particles position
#         Appli PID: calculate steer based on CTE considering multi segment
#         move my robot
#         move all particles alive
#         My robot senses its environment: not Landmark as such. Robot Sense returns a noisy position x,y like GPS would
#         Applies the sense to particles
#
# Class matrix: math class for matrix operations
# Class slamfilter: online graph slam
# Class userinteraction: User Interface. Input parameters and output plots.
#
# =======================================================================================

from math import *
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from commonDeepDriveDefine import *

import pickle


    
# ----------------------------------------
TIMEOUT      = 200000

MODE_SIMULATION = "Simulation"
MODE_REAL = "Real Deep Drive"
FUNCTIONAL_MODE = MODE_REAL

MOTION_MODEL_BYPASSED = 'Motion model: none, Bypassed'  #Bypass the bycle model in move function
MOTION_MODEL_BICYCLE = 'Motion model: bicycle'
MOTION_MODEL = MOTION_MODEL_BICYCLE

FILTER_ACTIVATED     = 'Particle Filter: Activated'
FILTER_NOT_ACTIVATED = 'Particle Filter: NOT Activated'
PARTICLE_FILTER = FILTER_ACTIVATED

PID_CTE    = 'PID Based on CTE'
PID_ORIENT = 'PID Based on Orientation'
SELECTED_PID = PID_ORIENT #PID_ORIENT  #PID_ORIENT#

N_PARTICLES = 150
VisibleRange = 3.0   # 1.0 to ...

# Matrix based circuit
# Noise setting, The reference Noise value to come back to:
# steering_noise    = 0.01 or 0.1
# distance_noise    = 0.03
# measurement_noise = 0.3
ROBOT_LENGTH = 0.5
Speedvalue = 0.1 #0.1 normafl value  # 0.1 #0.05 to 0.3
P_GAIN = 2.0 #normal value #1.0 for deep drive#
I_GAIN = 0.0 #normal value #1.0 for deep drive#
D_GAIN = 6.0 #normal value #0.4 for deep drive
P_GAIN_ORIENT = 0.6
I_GAIN_ORIENT = 0.0
D_GAIN_ORIENT = 0.2
STEERING_NOISE = 0.05
DISTANCE_NOISE = 0.03
MEAST_NOISE = 0.3
GOAL_DISTANCE = 0.5
OOL_MARGIN = 30  #Out of limit detection margin

if FUNCTIONAL_MODE == MODE_REAL:
    # Deep Drive circuit
    # Best known parameters for real life:
    #  Orient PID, Gains: P=0.6, D=0.2
    #  gps paper on front, N=150
    #  Speed = 45
    #  steering_noise = 0.2
    # distance_noise = 2.0
    # measurement_noise = 0.5
    #
    
    ROBOT_LENGTH_DEEP_DRIVE = 40.0  # 2 means 20 pixels in DDrive, 16cm.
    Speedvalue_FOR_DEEPDRIVE = 6.0 #for Deep drive 2.0
    P_GAIN_FOR_DEEPDRIVE = 0.5 #2.0 normal value #0.8 for deep drive#
    I_GAIN_FOR_DEEPDRIVE = 0.0
    D_GAIN_FOR_DEEPDRIVE = 0.0 #6.0 normal value #0.4 for deep drive
    P_GAIN_ORIENT_FOR_DEEPDRIVE = 0.6
    I_GAIN_ORIENT_FOR_DEEPDRIVE = 0.2
    D_GAIN_ORIENT_FOR_DEEPDRIVE = 0.2
    SHRINK_FACTOR = 10.0
    RANGE_X = 2500/SHRINK_FACTOR #400/SHRINK_FACTOR
    RANGE_Y = 5000/SHRINK_FACTOR #700/SHRINK_FACTOR
    STEERING_NOISE_FOR_DEEPDRIVE = 0.3  # 0.2 for real DDrive
    DISTANCE_NOISE_FOR_DEEPDRIVE = 0.5  # 2.0 for DDrive mais marche pas en simu !!
    MEAST_NOISE_FOR_DEEPDRIVE = 2.0    # 1.0 for deep drive means 10 pixels
    GOAL_DISTANCE_FOR_DEEPDRIVE = 20.0
else: #mode is then simulation
    ROBOT_LENGTH_DEEP_DRIVE = 16.0  # was 2.0, 2 means 20 pixels in DDrive, 16cm.
    Speedvalue_FOR_DEEPDRIVE = 8.0 # was 1.0
    P_GAIN_FOR_DEEPDRIVE = 2.0 #2.0 normal value #0.8 for deep drive#
    I_GAIN_FOR_DEEPDRIVE = 0.0
    D_GAIN_FOR_DEEPDRIVE = 0.4 #6.0 normal value #0.4 for deep drive
    P_GAIN_ORIENT_FOR_DEEPDRIVE = 0.6
    I_GAIN_ORIENT_FOR_DEEPDRIVE = 0.6 # was 0.6
    D_GAIN_ORIENT_FOR_DEEPDRIVE = 0.6 # was 0.2
    SHRINK_FACTOR = 10.0
    RANGE_X = 2500/SHRINK_FACTOR #was 400/SHRINK_FACTOR
    RANGE_Y = 5000/SHRINK_FACTOR #was 700/SHRINK_FACTOR
    STEERING_NOISE_FOR_DEEPDRIVE = 0.1  #
    DISTANCE_NOISE_FOR_DEEPDRIVE = 0.24 # was 0.03
    MEAST_NOISE_FOR_DEEPDRIVE = 4.0     # was 0.5
    GOAL_DISTANCE_FOR_DEEPDRIVE = 8.0 # was 1.0

# smooth algo weight
WEIGHT_DATA   = 0.1
WEIGHT_SMOOTH = 0.2

# =======================================================================================

PLOT_END = 'Plot at the end of execution'
PLOT_ALONG = 'Plot along execution'

NOISE_ON   = 'With noise'
NOISE_OFF  = 'No noise'

SENSING_LM = 'Sensing the LM'
SENSING_GPS = 'Sensing from GPS'

ODO_TRUE = 'Odometer from True Robot move'
ODO_GPS = 'Odometer from GPS'  # Not implemented here

SLAM_ON = 'Slam ON'
SLAM_OFF = 'Slam OFF'

GRID_TINY = 'Grid: Tiny 3*2'
GRID_SMALL = 'Grid: Small grid 5*6'
GRID_MEDIUM = 'Grid: Medium 10*12'
GRID_LARGE = 'Grid: Large 14*12'
GRID_EXTRALARGE = 'Grid: XLarge 22*19'
GRID_FROM_GPSMAP = 'Grid: From GPS MAP'

sensinglmx = [] # LM sensing plotting
sensinglmy = [] # LM sensing plotting
sensinggpsx = [] # GPS sensing plotting
sensinggpsy = [] # GPS sensing plotting

particlex = [] # particle plotting
particley = [] # particle plotting

# ================================ FCT TO PLOT ============================================

#increase size of the image by this factor
RESIZE_FACTOR = 2

circuit_zone_of_interest_file = "gps_camera_calib_folder/undistort_gps_calibration.tpz"
scale_XY = None

estimateWithoutLag = None


def cmToPixelX(x):
    return int((x*10.0*RESIZE_FACTOR)/scale_XY[0])

def cmToPixelY(y):
    return int((y*10.0*RESIZE_FACTOR)/scale_XY[1])


def plotPoint(img,point,color,label=''):
    #get x,y on image in pixel distance
    pxl = (cmToPixelX(point[0]),cmToPixelY(point[1]))
    #draw point which is a circle
    cv2.circle(img, pxl, 3, color,-1)
    #add label if necessary
    if label != '':
        cv2.putText(img, label, (pxl[0]+3,pxl[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def plotArrowedLine(img,pointA,pointB,color,label=''):
    #get x,y on image in pixel distance
    pxlA = (cmToPixelX(pointA[0]),cmToPixelY(pointA[1]))
    pxlB = (cmToPixelX(pointB[0]),cmToPixelY(pointB[1]))
    #draw point which is a circle
    cv2.arrowedLine(img, pxlA,pxlB, color,1)
    #add label if necessary
    if label != '':
        cv2.putText(img, label, (pxlA[0]+2,pxlA[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
def plotCross(img,center,color,label=''):
    #cross size in pixel
    CROSS_SIZE = 2
    if center[0] < CROSS_SIZE or center[1] < CROSS_SIZE:
        return
        
    #build 2 crossed line with a center 
    pxlAl1 = (cmToPixelX(center[0]-CROSS_SIZE),cmToPixelY(center[1]+CROSS_SIZE))
    pxlBl1 = (cmToPixelX(center[0]+CROSS_SIZE),cmToPixelY(center[1]-CROSS_SIZE))
    pxlAl2 = (cmToPixelX(center[0]-CROSS_SIZE),cmToPixelY(center[1]-CROSS_SIZE))
    pxlBl2 = (cmToPixelX(center[0]+CROSS_SIZE),cmToPixelY(center[1]+CROSS_SIZE))
    
    #draw point which is a circle
    cv2.line(img, pxlAl1,pxlBl1, color,1)
    cv2.line(img, pxlAl2,pxlBl2, color,1)
    #add label if necessary
    if label != '':
        cv2.putText(img, label, (pxlA[0]+2,pxlA[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def plotSeriesPoint(img,xs,ys,color):
    for i in range(0,len(xs)):
        plotPoint(img,(xs[i],ys[i]),color)
        #todo add label to the barycenter 
        
def plotLineTeta(img,point,teta,length,color,label):
    pxlA = (cmToPixelX(point[0]),cmToPixelY(point[1]))
    #projection of pointB in teta direction
    projx = pxlA[0] + cos(teta)*cmToPixelX(length)
    projy = pxlA[1] + sin(teta)*cmToPixelY(length)
    pxlB = (int(projx),int(projy))
    
    #draw point which is a circle
    cv2.arrowedLine(img, pxlA,pxlB, color,1)
    #add label if necessary
    if label != '':
        cv2.putText(img, label, (pxlA[0]+2,pxlA[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)    
    


# =======================================================================================

class plan:

    # --------
    # init: 
    #    creates an empty plan
    #

    def __init__(self, grid, init, goal, cost = 1):
        self.cost = cost
        self.grid = grid
        self.init = init
        self.goal = goal
        self.make_heuristic(grid, goal, self.cost)
        self.path = []
        self.spath = []

    # --------
    #
    # make heuristic function for a grid
        
    def make_heuristic(self, grid, goal, cost):
        self.heuristic = [[0 for row in range(len(grid[0]))] 
                          for col in range(len(grid))]
        for i in range(len(self.grid)):    
            for j in range(len(self.grid[0])):
                self.heuristic[i][j] = abs(i - self.goal[0]) + \
                    abs(j - self.goal[1])

    # ------------------------------------------------
    # 
    # A* for searching a path to the goal
    #
    #

    def astar(self):
        if self.heuristic == []:
            raise ValueError, "Heuristic must be defined to run A*"

        # internal motion parameters
        delta = [[-1,  0], # go up
                 [ 0, -1], # go left
                 [ 1,  0], # go down
                 [ 0,  1]] # do right

        # open list elements are of the type: [f, g, h, x, y]

        closed = [[0 for row in range(len(self.grid[0]))] 
                  for col in range(len(self.grid))]
        action = [[0 for row in range(len(self.grid[0]))] 
                  for col in range(len(self.grid))]

        closed[self.init[0]][self.init[1]] = 1

        x = self.init[0]
        y = self.init[1]
        h = self.heuristic[x][y]
        g = 0
        f = g + h

        open = [[f, g, h, x, y]]

        found  = False # flag that is set when search complete
        resign = False # flag set if we can't find expand
        count  = 0

        while not found and not resign:

            # check if we still have elements on the open list
            if len(open) == 0:
                resign = True
                print '###### Search terminated without success'
                
            else:
                # remove node from list
                open.sort()
                open.reverse()
                next = open.pop()
                x = next[3]
                y = next[4]
                g = next[1]

            # check if we are done

            if x == self.goal[0] and y == self.goal[1]:
                found = True
                # print '###### A* search successful'

            else:
                # expand winning element and add to new open list
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(self.grid) and y2 >= 0 \
                            and y2 < len(self.grid[0]):
                        if closed[x2][y2] == 0 and self.grid[x2][y2] == 0:
                            g2 = g + self.cost
                            h2 = self.heuristic[x2][y2]
                            f2 = g2 + h2
                            open.append([f2, g2, h2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i

            count += 1

        # extract the path

        invpath = []
        x = self.goal[0]
        y = self.goal[1]
        invpath.append([x, y])
        while x != self.init[0] or y != self.init[1]:
            x2 = x - delta[action[x][y]][0]
            y2 = y - delta[action[x][y]][1]
            x = x2
            y = y2
            invpath.append([x, y])

        self.path = []
        for i in range(len(invpath)):
            self.path.append(invpath[len(invpath) - 1 - i])

    # ------------------------------------------------
    # 
    # this is the smoothing function
    #
    # newpath[i][j] += weight_data (path[i][j] - newpath[i][j])
    #                  + weight_smooth (newpath[i-1][j] + newpath[i+1][j]
    #                  - 2.0 * newpath[i][j])
    #
    # ... les termes if i >= ou <= sont nouveau, un smoothing d'ordre 2, ???
    #
    # spath is the newpath
    
    def smooth(self, weight_data = 0.1, weight_smooth = 0.1, 
               tolerance = 0.000001):

        if self.path == []:
            raise ValueError, "Run A* first before smoothing path"

        self.spath = [[0 for row in range(len(self.path[0]))] \
                           for col in range(len(self.path))]
        for i in range(len(self.path)):
            for j in range(len(self.path[0])):
                self.spath[i][j] = self.path[i][j]

        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(self.path)-1):
                for j in range(len(self.path[0])):
                    aux = self.spath[i][j]
                    
                    self.spath[i][j] += weight_data * \
                        (self.path[i][j] - self.spath[i][j])
                    
                    self.spath[i][j] += weight_smooth * \
                        (self.spath[i-1][j] + self.spath[i+1][j] 
                         - (2.0 * self.spath[i][j]))
                    if i >= 2:
                        self.spath[i][j] += 0.5 * weight_smooth * \
                            (2.0 * self.spath[i-1][j] - self.spath[i-2][j] 
                             - self.spath[i][j])
                    if i <= len(self.path) - 3:
                        self.spath[i][j] += 0.5 * weight_smooth * \
                            (2.0 * self.spath[i+1][j] - self.spath[i+2][j] 
                             - self.spath[i][j])
                
            change += abs(aux - self.spath[i][j])

# =======================================================================================
# this is the robot class
#   init
#   set
#   set_noise
#   check_collision
#   check_goal
#   move
#   sense
#   sense_landmarks
#   measurement_prob
#   multi_measurement_prob
#   Gaussian
# =======================================================================================
class robot:

    # --------
    # init: 
    #	creates robot and initializes location/orientation to 0, 0, 0
    #

    def __init__(self, length):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0 #random.random() * 2.0 * pi #
        self.steering = 0.0 
        self.length = length
        self.steering_noise    = 0.0
        self.distance_noise    = 0.0
        self.measurement_noise = 0.0
        self.num_collisions    = 0
        self.num_steps         = 0
        self.cte = 0.0
        self.teta = 0.0
        self.tetatab = []
        self.index = 0
        self.estimate = [0.,0.,0.]
        self.prev_estimate = [0.,0.,0.]  # previous estimated X and Y position
    # --------
    # set: 
    #	sets a robot coordinate
    #

    def set(self, new_x, new_y, new_orientation):

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation) % (2.0 * pi)
        self.estimate = [new_x,new_y,self.orientation]
        self.prev_estimate = [new_x,new_y,self.orientation]  #[self.x, self.y]

    # --------
    # set_noise: 
    #	sets the noise parameters
    #

    def set_noise(self, new_s_noise, new_d_noise, new_m_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.steering_noise    = float(new_s_noise)
        self.distance_noise    = float(new_d_noise)
        self.measurement_noise = float(new_m_noise)

    # --------
    # check: 
    #    checks of the robot pose collides with an obstacle, or
    # is too far outside the plane

    def check_collision(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    dist = sqrt((self.x - float(i)) ** 2 + 
                                (self.y - float(j)) ** 2)
                    if dist < 0.5:
                        self.num_collisions += 1
                        return False
        return True
        
    def check_goal(self, goal, threshold = 0.5):
        dist =  sqrt((float(goal[0]) - self.estimate[0]) ** 2 + (float(goal[1]) - self.estimate[1]) ** 2)
        #dist =  sqrt((float(goal[0]) - self.x) ** 2 + (float(goal[1]) - self.y) ** 2)
        print
        print 'CHECK >>>> Dist to goal:', dist
        return dist < threshold
        
    # --------
    # move: 
    #    steering = front wheel steering angle, limited by max_steering_angle
    #    distance = total distance driven, most be non-negative

    def move(self, steering, distance, ismyrobot,
             tolerance = 0.001, max_steering_angle = pi / 4.0 ):
        

        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        # For DEBUG ONLY
        steer = steering
        orientbefore = self.orientation

        if ismyrobot == True:
            print 'MOVING from: ', self.x, self.y, 'orientbefore=', degrees(orientbefore), 'steer requested=', degrees(
                steering)

        # make a new copy
        res = copy.deepcopy(self)

        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # Execute motion
        turn = tan(steering2) * distance2 / res.length

        if True: #abs(turn) < tolerance:
            # approximate by straight line motion
            res.x = self.x + (distance2 * cos(self.orientation))
            res.y = self.y + (distance2 * sin(self.orientation))
            res.orientation = (self.orientation + turn) % (2.0 * pi)

        else:
            # approximate bicycle model for motion
            radius = distance2 / turn
            cx = self.x - (sin(self.orientation) * radius)
            cy = self.y + (cos(self.orientation) * radius)
            res.orientation = (self.orientation + turn) % (2.0 * pi)
            res.x = cx + (sin(res.orientation) * radius)
            res.y = cy - (cos(res.orientation) * radius)

        if MOTION_MODEL == MOTION_MODEL_BYPASSED:
            # DEBUG ONLY
            # Brute force the move to consider the steering
            res.orientation = (self.orientation + steer) % (2.0 * pi)
            res.x = self.x + (distance2 * cos(res.orientation))
            res.y = self.y + (distance2 * sin(res.orientation))

        if ismyrobot == True:
            print 'MOVED  to  : ',res.x, res.y, 'orientafter=', degrees(res.orientation), 'steer applied=', degrees(res.orientation-orientbefore)

        return res

    # --------
    # sense: capable of Landmark sensing or GPS sensing
    #    

    def sense(self, grid, sensing_method):

        #reset array for sensing
        
        global sensinglmx
        global sensinglmy
        global sensinggpsx
        global sensinggpsy
        
        sensinglmx = [] # LM sensing plotting
        sensinglmy = [] # LM sensing plotting
        sensinggpsx = [] # GPS sensing plotting
        sensinggpsy = [] # GPS sensing plotting
            
        # Landmark Sensing
        if(sensing_method == SENSING_LM):
            # estimate: provide de current robot location and orientation: x, y, orient

            inrange = []
            visible_range = VisibleRange
            count = 0

            
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 1:
                        # noisy measurement of "[i,j]"
                        i2 = random.gauss(i, self.measurement_noise)
                        j2 = random.gauss(j, self.measurement_noise)
                        dist = sqrt((self.x - float(i2)) ** 2 + (self.y - float(j2)) ** 2)
                        # print 'LM estimated distance:', i2, j2, dist
                        if dist < visible_range:  # if landmark is visible, store: id, dx, dy.
                            dx = i2 - self.x
                            dy = j2 - self.y
                            lmid = len(grid[0]) * i + j  # landmark id
                            # inrange.append([i,j,dist,lmid])
                            inrange.append([lmid, dx, dy, i2, j2, dist])
                            count += 1
                            sensinglmx.append(i2) # tmp trace
                            sensinglmy.append(j2) # tmp trace

            print 'SENSING LM - Number of lm in range:', count, 'Actual true position: ', self.x, self.y
            for i in range(len(inrange)):
                print 'SENSING LM :', inrange[i]

            return inrange

        # GPS sensing
        elif (sensing_method == SENSING_GPS):
            x = random.gauss(self.x, self.measurement_noise)
            y = random.gauss(self.y, self.measurement_noise)
            sensinggpsx.append(x)  # tmp trace
            sensinggpsy.append(y)  # tmp trace
            print 'SENSING GPS:', x, y, 'Actual true position: ', self.x, self.y
            return [x,y]


    # --------
    # sense: robot sense Landmarks
    #   
    def sense_landmarks(self, grid):

        # estimate: provide de current robot location and orientation: x, y, orient
        
        inrange = []
        visible_range = VisibleRange
        count = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    #noisy measurement of "[i,j]"
                    i2 = random.gauss(i, self.measurement_noise)
                    j2 = random.gauss(j, self.measurement_noise)
                    dist = sqrt((self.x - float(i2)) ** 2 + (self.y - float(j2)) ** 2)
                    #print 'LM estimated distance:', i2, j2, dist
                    if dist < visible_range:   # if landmark is visible, store: id, dx, dy.
                        dx = i2 - self.x
                        dy = j2 - self.y
                        lmid = len(grid[0])*i + j # landmark id
                        #inrange.append([i,j,dist,lmid])
                        inrange.append([lmid, dx, dy, i2, j2, dist])
                        count +=1          

        print 'Robot sensing LM - robot:',self.x,self.y,'Number of lm in range:', count
        for i in range(len(inrange)):
            print 'lm:', inrange[i]

        return inrange

    # --------
    # measurement_prob
    #    computes the probability of a measurement
    # 

    def measurement_prob(self, measurement):

        # compute errors
        error_x = measurement[0] - self.x
        error_y = measurement[1] - self.y

        # calculate Gaussian
        error = exp(- (error_x ** 2) / (self.measurement_noise ** 2) / 2.0) \
            / sqrt(2.0 * pi * (self.measurement_noise ** 2))
        error *= exp(- (error_y ** 2) / (self.measurement_noise ** 2) / 2.0) \
            / sqrt(2.0 * pi * (self.measurement_noise ** 2))
           
        return error

    # --------
    # measurement_prob
    #    computes the probability of a measurement
    # 

    def multi_measurement_prob(self, measurement):
        # measurement: a matrix, distances measured by the robot to the in range LMs.
        #  measurement[][0] = landmark id, unused here
        #  measurement[][1] = robot - landmark dx, unused here
        #  measurement[][2] = robot - landmark dy, unused here
        #  measurement[][3] = LM x coordinate
        #  measurement[][4] = LM y coordinate
        #  measurement[][5] = myrobot measured distance

        # calculates how likely a measurement should be
        
        prob = 1.0;
        #print 'len(measurement)',len(measurement)
        for i in range(len(measurement)):
            # measure dist btw this particle and the i'th LM
            dist = sqrt((self.x - measurement[i][3]) ** 2 + (self.y - measurement[i][4]) ** 2)
            # compute distance btween this measurement and myrobot measurement
            prob *= self.Gaussian(dist, self.measurement_noise, measurement[i][5])
            # print self.x, self.y, dist, measurement[i][5], self.measurement_noise
        #print '------particle proba:',prob
        return prob


    def Gaussian(self, mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


    def __repr__(self):
        # return '[x=%.5f y=%.5f orient=%.5f]'  % (self.x, self.y, self.orientation)
        return '[%.5f, %.5f]'  % (self.x, self.y)

    #return average distance run between 2 sample = Speed, assuming sampling in time is the same
    def speed_estimator(self, table, position):
        DEPTH = 2 # Depth for the averaging
        sumval = 0
        if len(table) < DEPTH:
            table.append(position)
        else:
            table.pop(0)
            table.append(position)
        for i in range(len(table) - 1):
            dx = table[i + 1][0] - table[i][0]
            dy = table[i + 1][1] - table[i][1]
            sumval += sqrt(dx ** 2 + dy ** 2)
        if len(table) - 1 > 0:
            avg = sumval / (len(table) - 1)
        else:
            avg = 0
        return table, avg


    def teta_integrator(self, table, value):
        DEPTH = 4
        sumval = 0
        if len(table) < DEPTH:
            table.append(value)
        else:
            table.pop(0)
            table.append(value)
        for i in range(len(table)): sumval += table[i]
        return table, sumval / len(table)


    def control(self, spath, estimate, params):
    # orient = current orientation
    # dist = actual distance moved every step

        
        diff_cte = - self.cte
        diff_teta = - self.teta
        # ----------------------------------------
        # compute the CTE

        # The code here implements the solution to the multi segment CTE computation
        # When U exceedes 1, we increment the index, it is the number of segments starting from 0.
        U = 2
        rx,ry,deltaX,deltaY = 0,0,1,1
        last_index = len(spath) - 2
        endofpath = False

        # While is searching for the 1st sgment that gives U <= 1, then that segment is considered
        # This is to handle case case where we changed segment or even we skipped some.
        # Note: U is how far we progressed along the segment
        while U > 1.0 and endofpath ==False:
            p1x = spath[self.index][0]
            p1y = spath[self.index][1]
            p2x = spath[self.index + 1][0]
            p2y = spath[self.index + 1][1]
            x = estimate[0]
            y = estimate[1]
            deltaX = p2x - p1x
            deltaY = p2y - p1y
            rx = x - p1x
            ry = y - p1y
            U = (rx * deltaX + ry * deltaY) / (deltaX ** 2 + deltaY ** 2)
            if U > 1.0:
                if self.index == last_index:
                    endofpath = True
                    print 'LAST SEGMENT REACHED AND PASSED', 'U:', U
                else:
                    self.index += 1
                    print 'PID - Changing segment index as U is:', U, 'move to new index:', self.index

        self.cte = (ry * deltaX - rx * deltaY) / sqrt(deltaX ** 2 + deltaY ** 2)
        diff_cte += self.cte

        if SELECTED_PID == PID_CTE:
            steer = - params[0] * self.cte - params[2] * diff_cte  # compute steer command using PID method (in fact only PD here)
            print 'PID-CTE - estimate:', estimate, 'U=', U, 'steer-deg:', degrees(steer), 'cte:', self.cte, 'Ppart=', - params[0] * self.cte, 'Dpart=',- params[2] * diff_cte

        if SELECTED_PID == PID_ORIENT:
             # Select next segment: P2 - P3 until eof of path is reached
            if self.index + 5 <= last_index:
                p3x = spath[self.index + 2][0]
                p3y = spath[self.index + 2][1]
            else:
                p3x = p2x
                p3y = p2y

            segorient = atan2(deltaY, deltaX)
            alpha1 = atan2(p2y-estimate[1], p2x-estimate[0]) # angle Estimate to P2, range -pi/pi
            alpha2 = atan2(p3y-estimate[1], p3x-estimate[0]) # angle Estimate to P3, range -pi/pi

            #Manage the issue at -pi/pi boundary in order to get both angles in same direction.
            #It Detects the meaningfull half circle and get both alpha1 and 2 in same rotation direction.
            if alpha1 > pi/2:
                if alpha2 < -pi/2: alpha2 += 2*pi
            if alpha1 < -pi/2:
                if alpha2 > pi/2: alpha1 += 2*pi

            alpha = (1-U)*alpha1 + U*alpha2 # Compute weighted target angle

            self.teta = estimate[2] - alpha # orientation error to be corrected
            print 'estimated orientation:', degrees(estimate[2]),'alpha1:',degrees(alpha1),'alpha2:',degrees(alpha2),'alpha:',degrees(alpha),'teta:',degrees(self.teta)

            # limit to -pi/pi
            if self.teta > pi: self.teta -=  2*pi
            else:
                if self.teta < -pi: self.teta += 2*pi
                else: pass

            diff_teta += self.teta
            self.tetatab, integteta = self.teta_integrator(self.tetatab, self.teta)
            print 'INTEGRATOR:', self.tetatab, integteta

            steer = - params[0] * self.teta - params[1] * integteta - params[2] * diff_teta

            print 'PID-ORIENT - estimate:', estimate, 'U', U, \
             'segment orient;:', degrees(segorient),'actual orient-deg:', \
             degrees(self.orientation), 'steer-deg:', degrees(steer), 'teta-deg:', \
             degrees(self.teta), 'integteta:-deg:', degrees (integteta), 'cte:', self.cte

        # Max Steering +-Pi/4
        if steer > pi/4:
            steer = pi/4
            print 'Steer forced to Pi/4'
        if steer < -pi/4:
            steer = -pi/4
            print 'Steer forced to -Pi/4'

        self.steering = steer

        return steer  # return steering correction in Radian

    def odometer(self):
        # Odometer: My robot estimate its move.
        # We considere here a perfect odometer: the move is compute as a diff between the real position
        # before and after the move.
        # Other solution:
        # 1)GPS sensing, but it adds noise, probably too much.
        # 2)also tried diff between new real position and the estimate before the move...
        # ----------------------------
        move_dx = self.estimate[0] - self.prev_estimate[0]
        move_dy = self.estimate[1] - self.prev_estimate[1]
        speed = sqrt(move_dx** 2 + move_dy** 2)
        print 'ODOMETER - dx:', move_dx, 'dy:', move_dy, 'Speed:', speed
        return [move_dx, move_dy, speed]

# =======================================================================================
# this is the particle filter class
#   init
#   get_position
#   move
#   sense
# =======================================================================================
class particles:

    # --------
    # init: 
    #	creates particle set with given initial position
    #

    def __init__(self, x, y, theta, 
                 steering_noise, distance_noise, measurement_noise, vehicle_length, N = N_PARTICLES):
        self.N = N
        self.steering_noise    = steering_noise
        self.distance_noise    = distance_noise
        self.measurement_noise = measurement_noise

        self.data = []
        for i in range(self.N):
            r = robot(vehicle_length)
            r.set(x, y, theta) #random.random() * 2.0 * pi )#
            r.set_noise(steering_noise, distance_noise, measurement_noise)
            self.data.append(r)

    # --------
    #
    # extract position from a particle set
    #
    # Ca retourne la position du robot comme la position moyenne x/y/orientation de
    # toutes les particules (chacune etant une estimation de position du robot)
   
    def get_position(self):
        x = 0.0
        y = 0.0
        orientation = 0.0
        
        #reset before to use it
        global particlex
        global particley
        
        particlex = []
        particley = []
        for i in range(self.N):
            particlex.append(self.data[i].x)
            particley.append(self.data[i].y)
            x += self.data[i].x
            y += self.data[i].y
            # orientation is tricky because it is cyclic. By normalizing
            # around the first particle we are somewhat more robust to
            # the 0=2pi problem
            orientation += (((self.data[i].orientation
                              - self.data[0].orientation + pi) % (2.0 * pi)) 
                            + self.data[0].orientation - pi)

        return [x / self.N, y / self.N, orientation / self.N]

    # --------
    #
    # motion of the particles
    # 

    def move(self, steer, speed):
        newdata = []

        for i in range(self.N):
            r = self.data[i].move(steer, speed, False)
            newdata.append(r)
        self.data = newdata

    # --------
    #
    # (Particle) sensing and resampling
    # 

    def sense(self, Z, sensing_method):
        w = []

        if sensing_method == SENSING_LM:
            for i in range(self.N):
                w.append(self.data[i].multi_measurement_prob(Z))  # Multiple LM sensing method
        elif sensing_method == SENSING_GPS:
            for i in range(self.N):
                w.append(self.data[i].measurement_prob(Z))  # GPS sensing method

        # resampling (careful, this is using shallow copy)
        p3 = []
        index = int(random.random() * self.N)
        beta = 0.0
        mw = max(w)

        for i in range(self.N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % self.N
            p3.append(self.data[index])
        self.data = p3
 
# =======================================================================================
# run:  Move the robot along the planned path
# =======================================================================================

def run_init(myUI, spath, steering_noise, distance_noise, measurement_noise, initpos, initorient):
    # create and init my robot, the particle set and my slam filter
    myrobot = robot(myUI.vehicle_length)
    myrobot.set(initpos[0], initpos[1], initorient)
    myrobot.set_noise(steering_noise, distance_noise, measurement_noise)
    print 'Init -', 'Position:', myrobot.x, myrobot.y, 'Orient-Deg:',degrees(myrobot.orientation),'Noises:',steering_noise, distance_noise, measurement_noise

    filter = particles(initpos[0], initpos[1], initorient, steering_noise, distance_noise, measurement_noise, myUI.vehicle_length, 100)
    myslam = slamfilter(distance_noise, measurement_noise)

    # plot along the run
    myUI.plot_live_init(spath)

    return myrobot, filter, myslam

#----------------------------------------------------



def run_asfunction(myUI, spath, myrobot, myslam, filter, N, measurements, prev_speed, prev_steering):
    global estimateWithoutLag
    #   Feed the control process with 1 new measurement set: can be 1 GPS position, or a table of LM positions
    #   Estimate current position as the average of particles position
    #   Appli PID: calculate steer based on CTE
    #   move my robot
    #   move all particles alive
    #   My robot senses its environment: LM sensing or GPS sensing
    #   Applies the sense to particles
    print 'RUN: Motion number: ', N
    print 'RUN: True RobotXYO:', myrobot.x, myrobot.y, degrees(myrobot.orientation)
    print 'RUN: spath length:', len(spath), 'current index:', myrobot.index
    print 'RUN: Measurements:', measurements
    print 'RUN: Previous Speed:', prev_speed
    
    
    if FUNCTIONAL_MODE == MODE_REAL:
        global sensinggpsx
        global sensinggpsy
        sensinggpsx.append(measurements[0])  # trace
        sensinggpsy.append(measurements[1])  # trace

    # Move particles according to latest applied steering and last speed estimate
    filter.move(prev_steering, prev_speed)  # apply the same move to all particles

    # Feed Particle filter
    filter.sense(measurements, myUI.sensing_cfg)

    if GPS_LAG_TIME != 0:
    
        estimateWithoutLag = filter.get_position() 
        
        #move particule with GPS lag
        particuleTmp = copy.deepcopy(filter)
        #to be rework with real speed
        particuleTmp.move(prev_steering,prev_speed * GPS_LAG_TIME * 7.1)    # Estimation robot position   
    
        #compute the new position and estimate based on prticule filter
        myrobot.prev_estimate = myrobot.estimate
        myrobot.estimate = particuleTmp.get_position()   # Estimate current position as average of particles position
    else:
        myrobot.prev_estimate = myrobot.estimate
        myrobot.estimate = filter.get_position()   # Estimate current position as average of particles position        
    
    print 'FILTER ESTIMATE: ', myrobot.estimate

    # Uncomment to Bypass Particle Filter and force the estimate to be last measurement
    if PARTICLE_FILTER == FILTER_NOT_ACTIVATED:
        myrobot.estimate[0] = measurements[0]
        myrobot.estimate[1] = measurements[1]
        myrobot.estimate[2] = measurements[2]
        print 'FILTER BYPASSED', myrobot.estimate


    # Graph slam to estimate Robot and LM positions:
    if myUI.slam == SLAM_ON:
        myslam.online_slam(N,measurements,robotTmp.odometer())

    steer = myrobot.control(spath, myrobot.estimate, [myUI.p_gain, myUI.i_gain, myUI.d_gain])

    return myrobot, filter, myslam, steer


# =======================================================================================
# this is the matrix class
# we use it because it makes it easier to collect constraints in GraphSLAM
# and to calculate solutions (albeit inefficiently)
#   init
#   zero
#   identity
#   show
#   transpose
#   take
#   expand
#   Cholesky
#   inverse
#   insert_one_dimension

# =======================================================================================
class matrix:
    
    # implements basic operations of a matrix class

    # ------------
    #
    # initialization - can be called with an initial matrix
    #

    def __init__(self, value = [[]]):
        self.value = value
        self.dimx  = len(value)
        self.dimy  = len(value[0])
        if value == [[]]:
            self.dimx = 0

    # ------------
    #
    # makes matrix of a certain size and sets each element to zero
    #

    def zero(self, dimx, dimy):
        if dimy == 0:
            dimy = dimx
        # check if valid dimensions
        if dimx < 1 or dimy < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            
            self.dimx  = dimx
            self.dimy  = dimy
            self.value = [[0.0 for row in range(dimy)] for col in range(dimx)]

    # ------------
    #
    # makes matrix of a certain (square) size and turns matrix into identity matrix
    #

    def identity(self, dim):
        # check if valid dimension
        if dim < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx  = dim
            self.dimy  = dim
            self.value = [[0.0 for row in range(dim)] for col in range(dim)]
            for i in range(dim):
                self.value[i][i] = 1.0
    # ------------
    #
    # prints out values of matrix
    #

    def show(self, txt = ''):
        for i in range(len(self.value)):
            print txt + '['+ ', '.join('%.3f'%x for x in self.value[i]) + ']' 
        print ' '

    # ------------
    #
    # defines elmement-wise matrix addition. Both matrices must be of equal dimensions
    #

    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimx != other.dimx:
            raise ValueError, "Matrices must be of equal dimension to add"
        else:
            # add if correct dimensions
            res = matrix()
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] + other.value[i][j]
            return res

    # ------------
    #
    # defines elmement-wise matrix subtraction. Both matrices must be of equal dimensions
    #

    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimx != other.dimx:
            raise ValueError, "Matrices must be of equal dimension to subtract"
        else:
            # subtract if correct dimensions
            res = matrix()
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] - other.value[i][j]
            return res

    # ------------
    #
    # defines multiplication. Both matrices must be of fitting dimensions
    #

    def __mul__(self, other):
        # check if correct dimensions
        if self.dimy != other.dimx:
            raise ValueError, "Matrices must be m*n and n*p to multiply"
        else:
            # multiply if correct dimensions
            res = matrix()
            res.zero(self.dimx, other.dimy)
            for i in range(self.dimx):
                for j in range(other.dimy):
                    for k in range(self.dimy):
                        res.value[i][j] += self.value[i][k] * other.value[k][j]
        return res


    # ------------
    #
    # returns a matrix transpose
    #

    def transpose(self):
        # compute transpose
        res = matrix()
        res.zero(self.dimy, self.dimx)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        return res

    # ------------
    #
    # creates a new matrix from the existing matrix elements.
    #
    # Example:
    #       l = matrix([[ 1,  2,  3,  4,  5], 
    #                   [ 6,  7,  8,  9, 10], 
    #                   [11, 12, 13, 14, 15]])
    #
    #       l.take([0, 2], [0, 2, 3])
    #
    # results in:
    #       
    #       [[1, 3, 4], 
    #        [11, 13, 14]]
    #       
    # 
    # take is used to remove rows and columns from existing matrices
    # list1/list2 define a sequence of rows/columns that shall be taken
    # if no list2 is provided, then list2 is set to list1 (good for 
    # symmetric matrices)
    #

    def take(self, list1, list2 = []):
        if list2 == []:
            list2 = list1
        if len(list1) > self.dimx or len(list2) > self.dimy:
            raise ValueError, "list invalid in take()"

        res = matrix()
        res.zero(len(list1), len(list2))
        for i in range(len(list1)):
            for j in range(len(list2)):
                res.value[i][j] = self.value[list1[i]][list2[j]]
        return res

    # ------------
    #
    # creates a new matrix from the existing matrix elements.
    #
    # Example:
    #       l = matrix([[1, 2, 3],
    #                  [4, 5, 6]])
    #
    #       l.expand(3, 5, [0, 2], [0, 2, 3])
    #
    # results in:
    #
    #       [[1, 0, 2, 3, 0], 
    #        [0, 0, 0, 0, 0], 
    #        [4, 0, 5, 6, 0]]
    # 
    # expand is used to introduce new rows and columns into an existing matrix
    # list1/list2 are the new indexes of row/columns in which the matrix
    # elements are being mapped. Elements for rows and columns 
    # that are not listed in list1/list2 
    # will be initialized by 0.0.
    #
    
    def expand(self, dimx, dimy, list1, list2 = []):
        if list2 == []:
            list2 = list1
        if len(list1) > self.dimx or len(list2) > self.dimy:
            raise ValueError, "list invalid in expand()"

        res = matrix()
        res.zero(dimx, dimy)
        for i in range(len(list1)):
            for j in range(len(list2)):
                res.value[list1[i]][list2[j]] = self.value[i][j]
        return res

    # ------------
    #
    # Computes the upper triangular Cholesky factorization of  
    # a positive definite matrix.
    # This code is based on http://adorio-research.org/wordpress/?p=4560
    #
    
    def Cholesky(self, ztol= 1.0e-5):

        res = matrix()
        res.zero(self.dimx, self.dimx)

        for i in range(self.dimx):
            S = sum([(res.value[k][i])**2 for k in range(i)])
            d = self.value[i][i] - S
            if abs(d) < ztol:
                res.value[i][i] = 0.0
            else: 
                if d < 0.0:
                    raise ValueError, "Matrix not positive-definite"
                res.value[i][i] = sqrt(d)
            for j in range(i+1, self.dimx):
                S = sum([res.value[k][i] * res.value[k][j] for k in range(i)])
                if abs(S) < ztol:
                    S = 0.0
                res.value[i][j] = (self.value[i][j] - S)/res.value[i][i]
        return res 
 
    # ------------
    #
    # Computes inverse of matrix given its Cholesky upper Triangular
    # decomposition of matrix.
    # This code is based on http://adorio-research.org/wordpress/?p=4560
    #
    
    def CholeskyInverse(self):

        res = matrix()
        res.zero(self.dimx, self.dimx)

        # Backward step for inverse.
        for j in reversed(range(self.dimx)):
            tjj = self.value[j][j]
            S = sum([self.value[j][k]*res.value[j][k] for k in range(j+1, self.dimx)])
            res.value[j][j] = 1.0/ tjj**2 - S/ tjj
            for i in reversed(range(j)):
                res.value[j][i] = res.value[i][j] = \
                    -sum([self.value[i][k]*res.value[k][j] for k in \
                              range(i+1,self.dimx)])/self.value[i][i]
        return res
    
    # ------------
    #
    # computes and returns the inverse of a square matrix
    #
    def inverse(self):
        aux = self.Cholesky()
        res = aux.CholeskyInverse()
        return res

    # ------------
    #
    # prints matrix (needs work!)
    #
    def __repr__(self):
        return repr(self.value)

    # ------------
    #
    # Insert or append one column and row (all 0) in the requested position
    # Not so nice... could be done with expand but now it works so i keep it.
    
    def insert_one_dimension(self, position):

        #works well with rectangular matrix.
        dimx = len(self.value)
        dimy = len(self.value[0])
        res = matrix()
        if dimx == dimy:
            res.zero(dimx+1, dimy+1)
        else:
            res.zero(dimx+1, 1)

        #print self.value
        #print 'input:', self.value, 'x:',dimx, 'y:',dimy, 'pos:',position

        # copy all that is preserved
        for n in range(dimx):
            for m in range(dimy):
                res.value[n][m] = self.value[n][m]
                
        if dimy == 1:  # this is a vector
            #print 'Vector'
            for n in range(dimx-position):
                #print (dimx-n, 0, ' get ', dimx-n-1, 0)
                res.value[dimx-n][0] = res.value[dimx-n-1][0]
        
            res.value[position][0] = 0
 
        else: # this is a matrix
            dim = dimx
            #print 'Matrix dim:', dim

            # Part that is moved
            for n in range(dim):
                for m in range(dim-position):
                    #print (n,dim-m, ' get ', n, dim-m-1)
                    res.value[n][dim-m] = self.value[n][dim-m-1]
            for n in range(dim-position):
                for m in range(dim+1):
                    #print (dim-n, m, ' get ', dim-n-1, m)
                    res.value[dim-n][m] = res.value[dim-n-1][m]
            for n in range(dim+1):
                #print n
                res.value[n][position] = 0
            for m in range(dim+1):
                res.value[position][m] = 0
        return res
        
# =======================================================================================
# print the result of SLAM, the robot pose(s) and the landmarks
# =======================================================================================
def print_result(N, num_landmarks, result):
    print
    print 'Estimated Pose(s):'
    for i in range(N):
        print '    ['+ ', '.join('%.3f'%x for x in result.value[2*i]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*i+1]) +']'
    print
    print 'Estimated Landmarks:'
    for i in range(num_landmarks):
        print '    ['+ ', '.join('%.3f'%x for x in result.value[2*(N+i)]) + ', ' \
            + ', '.join('%.3f'%x for x in result.value[2*(N+i)+1]) +']'

####################################################

# =======================================================================================
# This is the Graph Slam class
#   init
#   integrate_measurements
#   integrate_motion
#   online_slam
#   estimation
#   display
# =======================================================================================
class slamfilter:

    # --------
    # init: 
    #	creates slam filter set
    #

    def __init__(self,motion_noise,measurement_noise):
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        
        # Init the dimension of the filter
        dim = 2  # x and y for initial position

        #make the matrix and vector
        self.Omega = matrix()
        self.Omega.zero(dim, dim)
        self.Omega.value[0][0]=1.0 #initial position x
        self.Omega.value[1][1]=1.0 #initial position y

        self.Xi = matrix()
        self.Xi.zero(dim, 1)
        self.Xi.value[0][0]=0.0  # Need to put the init location instead of fix x=0 here
        self.Xi.value[1][0]=0.0  # Need to put the init location instead of fix y=0 here

        self.mu = matrix()
        self.mu.zero(dim, 1)
        self.mu.value[0][0]=0.0
        self.mu.value[1][0]=0.0

        self.lmlist = [] # list of landmarks, filled up as their id appear in the measurements
        print 'Slam initialized'
    
    # --------
    # integrate_measurements: 
    #	integrate landmark measurements
    #

    def integrate_measurements(self,k,measurement):

        # measurement: a matrix, distances measured by the robot to the in range LMs.
        #  measurement[][0] = landmark id, unused here
        #  measurement[][1] = robot - landmark dx, unused here
        #  measurement[][2] = robot - landmark dy, unused here
        #  measurement[][3] = LM x coordinate
        #  measurement[][4] = LM y coordinate
        #  measurement[][5] = myrobot measured distance
        
        print 'pose:',k, 'Omegasize:',len(self.Omega.value), 'Xisize:',len(self.Xi.value)

        measurement_noise = self.measurement_noise
        
        # index of the robot pose in the matrix/vector
        n = k * 2
        
        #integrate the measurements
        for i in range(len(measurement)):

            # Expand the matrix if needed
            # ...Is the landmark already in the list (so also in the matrix)?
            lmid = measurement[i][0]
            lmindex = -1
            for j in range(len(self.lmlist)):          
                if self.lmlist[j] == lmid:
                    lmindex = j
            if lmindex == -1:  # lm not found, add it to the list and expand the matrix
                self.lmlist.append(lmid)
                lmindex = len(self.lmlist)-1
                self.Omega = self.Omega.insert_one_dimension(len(self.Omega.value)) # for x
                self.Omega = self.Omega.insert_one_dimension(len(self.Omega.value)) # for y
                self.Xi = self.Xi.insert_one_dimension(len(self.Xi.value))
                self.Xi = self.Xi.insert_one_dimension(len(self.Xi.value))

            #m is the index of the landmark coordinate in the matrix/vector
            m = 2 * (k+1 + lmindex) #see data structure
            print '..meas#:',i, 'lmid: ',lmid, 'index in matrix: ',m
          
            for b in range(2):
                self.Omega.value[n+b][n+b] += 1.0 / measurement_noise 
                self.Omega.value[m+b][m+b] += 1.0 / measurement_noise
                self.Omega.value[n+b][m+b] +=-1.0 / measurement_noise
                self.Omega.value[m+b][n+b] +=-1.0 / measurement_noise
                self.Xi.value[n+b][0] += -measurement[i][1+b] / measurement_noise
                self.Xi.value[m+b][0] +=  measurement[i][1+b] / measurement_noise        


    # --------
    # integrate_motion: 
    #	integrate robot motion
    #
    def integrate_motion(self,k,motion):

        #self.motion_noise=1
        
        # index of the robot pose in the matrix/vector
        n = k * 2
               
       
        # Update based on robot motion
        # Expand the matrix to insert the motion
        self.Omega = self.Omega.insert_one_dimension(n+2) # for x
        self.Omega = self.Omega.insert_one_dimension(n+2) # for y
        self.Xi = self.Xi.insert_one_dimension(n+2) # for x
        self.Xi = self.Xi.insert_one_dimension(n+2) # for y
      
        for b in range(4):
            self.Omega.value[n+b][n+b] += 1.0 / self.motion_noise
        
        for b in range(2):
            self.Omega.value[n+b  ][n+b+2] += -1.0 / self.motion_noise 
            self.Omega.value[n+b+2][n+b  ] += -1.0 / self.motion_noise
            self.Xi.value[n+b  ][0] += -motion[b] / self.motion_noise
            self.Xi.value[n+b+2][0] +=  motion[b] / self.motion_noise

    # --------
    # This is the all in 1 function: integrates both motion and measurements

    def online_slam(self, k, measurement, motion):
    #
        # measurement: a matrix, distances measured by the robot to the in range LMs.
        #  measurement[][0] = landmark id, unused here
        #  measurement[][1] = robot - landmark dx, unused here
        #  measurement[][2] = robot - landmark dy, unused here
        #  measurement[][3] = LM x coordinate
        #  measurement[][4] = LM y coordinate
        #  measurement[][5] = myrobot measured distance

        print 'SLAM pose:',k, 'Omegasize:',len(self.Omega.value), 'Xisize:',len(self.Xi.value)
                
        measurement_noise = self.measurement_noise
        motion_noise = self.motion_noise
        
        # integrate_motion
        # expand the information matrix and vector by one new position
        dim = len(self.Omega.value)
        list = [0, 1] + range(4,dim+2)
        self.Omega = self.Omega.expand(dim+2, dim+2, list, list)
        self.Xi  = self.Xi.expand(dim+2, 1, list, [0])
        
        # update the information maxtrix/vector based on the robot motion
        for b in range(4):
            self.Omega.value[b][b] +=  1.0 / motion_noise

        for b in range(2):
            self.Omega.value[b  ][b+2] += -1.0 / motion_noise
            self.Omega.value[b+2][b  ] += -1.0 / motion_noise
            self.Xi.value[b  ][0]      += -motion[b] / motion_noise
            self.Xi.value[b+2][0]      +=  motion[b] / motion_noise
        
        # integrate the measurements
        for i in range(len(measurement)):

            # Expand the matrix if needed
            # ...Is the landmark already in the list (so also in the matrix)?
            lmid = measurement[i][0]
            lmindex = -1
            for j in range(len(self.lmlist)):          
                if self.lmlist[j] == lmid:
                    lmindex = j
            if lmindex == -1:  # lm not found, add it to the list and expand the matrix
                self.lmlist.append(lmid)
                lmindex = len(self.lmlist)-1
                self.Omega = self.Omega.insert_one_dimension(len(self.Omega.value)) # for x
                self.Omega = self.Omega.insert_one_dimension(len(self.Omega.value)) # for y
                self.Xi = self.Xi.insert_one_dimension(len(self.Xi.value))
                self.Xi = self.Xi.insert_one_dimension(len(self.Xi.value))
        
            #m is the index of the landmark coordinate in the matrix/vector
            m = 2 * (2 + lmindex) #see data structure
            print 'SLAM meas#:',i, 'lmid: ',lmid, 'index in matrix: ',m

            n=2
            
            # update the information matrix/vector based on the measurement
            for b in range(2):
                self.Omega.value[n+b][n+b] +=  1.0 / measurement_noise
                self.Omega.value[m+b][m+b] +=  1.0 / measurement_noise
                self.Omega.value[n+b][m+b] += -1.0 / measurement_noise
                self.Omega.value[m+b][n+b] += -1.0 / measurement_noise
                self.Xi.value[n+b][0] += -measurement[i][1+b] / measurement_noise
                self.Xi.value[m+b][0] +=  measurement[i][1+b] / measurement_noise
            
        # now factor out the previous pose
        newlist = range(2, len(self.Omega.value))
        a = self.Omega.take([0, 1], newlist)
        b = self.Omega.take([0, 1])
        c = self.Xi.take([0, 1], [0])
        self.Omega = self.Omega.take(newlist) - a.transpose() * b.inverse() * a
        self.Xi = self.Xi.take(newlist, [0]) - a.transpose() * b.inverse() * c    

        # compute best estimate
        self.mu = self.Omega.inverse() * self.Xi    
    
        #return mu, Omega # make sure you return both of these matrices to be marked correct.


    # --------
    # estimation
    #   Estimate robot and LM positions
    #
    def estimation(self):
        # compute best estimate
        self.mu = self.Omega.inverse() * self.Xi
        #print_result(len(self.Xi.value)/2-len(self.lmlist), len(self.lmlist), self.mu)
        #Omega.show()
        #Xi.show()

    # --------
    #
    def display(self,text):
        print ''
        print 'Step:', text
        print 'size of Omega:', len(self.Omega.value)
        print 'size of Xi:', len(self.Xi.value)
        print 'number of landmarks detected:', len(self.lmlist)
        print 'landmarks:', self.lmlist
        print 'number of poses:',(len(self.Xi.value)-len(self.lmlist))/2
        print 'Omega:'
        for i in range(len(self.Omega.value)):
            print self.Omega.value[i]
        print 'Xi:'
        print self.Xi.value

# =======================================================================================
class userinteraction:
# =======================================================================================

    def __init__(self):
        #Default configuration

        self.noise_setting = NOISE_ON #NOISE_OFF #
        self.plot = PLOT_END  #PLOT_ALONG  #
        self.sensing_cfg = SENSING_GPS #SENSING_LM
        self.odometer_method = ODO_TRUE
        self.slam = SLAM_OFF  #SLAM_ON
        self.range = VisibleRange
        self.weight_data = WEIGHT_DATA  # smooth algo weight
        self.weight_smooth = WEIGHT_SMOOTH  # smooth algo weight
        self.gridname = GRID_FROM_GPSMAP #GRID_SMALL #


        #plotting...
        self.xgrid = []
        self.ygrid = []
        self.xestimated = []
        self.yestimated = []
        self.orientation = []
        self.steering = []
        self.xcollision = [0]
        self.ycollision = [0]
        self.xslampath = []
        self.yslampath = []
        self.xslamlm = []
        self.yslamlm = []
        self.xslamlmfinal = []
        self.yslamlmfinal = []
        self.xrobot = []
        self.yrobot = []
        self.xspath = []
        self.yspath = []
        
        self.GraphSlamImg = None
        

    # ------------------------------

    def print_current_setting(self):
        print
        print 'Current setting:'
        print 'Functional mode:', FUNCTIONAL_MODE
        print '  ', self.noise_setting
        print '  ', self.plot
        print '  ', self.sensing_cfg
        print '  ', self.odometer_method
        print '  ', self.slam
        print '  ', self.gridname
        print '  ', SELECTED_PID
        print '  ', MOTION_MODEL
        print '  ', PARTICLE_FILTER

    # ------------------------------
    def print_current_parameters(self):
        print
        print 'Current parameters:'
        print '   Speedvalue:', self.speed
        print '   VisibleRange:', self.range
        print '   steering_noise:', self.steering_noise
        print '   distance_noise:', self.distance_noise
        print '   measurement_noise:', self.measurement_noise
        print '   weight data:', self.weight_data
        print '   weight smooth:', self.weight_smooth
        print '   p gain:', self.p_gain
        print '   d gain:', self.d_gain
        print '   Vehicle length:', self.vehicle_length

    # ------------------------------

    def select_parameters(self, fixedconfig=False):
        self.print_current_setting()

        if fixedconfig == False: #Menu+interaction only if config is not fixed (this is for real DDrive)
            print
            print 'Menu:'
            menutop = {}
            menutop['0'] = 'EXIT'
            menutop['1'] = self.plot
            menutop['2'] = self.noise_setting
            menutop['3'] = self.sensing_cfg
            menutop['4'] = self.odometer_method
            menutop['5'] = self.slam
            menutop['6'] = self.gridname
            menutop['7'] = 'Display Current Setting'

            selection = 6
            while selection:
                options = menutop.keys()
                options.sort()
                print
                for entry in options:
                    print entry, menutop[entry]

                print
                selection = raw_input("Please Select: ")
                if selection == '0':
                    self.print_current_setting()
                    break
                elif selection == '1': # Toggle btw Plot along or at the en
                    if self.plot == PLOT_END:
                        self.plot = PLOT_ALONG
                    else:
                        self.plot = PLOT_END
                    menutop['1'] = self.plot
                elif selection == '2': # Toggle btw Noise On and Off
                    if self.noise_setting == NOISE_OFF:
                        self.noise_setting = NOISE_ON
                    else:
                        self.noise_setting = NOISE_OFF
                    menutop['2'] = self.noise_setting
                elif selection == '3':  # Toggle btw GPS and LM
                    if self.sensing_cfg == SENSING_GPS:
                        self.sensing_cfg = SENSING_LM
                    else:
                        self.sensing_cfg = SENSING_GPS
                    menutop['3'] = self.sensing_cfg
                elif selection == '4':
                    self.odometer_method = ODO_GPS
                    menutop['4'] = self.odometer_method
                elif selection == '5': # Toggle btw Slam on and off
                    if self.slam == SLAM_ON:
                        self.slam = SLAM_OFF
                    else:
                        self.slam = SLAM_ON
                    menutop['5'] = self.slam
                elif selection == '6':
                    menugrid = {}
                    menugrid['2'] = GRID_TINY
                    menugrid['3'] = GRID_SMALL
                    menugrid['4'] = GRID_LARGE
                    menugrid['5'] = GRID_EXTRALARGE
                    menugrid['6'] = GRID_FROM_GPSMAP

                    optiongrid = menugrid.keys()
                    optiongrid.sort()
                    print
                    for entry in optiongrid:
                        print entry, menugrid[entry]
                    selectgrid = raw_input("Please Select: ")
                    if selectgrid == '2': self.gridname = GRID_TINY
                    elif selectgrid == '3': self.gridname = GRID_SMALL
                    elif selectgrid == '4': self.gridname = GRID_LARGE
                    elif selectgrid == '5': self.gridname = GRID_EXTRALARGE
                    elif selectgrid == '6': self.gridname = GRID_FROM_GPSMAP
                    else: print "Unknown Option Selected!"
                    menutop['6'] = self.gridname
                elif selection == '7':
                    self.print_current_setting()

        # Set parameters according to the user configuration
        if self.gridname == GRID_FROM_GPSMAP:

            if self.noise_setting == NOISE_ON:
                self.steering_noise = STEERING_NOISE_FOR_DEEPDRIVE
                self.distance_noise = DISTANCE_NOISE_FOR_DEEPDRIVE
                self.measurement_noise = MEAST_NOISE_FOR_DEEPDRIVE
            else:
                # without noise
                self.steering_noise = 0.0001
                self.distance_noise = 0.0001
                self.measurement_noise = 0.0001

            if SELECTED_PID == PID_ORIENT:
                self.p_gain = P_GAIN_ORIENT_FOR_DEEPDRIVE
                self.i_gain = I_GAIN_ORIENT_FOR_DEEPDRIVE
                self.d_gain = D_GAIN_ORIENT_FOR_DEEPDRIVE
            elif SELECTED_PID == PID_CTE:
                self.p_gain = P_GAIN_FOR_DEEPDRIVE
                self.i_gain = I_GAIN_FOR_DEEPDRIVE
                self.d_gain = D_GAIN_FOR_DEEPDRIVE
            else:  # default
                print 'ERROR - PID NOT SELECTED'

            self.speed = Speedvalue_FOR_DEEPDRIVE
            self.vehicle_length = ROBOT_LENGTH_DEEP_DRIVE
            self.goal_distance = GOAL_DISTANCE_FOR_DEEPDRIVE

        else:
            if self.noise_setting == NOISE_ON:
                self.steering_noise = STEERING_NOISE
                self.distance_noise = DISTANCE_NOISE
                self.measurement_noise = MEAST_NOISE
            else:
                # without noise
                self.steering_noise = 0.0001
                self.distance_noise = 0.0001
                self.measurement_noise = 0.0001

            if SELECTED_PID == PID_ORIENT:
                self.p_gain = P_GAIN_ORIENT
                self.i_gain = I_GAIN_ORIENT
                self.d_gain = D_GAIN_ORIENT
            elif SELECTED_PID == PID_CTE:
                self.p_gain = P_GAIN
                self.i_gain = I_GAIN
                self.d_gain = D_GAIN
            else: #default
                print 'ERROR - PID NOT SELECTED'

            self.speed = Speedvalue
            self.vehicle_length = ROBOT_LENGTH
            self.goal_distance = GOAL_DISTANCE

        # Print the parameters we end up to
        self.print_current_parameters()

    # ------------------------------

    def build_grid(self):
        # grid format:
        #   0 = navigable space
        #   1 = occupied space

        if self.gridname == GRID_TINY:
            self.grid = [[0, 1],
                         [0, 1],
                         [0, 0]]
            self.init = [0, 0]
            self.goal = [len(self.grid) - 1, len(self.grid[0]) - 1]

        elif self.gridname == GRID_SMALL:
            self.grid = [[0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 1, 1, 0],
                         [0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0]]
            self.init = [0, 0]
            self.goal = [len(self.grid) - 1, len(self.grid[0]) - 1]

        elif self.gridname == GRID_MEDIUM:
            self.grid = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                         [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0]]
            self.init = [0, 0]
            self.goal = [len(self.grid) - 1, len(self.grid[0]) - 1]

        elif self.gridname == GRID_LARGE:
            self.grid = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                         [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                         [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                         [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
                         [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                         [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                         [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
            self.init = [0, 0]
            self.goal = [len(self.grid) - 1, len(self.grid[0]) - 1]

        elif self.gridname == GRID_EXTRALARGE:
            self.grid = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                          [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
                          [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
                          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                          [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                          [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                          [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                          [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                          [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
            self.init = [0, 0]
            self.goal = [len(self.grid) - 1, len(self.grid[0]) - 1]

        elif self.gridname == GRID_FROM_GPSMAP:
            self.grid = [[0 for row in range(int(RANGE_X))] \
                          for col in range(int(RANGE_Y))]
            self.init = [0, 0] #fake
            self.goal = [0, 2] #fake

        elif 0: # saveguard
            self.grid = [[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]

    # ------------------------------



    def plot_live_init(self, spath):
        global scale_XY
        #get scaling factor to convert millimeter into pixel
        with open(circuit_zone_of_interest_file, 'rb') as f:
            save_dict_circuit = pickle.load(f)
        scale_XY = save_dict_circuit['scale']
        f.close()
                
        #get image of circuit undistorded
        self.GraphSlamImg = cv2.imread('circuit.png',cv2.IMREAD_COLOR)
        #resize to have better view
        self.GraphSlamImg = cv2.resize(self.GraphSlamImg,None,fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation = cv2.INTER_CUBIC)
        
        #create grid
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == 1:
                    #append point
                    self.ygrid.append(j)
                    self.xgrid.append(i)
                    #plot cv2
                    plotPoint(self.GraphSlamImg,(i,j),(0,255,0),str(i))
        
        for i in range(len(spath)-1):
            #get first point and following from path
            pointA = spath[i][0],spath[i][1]
            pointB = spath[i+1][0],spath[i+1][1]
            #draw black point 
            plotPoint(self.GraphSlamImg,pointA,(0,0,0),str(i))
            #draw arrowf
            plotArrowedLine(self.GraphSlamImg,pointA,pointB,(0,0,255),'')
            #append the point 
            self.xspath.append(spath[i][0])
            self.yspath.append(spath[i][1])


        # matplot the real path
        plt.axis([-1.0, len(self.grid) + 2 - 0.5, len(self.grid[0]) + 2 - 0.5, -1.0])
        plt.grid(True)

        plt.plot(self.xgrid, self.ygrid, 'g^', \
                self.xspath, self.yspath, 'r--', \
                self.xspath, self.yspath, 'k.')

    #------------------------------

    def log_along_run(self,myrobot,myslam,estimate):
        # Log estimated and real path followed by the robot
        self.xestimated.append(estimate[0])
        self.yestimated.append(estimate[1])
        self.orientation.append(estimate[2])
        self.steering.append(myrobot.steering)
        self.xrobot.append(myrobot.x)
        self.yrobot.append(myrobot.y)

        self.xslampath.append(myslam.mu.value[0][0])
        self.yslampath.append(myslam.mu.value[1][0])

        # log collisions...
        if not myrobot.check_collision(self.grid):
            print '##### Collision ####'
            self.xcollision.append(myrobot.x)
            self.ycollision.append(myrobot.y)

        # log and plot along the run...
        self.xslamlm = []
        self.yslamlm = []
        number_of_poses = len(myslam.Xi.value) / 2 - len(myslam.lmlist)
        for i in range(len(myslam.lmlist)):
            self.xslamlm.append(myslam.mu.value[number_of_poses * 2 + 2 * i])
            self.yslamlm.append(myslam.mu.value[number_of_poses * 2 + 2 * i + 1])


    #------------------------------

    def plot_track(self,myrobot,myslam,estimate):
        #plot in yellow sensing
        plotSeriesPoint(self.GraphSlamImg,sensinglmx,sensinglmy,(0,255,255))
        plotPoint(self.GraphSlamImg,(sensinggpsx[-1],sensinggpsy[-1]),(0,255,255))
        #plotSeriesPoint(self.GraphSlamImg,particlex,particley,(255,0,0))
        plotPoint(self.GraphSlamImg,(self.xestimated[-1],self.yestimated[-1]),(0,0,0))
        plotLineTeta(self.GraphSlamImg,(self.xestimated[-1],self.yestimated[-1]),self.orientation[-1],10.0,(0,0,0),'')
        plotLineTeta(self.GraphSlamImg,(self.xestimated[-1],self.yestimated[-1]),self.orientation[-1] + self.steering[-1] ,15.0,(0,0,255),'')
        plotPoint(self.GraphSlamImg,(self.xcollision[-1],self.ycollision[-1]),(0,0,255))
        plotCross(self.GraphSlamImg,(self.xrobot[-1],self.yrobot[-1]),(0,0,255))
        plotCross(self.GraphSlamImg,(self.xslampath[-1],self.yslampath[-1]),(255,0,0))
        plotSeriesPoint(self.GraphSlamImg,self.xslamlm,self.yslamlm,(255,0,255))
        plotSeriesPoint(self.GraphSlamImg,self.xslamlmfinal,self.yslamlmfinal,(150,150,150))
        #plotCross(self.GraphSlamImg,(sensinggpsx[-2],sensinggpsy[-2]),(255,255,0))
        #plotArrowedLine(self.GraphSlamImg,(sensinggpsx[-2],sensinggpsy[-2]),(sensinggpsx[-1],sensinggpsy[-1]),(255,255,0),'')
        #plotArrowedLine(self.GraphSlamImg,(sensinggpsx[-1],sensinggpsy[-1]),(self.xestimated[-1],self.yestimated[-1]),(0,255,255),'')
        if GPS_LAG_TIME != 0:
            plotArrowedLine(self.GraphSlamImg,(estimateWithoutLag[0],estimateWithoutLag[1]),(self.xestimated[-1],self.yestimated[-1]),(0,255,0),'')
        else:
            plotArrowedLine(self.GraphSlamImg,(sensinggpsx[-1],sensinggpsy[-1]),(self.xestimated[-1],self.yestimated[-1]),(0,255,255),'')
            


        plt.plot(sensinglmx, sensinglmy, 'y.',  # tmp trace
                 sensinggpsx[-1], sensinggpsy[-1], 'y.',  # tmp trace
                 particlex, particley, 'b.',
                 self.xestimated[-1], self.yestimated[-1], 'ko',  # ''b^', \
                 self.xcollision[-1], self.ycollision[-1], 'ro', \
                 self.xrobot[-1], self.yrobot[-1], 'r^', \
                 self.xslampath[-1], self.yslampath[-1], 'bo', \
                 self.xslamlm, self.yslamlm, 'y^', \
                 self.xslamlmfinal, self.yslamlmfinal, 'k^')


    #------------------------------

    def textual_final_plot(self,myrobot,myslam, nposes, err, outoflimit):
        # Print SLAM outcome.
        # -------------------
        if self.slam == SLAM_ON:
            print_result(len(myslam.Xi.value) / 2 - len(myslam.lmlist), len(myslam.lmlist), myslam.mu)
            number_of_poses = len(myslam.Xi.value) / 2 - len(myslam.lmlist)
            print 'number_of_poses:', number_of_poses

        # compute Landmark position error
        lmerror = 0.0
        for i in range(len(myslam.lmlist)):
            lmid = myslam.lmlist[i]
            lmy = float(lmid % len(self.grid[0]))
            lmx = float(round(lmid / len(self.grid[0])))
            lmestimatex = myslam.mu.value[number_of_poses * 2 + 2 * i][0]
            lmestimatey = myslam.mu.value[number_of_poses * 2 + 2 * i + 1][0]
            lmerror += sqrt(((lmestimatex - lmx) ** 2) + ((lmestimatey - lmy) ** 2))
        if len(myslam.lmlist) != 0:
            avglmerror = lmerror / len(myslam.lmlist)
        else:
            avglmerror = 0

        self.xslamlmfinal = []
        self.yslamlmfinal = []
        # Final LM position estimate
        for i in range(len(myslam.lmlist)):
            self.xslamlmfinal.append(myslam.mu.value[number_of_poses * 2 + 2 * i])
            self.yslamlmfinal.append(myslam.mu.value[number_of_poses * 2 + 2 * i + 1])

        print
        print '--------------------------------------'
        if nposes >= TIMEOUT:
            print 'ERROR: TIME OUT'
        if outoflimit == 1:
            print 'ERROR: OUT OF WORLD LIMITS'

        myrobot.check_goal(self.goal, self.goal_distance)

        self.print_current_setting()
        self.print_current_parameters()

        print ' Results: '
        print '   CTE cumulated error:', err
        print '   Collisions:', len(self.xcollision)
        print '   poses:', nposes
        print '   SLAM avg-lmerror:', avglmerror
        print '   SLAM Sensed LMs:', len(myslam.lmlist)
        print '--------------------------------------'

#----------------------------------------------------------------------

def graph_slam_init(runningfromDeepDrive=False, startingposition = [3268, 630]):
    myUI = userinteraction() #create my user interface: parameters and plot
    myUI.select_parameters(runningfromDeepDrive) # set the parameters, menu
    myUI.build_grid()

    # Planning phase...
    path = plan(myUI.grid, myUI.init, myUI.goal) # create an empty path plan for the grid and init and goal
    path.astar()                  # find the optimal path using A* algo, stored in path.path

    # Overide the path in case of multiple loops...
    if myUI.gridname == GRID_FROM_GPSMAP:
        data_pos = open("gps_plot_8_1.txt", "rb")
        #data_pos = open("gps_plot_8_2.txt", "rb")
        #data_pos = open("gps_map.txt", "rb")
        poselist = pickle.load(data_pos)
        data_pos.close()

        #Un-comment one of those below
        #searchdepth = 53 # This is to make only 1  cicuit cycle and stop
        searchdepth = len(poselist)

        path.path = []
        path.path = [[0 for row in range(2)] \
                     for col in range(searchdepth)]

        if True: #runningfromDeepDrive == True:
            #Search closest point in poselist
            mindist = 1000
            for i in range (searchdepth):
                deltaX = startingposition[0] - poselist[i][1]
                deltaY = startingposition[1] - poselist[i][2]
                dist = sqrt(deltaX**2 + deltaY**2)
                if dist < mindist:
                    mindist = dist
                    startpose = i

            # Build the path
            for i in range(searchdepth):
                index = (startpose + i) % searchdepth
                path.path[i][0] = float(poselist[index][1]) / SHRINK_FACTOR
                path.path[i][1] = float(poselist[index][2]) / SHRINK_FACTOR
        else:
            for i in range(searchdepth):
                path.path[i][0] = float(poselist[i][1])/SHRINK_FACTOR
                path.path[i][1] = float(poselist[i][2])/SHRINK_FACTOR

        path.smooth(myUI.weight_data, myUI.weight_smooth) # smooth the path, store it in path.spath

        myUI.init = path.spath[0]
        myUI.goal = path.spath[len(path.spath)-1]
        myUI.initorient = atan2(path.spath[1][1]-path.spath[0][1],path.spath[1][0]-path.spath[0][0])
        #print 'Initial Orientation: ', degrees(myUI.initorient)

    else:
        path.smooth(myUI.weight_data, myUI.weight_smooth)  # smooth the path, store it in path.spath
        myUI.initorient = atan2(path.spath[1][1] - path.spath[0][1], path.spath[1][0] - path.spath[0][0])

    # Compute smalest segment lenght
    minsegment = 1000
    for i in range(0, len(path.spath)-1):
        p1x = path.spath[i][0]
        p1y = path.spath[i][1]
        p2x = path.spath[i + 1][0]
        p2y = path.spath[i + 1][1]
        deltaX = p2x - p1x
        deltaY = p2y - p1y
        length = sqrt((deltaX**2 + deltaY**2))
        if length < minsegment: minsegment = length
    print "Path Segment Lenght minimum is: ",  minsegment
    if Speedvalue > minsegment:
        print '!!!!! WARNING !!! Speed is: ',Speedvalue, ' You run too FAST for a path with segment min: ',minsegment,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # Set plotting parameter proportionally to the length of the path
    # Plot typically once every path index
    myUI.modulo = len(path.spath)

    return(myUI,path)

# =======================================================================================
# MAIN this is our main routine
# =======================================================================================

# JE DOIS CREER UNE SUPER CLASS GRAPH SLAM... A FAIRE
#class graphSlam:



#-----------------------------------------------------------------------------
#def main():
if __name__ == '__main__':
    outoflimit = 0
    InitNeedToBeDone = True
    while  outoflimit == 0:

        if InitNeedToBeDone == True :
            InitNeedToBeDone = False
            print "Graph Slam starting..."
        
            # Init world and setting
            myUserInterface, mypath = graph_slam_init()
        
            # Init my robot, filter and slam...
            myrobot, filter, myslam = run_init(myUserInterface,
                                               mypath.spath,
                                               myUserInterface.steering_noise,
                                               myUserInterface.distance_noise,
                                               myUserInterface.measurement_noise,
                                               myUserInterface.init,
                                               myUserInterface.initorient)
            err = 0.0
            N = 0
            outoflimit = 0
            previoussteering = 0
            prev_speed = 0
            speed = myUserInterface.speed
            lastPosTable = []
              
        # Sense: My robot Sense its environment
        measurements = myrobot.sense(myUserInterface.grid, myUserInterface.sensing_cfg)

        # Select next portion of the spath
        # ADD CODE HERE

        # Perform Graph Slam processing
        myrobot, \
        filter, \
        myslam, \
        steering = run_asfunction(myUserInterface,
                                    mypath.spath,
                                    myrobot,
                                    myslam,
                                    filter,
                                    N,
                                    measurements,
                                    prev_speed,
                                    previoussteering)

        # Move the real robot, according to the speed from UI
        if True: #N > 15: #
            myrobot = myrobot.move(steering, myUserInterface.speed, True)  # move my robot accordingly considering noise

        previoussteering = steering
        prev_speed = speed

        # Log and plot along the run...
        myUserInterface.log_along_run(myrobot, myslam, myrobot.estimate)
        myUserInterface.plot_track(myrobot, myslam, myrobot.estimate)
        if myUserInterface.plot == PLOT_ALONG:
            #refresh plot immediatly
            plt.pause(0.001)
        elif N % myUserInterface.modulo == 0:
            #refresh plot only sometimes
            plt.pause(0.001)

        err += (myrobot.cte ** 2)
        N += 1

        # Check out of limit
        # loop until goal is reached
        if myrobot.x > len(myUserInterface.grid) + OOL_MARGIN or myrobot.y > len(myUserInterface.grid[0]) + OOL_MARGIN:
            outoflimit = 1
        
        if myUserInterface.GraphSlamImg != None :
            cv2.imshow('GraphSlam',myUserInterface.GraphSlamImg)
            cv2.waitKey(1)
            
            
        if myrobot.check_goal(myUserInterface.goal, myUserInterface.goal_distance) == True:
            InitNeedToBeDone = True

    # Final Plot.
    # ---------------
    myUserInterface.textual_final_plot(myrobot, myslam, N, err, outoflimit)
    myUserInterface.plot_track(myrobot, myslam, myrobot.estimate)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        plt.pause(0.001)





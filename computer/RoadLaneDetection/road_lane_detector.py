from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
import imutils
import skimage.morphology
import numpy as np
import cv2
import math
import os
import sys
import traceback
import global_vars as g
    
from scipy.misc import imread, imshow, imresize
from line_fit_video import annotate_image
from laneconfiguration import *
from util import *


# Helper functions
def nothing(position):
    # print position
    pass

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


def get_fl_name():
    stack = traceback.extract_stack()
    filename, codeline, funcName, text = stack[-2]
    return (funcName, codeline)


def PID_process(error):
    """
    """
    Kp = g.Kp*100                                             # REMEMBER we are using Kp*100 so this is really 10 !
    Ki = g.Ki*100                                             # REMEMBER we are using Ki*100 so this is really 1 !
    Kd = g.Kd*100                                             # REMEMBER we are using Kd*100 so this is really 100!

    g.integral = g.integral + error                           # calculate the integral
    g.derivative = error - g.lastError                        # calculate the derivative
    steer_angle = Kp*error + Ki*g.integral + Kd*g.derivative  # the "P term" the "I term" and the "D term"
    steer_angle = steer_angle/100                             # REMEMBER to undo the affect of the factor of 100 in Kp, Ki and Kd!
    g.lastError = error                                       # save the current error so it can be the lastError next time around

    return steer_angle


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.	
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # In case of error, don't draw the line(s)
    if lines is None:
        return
    if len(lines) == 0:
        return

    draw_right = True
    draw_left = True
    color_blue = [0, 0, 255]

    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    slope_left = 0
    slope_right = 0
    angle = 0
    offset = 0
    angle_offset = 0
    angle_offset_pid = 0

    ####print ('lines:',lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)

    lines = new_lines

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    img_x_center = img.shape[1] / 2  # x coordinate of center of image
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
        x1, y1, x2, y2 = line[0]

        right_lines_x.append(x1)
        right_lines_x.append(x2)

        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
        x1, y1, x2, y2 = line[0]

        left_lines_x.append(x1)
        left_lines_x.append(x2)

        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - g.trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # compute 2 end points for middle line and compute slope
    slope_middle=999.
    if draw_right and draw_left:
        middle_x1 = img_x_center # True if camera is really center in middle of the car
        #middle_x1 = (left_x1 + right_x1)/2
        middle_lineH_x1 = (left_x1 + right_x1)/2
        middle_y1 = y1
        middle_x2 = -(left_b - right_b)/(left_m - right_m)
        middle_y2 = left_m * middle_x2 + left_b
        # Calculate slope
        if middle_x2 - middle_x1 == 0.:  # corner case, avoiding division by 0
            slope_middle = 999.  # practically infinite slope
            angle = 0.
            offset = middle_x1 - middle_lineH_x1
        else:
            slope_middle = (middle_y2 - middle_y1) / (middle_x2 - middle_x1)
            angle = math.atan((middle_x2 - middle_x1) / (middle_y1 - middle_y2)) * 180./np.pi
            offset = middle_x1 - middle_lineH_x1
            ### just for test offset = left_x1 - middle_lineH_x1
            angle_offset = math.degrees(math.atan(offset/(img.shape[0] - img.shape[0] * g.trap_height)))
            angle_offset_pid = PID_process(offset)


    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    if draw_right and draw_left:
        middle_x1 = int(middle_x1)
        middle_y1 = int(middle_y1)
        middle_x2 = int(middle_x2)
        middle_y2 = int(middle_y2)

    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
        slope_right = (y2 - y1) / (right_x2 - right_x1)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
        slope_left = (y2 - y1) / (left_x2 - left_x1)
    if draw_right and draw_left:
        s1 = 'Angle: ' + str(round(angle, 2)) + u'\xb0'
        cv2.line(img, (middle_x1, middle_y1), (middle_x2, middle_y2), color_blue, 5)
        #cv2.putText(img, s1, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Angle: ' + str(round(angle, 2)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Slope: ' + str(round(slope_middle, 2)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Offset: ' + str(round(offset, 2)) + 'Angle Offset: ' + str(round(angle_offset,2)) + ' angle_offset_pid: ' + str(round(angle_offset_pid,2)),
                    (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)

    if DEBUG_LEVEL >= DEBUG_LEVEL1:
        #print('slope_left:',slope_left ,'slope_right:',slope_right)
        print(get_fl_name(), "slope_left: %.2f -- slope_right: %.2f | slope_middle: %.2f -> angle: %.1f | offset: %3.2f | angle_offset: %3.2f | angle_offset_pid: %3.2f" % (slope_left, slope_right, slope_middle, angle, offset, angle_offset, angle_offset_pid))


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 3-channel RGB image
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., lambda1=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + lambda1
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambda1)


def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 200  # Initial 200 -> 130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower_yellow = np.array([90,100,100])
    lower_yellow = np.array([89,95,95])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return image2


def annotate_image_array_1(image_in):

    # Get Image properties
    g.vi_width = image_in.shape[1]
    g.vi_height = image_in.shape[0]

    """ Given an image Numpy array, return the annotated image as a Numpy array """
    # Only keep white and yellow pixels in the image, all other pixels become black
    #image_in = cv2.resize(image_in, (640, 480))
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(image_in)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        image1=cv2.cvtColor(image_in, cv2.COLOR_RGB2BGR)
        if g.cold_boot:
            cv2.namedWindow('image_in', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image_in', g.s_win_width, g.s_win_height)
            cv2.moveWindow('image_in', 0, g.s_height//2)
        cv2.imshow('image_in', image1)

    image = filter_colors(image_in)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(image)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        image1=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if g.cold_boot:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', g.s_win_width, g.s_win_height)
            cv2.moveWindow('image', g.s_win_width, 0)
        cv2.imshow('image', image1)

    # Read in and grayscale the image
    gray = grayscale(image)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(gray)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray', g.s_win_width, g.s_win_height)
            cv2.moveWindow('gray', g.s_win_width , g.s_win_height + g.s_win_height_offset)
        cv2.imshow('gray', gray)

    # Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, g.kernel_size)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(blur_gray)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('blur_gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('blur_gray', g.s_win_width, g.s_win_height)
            cv2.moveWindow('blur_gray', g.s_win_width, 2*g.s_win_height + 2*g.s_win_height_offset)
        cv2.imshow('blur_gray', blur_gray)

    # Apply Canny Edge Detector
    edges = canny(blur_gray, g.low_threshold, g.high_threshold)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(edges)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('canny_edges', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('canny_edges', g.s_win_width, g.s_win_height)
            cv2.moveWindow('canny_edges', g.s_win_width , 3*g.s_win_height + 2*g.s_win_height_offset)
        cv2.imshow('canny_edges', edges)

    # Create masked edges using trapezoid-shaped region-of-interest
    imshape = image.shape
    vertices = np.array([[\
        ((imshape[1] * (1 - g.trap_bottom_width)) // 2, imshape[0]),\
        ((imshape[1] * (1 - g.trap_top_width)) // 2, imshape[0] - imshape[0] * g.trap_height),\
        (imshape[1] - (imshape[1] * (1 - g.trap_top_width)) // 2, imshape[0] - imshape[0] * g.trap_height),\
        (imshape[1] - (imshape[1] * (1 - g.trap_bottom_width)) // 2, imshape[0])]]\
        , dtype=np.int32)
    if DEBUG_LEVEL >= DEBUG_LEVEL2:
        print (get_fl_name(), 'vertices:', vertices,'vertices.shape:', vertices.shape)

    masked_edges = region_of_interest(edges, vertices)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(masked_edges)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('masked_edges', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('masked_edges', g.s_win_width*2, g.s_win_height*2)
            cv2.moveWindow('masked_edges', 2*g.s_win_width, 1*g.s_win_height)
            if g.trackbar_enabled:
                cv2.createTrackbar('top', 'masked_edges', int(g.trap_top_width*100), 100, nothing)
                cv2.createTrackbar('bottom', 'masked_edges', int(g.trap_bottom_width*100), 100, nothing)
                cv2.createTrackbar('height', 'masked_edges', int(g.trap_height*100), 100, nothing)
                cv2.setTrackbarPos('top', 'masked_edges', int(g.trap_top_width*100))
                cv2.setTrackbarPos('bottom', 'masked_edges', int(g.trap_bottom_width*100))
                cv2.setTrackbarPos('height', 'masked_edges', int(g.trap_height*100))
        if g.trackbar_enabled:
            l_top = cv2.getTrackbarPos('top', 'masked_edges')
            g.trap_top_width = float(l_top)/100
            l_bottom = cv2.getTrackbarPos('bottom', 'masked_edges')
            g.trap_bottom_width = float(l_bottom)/100
            l_height = cv2.getTrackbarPos('height', 'masked_edges')
            g.trap_height = float(l_height)/100
        cv2.polylines(masked_edges,np.int32([vertices]), True, (255, 255, 0), 1, 0)
        cv2.imshow('masked_edges', masked_edges)

    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, g.rho, g.theta, g.threshold, g.min_line_length, g.max_line_gap)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(line_image)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('line_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('line_image', g.s_win_width, g.s_win_height)
            cv2.moveWindow('line_image', 4*g.s_win_width, 0)
        cv2.imshow('line_image', line_image)

    # Draw lane lines on the original image
    initial_image = image_in.astype('uint8')
    annotated_image = weighted_img(line_image, initial_image)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL1:
        imshow(annotated_image)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL1:
        image1=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        if g.cold_boot:
            cv2.namedWindow('annotated_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('annotated_image', g.s_win_width*2, g.s_win_height*2)
            cv2.moveWindow('annotated_image', 4*g.s_win_width, 1*g.s_win_height)
        cv2.imshow('annotated_image', image1)
        cv2.waitKey(1)

    g.cold_boot = False

    return annotated_image


def annotate_image_array_2(image_in):
    """ to be factorized """
    anotated_image=annotate_image(image_in)
    return anotated_image


def annotate_image_array_3(image_in):
    # Get Image properties
    g.vi_width = image_in.shape[1]
    g.vi_height = image_in.shape[0]

    """ Hough Transform parameters """
    g.rho = 0.8                 # (2 -> 0.8) distance resolution in pixels of the Hough grid
    g.theta = 1 * np.pi/180     # angular resolution in radians of the Hough grid
    g.threshold = 25            # (15 -> 25) minimum number of votes (intersections in Hough grid cell)
    g.min_line_length = 50      # (10 -> 50) minimum number of pixels making up a line
    g.max_line_gap = 200        # (20 -> 200) maximum gap in pixels between connectable line segments

    """ Given an image Numpy array, return the annotated image as a Numpy array """
    # Replace hsv mask filetring and canny filter with a second method
    #image_in = cv2.resize(image_in, (640, 480))
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(image_in)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        image1=cv2.cvtColor(image_in, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('image_in', cv2.WINDOW_NORMAL)
        cv2.imshow('image_in', image1)

    # Read in and grayscale the image
    gray = grayscale(image_in)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(gray)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
        cv2.imshow('gray', gray)

    # Improve contrast
    gray = cv2.equalizeHist(gray)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(gray)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('gray_eq_hist', cv2.WINDOW_NORMAL)
        cv2.imshow('gray_eq_hist', gray)


    # Gaussian blur
    #filter_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #filter_gray = cv2.GaussianBlur(gray, (3,3), 5)
    #filter_gray = cv2.GaussianBlur(gray, (3,3), 5)
    #filter_gray = cv2.medianBlur(gray, 5)
    filter_gray = cv2.bilateralFilter(gray,9,50,50)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(filter_gray)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('blur_gray', cv2.WINDOW_NORMAL)
        cv2.imshow('blur_gray', filter_gray)


    # Threshold
    filter_gray_thresh = cv2.adaptiveThreshold(filter_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(filter_gray_thresh)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('thresh_gray', cv2.WINDOW_NORMAL)
        cv2.imshow('thresh_gray', filter_gray_thresh)

    # Opening (erode/dilate = noise cleanup)
    kernel = np.ones((2,2),np.uint8)
    erode_frame = cv2.morphologyEx(filter_gray_thresh, cv2.MORPH_OPEN, kernel)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(erode_frame)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('erode_frame', cv2.WINDOW_NORMAL)
        cv2.imshow('erode_frame', erode_frame)

    # Skeletonize (substract/bitwise_or)
    # http://opencvpython.blogspot.fr/2012/05/skeletonization-using-opencv-python.html
    # TODO KO roi_frame = skimage.morphology.skeletonize(roi_frame)
    ####kernel = np.ones((1,1),np.uint8
    ####img_morphex = cv2.morphologyEx(erode_frame, cv2.MORPH_OPEN, kernel)
    ####blur = cv2.GaussianBlur(img_morphex,(1,1),0)
    ####ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ####print ('th4:',th4)
    ####th4[th4 == 255] = 1
    ####skeleton_frame = skimage.morphology.skeletonize(th4)
    skeleton_frame = imutils.skeletonize(erode_frame, size=(3, 3))

    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(skeleton_frame)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('skeleton_frame', cv2.WINDOW_NORMAL)
        cv2.imshow('skeleton_frame', skeleton_frame)

    bitwise_frame = cv2.subtract(filter_gray_thresh, erode_frame)
    ####bitwise_frame = cv2.subtract(filter_gray_thresh, skeleton_frame)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(bitwise_frame)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('bitwise_frame', cv2.WINDOW_NORMAL)
        cv2.imshow('bitwise_frame', bitwise_frame)

    # Create masked edges using trapezoid-shaped region-of-interest
    imshape = image_in.shape
    vertices = np.array([[\
            ((imshape[1] * (1 - g.trap_bottom_width)) // 2, imshape[0]),\
            ((imshape[1] * (1 - g.trap_top_width)) // 2, imshape[0] - imshape[0] * g.trap_height),\
            (imshape[1] - (imshape[1] * (1 - g.trap_top_width)) // 2, imshape[0] - imshape[0] * g.trap_height),\
            (imshape[1] - (imshape[1] * (1 - g.trap_bottom_width)) // 2, imshape[0])]]\
            , dtype=np.int32)
    if DEBUG_LEVEL >= DEBUG_LEVEL2:
        print (get_fl_name(), 'vertices:', vertices , 'vertices.shape:', vertices.shape)

    masked_edges = region_of_interest(bitwise_frame, vertices)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(masked_edges)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('masked_edges', cv2.WINDOW_NORMAL)
        cv2.imshow('masked_edges', masked_edges)

    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, g.rho, g.theta, g.threshold, g.min_line_length, g.max_line_gap)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(line_image)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('line_image', cv2.WINDOW_NORMAL)
        cv2.imshow('line_image', line_image)

    # Draw lane lines on the original image
    initial_image = image_in.astype('uint8')
    annotated_image = weighted_img(line_image, initial_image)
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL1:
        imshow(annotated_image)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL1:
        image1 = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('annotated_image', cv2.WINDOW_NORMAL)
        cv2.imshow('annotated_image', image1)
        cv2.waitKey(1)

    return annotated_image

def annotate_image_array_4(image_in):

    # Create the camera (recreate each time in case trackers are enabled, to refine fu/fv/alpha parameters)
    if g.cold_boot or g.trackbar_enabled :
        from inverse_perspective_transform import Camera, GroundImage
        img_height_ratio = 0.4
        camera = Camera(img_width=image_in.shape[1], img_height=int(image_in.shape[0] * img_height_ratio),
                        fu=g.ipm.camera_parameters['fu'],
                        fv=g.ipm.camera_parameters['fv'],
                        alpha=g.ipm.camera_parameters['alpha'],
                        cu_ratio=g.ipm.camera_parameters['cu_ratio'],
                        cv_ratio=g.ipm.camera_parameters['cv_ratio'])
        g.ipm.camera = camera
        g.ipm.ground_transform = GroundImage(camera)

    if g.cold_boot:
        from lanes_manager import SimpleLanesManager
        g.ipm.lanes_mgr = SimpleLanesManager(record=False)

    # Extract ROI (bottom of image)
    img_height = g.ipm.camera.img_height
    roi = image_in[-img_height:,:,:]

    # Grayscale the image
    gray = grayscale(roi)
    if gray.dtype == 'uint8':
        gray = gray / 255
        gray = gray.astype(np.float32)

    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        imshow(gray)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        cv2.namedWindow('1. gray', cv2.WINDOW_NORMAL)
        cv2.imshow('1. gray', gray)

    # Get de IPM ground view
    ground_view = g.ipm.ground_transform.generate_ground_image_fast(gray)

    display_debug_image(img=ground_view, frame_name='2. IPM ground view', position=(0, 0),
                        param_dict=g.ipm.camera_parameters, range_dict=g.ipm.camera_parameters_ranges)

    # Simple check: get image view from the camera view
    # image_view = g.ipm.ground_transform.generate_camera_image(ground_view, gray)
    # display_debug_image(img=image_view, frame_name='2.1 IPM image view', position=(0, 1))

    # Filter image
    from inverse_perspective_transform import filter_image
    pg_filtered = filter_image(ground_view)
    display_debug_image(pg_filtered, frame_name='3. Gaussian Kernel X+Y', position=(2, 2))

    # Threshold
    from inverse_perspective_transform import threshold_image
    pg_threshold = threshold_image(pg_filtered, g.ipm.threshold['percentile'])
    display_debug_image(pg_threshold, frame_name='4. Threshold', position=(3, 2),
                        param_dict=g.ipm.threshold)

    # Run Hough lines detection
    from inverse_perspective_transform import compute_hough_lines
    points_list = compute_hough_lines(pg_threshold)

    hough_lines_image = cv2.cvtColor(pg_threshold.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw hough lines
    for p1, p2 in points_list:
        cv2.line(hough_lines_image, p1, p2, (0, 255, 0), 1)
    display_debug_image(hough_lines_image, frame_name='5. Hough Lines', position=(4, 1),
                        param_dict=g.ipm.hough_parameters)


    # Run Probabilistic Hough lines detection
    from inverse_perspective_transform import compute_probabilistic_hough_lines
    points_list_p = compute_probabilistic_hough_lines(pg_threshold)

    hough_lines_p_image = cv2.cvtColor(pg_threshold.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for p1, p2 in points_list_p:
        cv2.line(hough_lines_p_image, p1, p2, (0, 255, 0), 1)
    display_debug_image(hough_lines_p_image, frame_name='5. Hough Lines P', position=(4, 2),
                        param_dict=g.ipm.hough_p_parameters)


    # IPM view with hough lines
    for p1, p2 in points_list:
        cv2.line(ground_view, p1, p2, (0, 255, 0), 1)
    display_debug_image(ground_view, frame_name='6. Hough Lines on IPM', position=(5, 2))

    # RANSAC
    from inverse_perspective_transform import ransac_fit
    image_ransac, lines_list = ransac_fit(pg_threshold, points_list)

    # Reverse
    im = g.ipm.ground_transform.generate_camera_image_lines(lines_list, gray)
    display_debug_image(im, frame_name='8. Reverse', position=(5, 2))

    lines_list = np.array(lines_list)

    lanes_img = None

    # sort lines based on their 1st x value
    index_sorted = np.argsort(lines_list[:, :, 1][:, -1])
    lines = lines_list[index_sorted,::]

    y = lines[0, :, 0]
    # only keep X data (Y data are the same for each line)
    lines = lines[:, :, 1]

    g.ipm.lanes_mgr.update_lines(lines)

    # Display 2d Lines
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL3:
        plt.clf()

        for lane in g.ipm.lanes_mgr.current_road_state.current_lanes:
            plt.plot(lane.skeleton, y, 'b', lw=1)

        for line in lines:
            plt.plot(line, y, 'r', lw=1)
            plt.draw()
        plt.xlim(0, 200)
        plt.ylim(np.max(y), 0)
        plt.show(block=False)


    # Reverse Lanes
    lanes_img = g.ipm.ground_transform.generate_camera_image_lanes(g.ipm.lanes_mgr.current_road_state.current_lanes, y, gray)
    display_debug_image(im, frame_name='9. Reverse Lanes', position=(5, 3))

    annotated_image = image_in

    if lanes_img is not None:
        offset = lanes_img.shape[0]

        if annotated_image.dtype == 'float32':
            annotated_image = (annotated_image / np.max(annotated_image)) * 255
            annotated_image = annotated_image.astype(np.uint8)

        if annotated_image.shape[2] > 3:
            annotated_image = annotated_image[:,:,0:3]

        #annotated_image[-offset:,:,:] = weighted_img(lanes_img, annotated_image[-offset:,:,:], alpha=0.1, beta=0.9)
        annotated_image[-offset:, :, :] = weighted_img(annotated_image[-offset:, :, :], lanes_img, alpha=0.2, beta=0.8)

    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL3:
        imshow(annotated_image)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL3:
        image1 = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('annotated_image', cv2.WINDOW_NORMAL)
        cv2.imshow('annotated_image', image1)
        cv2.waitKey(1)
        #plt.waitforbuttonpress()
    g.cold_boot = False

    return annotated_image

def remove_shadows_image_array(image_in):
    # Remove shadows
    from shadow import remove_shadow
    return remove_shadow(image_in)

def lane_detector_method_1(input_file):
    # Read Video input file
    stream = cv2.VideoCapture(input_file)
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break

        """ Important Note: To keep compatibility between movepi & VideoCapture     """
        """ VideoCapture returns RGB format while movepi returns BGR                """
        """ annotate_image_array takes BGR as input, so convertion from RGB to BGR  """
        """ is required                                                             """

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotate_image_array_1(frame)

    stream.release()
    cv2.destroyAllWindows()


def lane_detector_method_2(input_file):
    # Read Video input file
    stream = cv2.VideoCapture(input_file)
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break

        """ Important Note: To keep compatibility between movepi & VideoCapture     """
        """ VideoCapture returns RGB format while movepi returns BGR                """
        """ annotate_image_array takes BGR as input, so convertion from RGB to BGR  """
        """ is required                                                             """

        frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotate_image_array_2(frame)

    stream.release()
    cv2.destroyAllWindows()


def lane_detector_method_3(input_file):
    # Read Video input file
    stream = cv2.VideoCapture(input_file)
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break

        """ Important Note: To keep compatibility between movepi & VideoCapture     """
        """ VideoCapture returns RGB format while movepi returns BGR                """
        """ annotate_image_array takes BGR as input, so convertion from RGB to BGR  """
        """ is required                                                             """

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotate_image_array_3(frame)

    stream.release()
    cv2.destroyAllWindows()


def lane_detector_method_4(input_file):
    first_time = True
    # Read Video input file
    stream = cv2.VideoCapture(input_file)
    while(stream.isOpened()):
        ret, frame = stream.read()
        if not ret:
            break

        """ Important Note: To keep compatibility between movepi & VideoCapture     """
        """ VideoCapture returns RGB format while movepi returns BGR                """
        """ annotate_image_array takes BGR as input, so convertion from RGB to BGR  """
        """ is required                                                             """

        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        annotate_image_array_4(frame)

    stream.release()
    cv2.destroyAllWindows()

def remove_shadows(input_file):
    # Read Video input file
    stream = cv2.VideoCapture(input_file)
    while (stream.isOpened()):
        ret, frame = stream.read()
        if not ret:
            break

        """ Important Note: To keep compatibility between movepi & VideoCapture     """
        """ VideoCapture returns RGB format while movepi returns BGR                """
        """ annotate_image_array takes BGR as input, so convertion from RGB to BGR  """
        """ is required                                                             """

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        remove_shadows_image_array(frame)

    stream.release()
    cv2.destroyAllWindows()

def annotate_video_preview_1(input_file):
        """ Given input_file video, annotated video and preview """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        """ Important Note: VideoFileClip from movepi return BGR format """
        video = video.to_RGB()
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_1)
        annotated_video = annotated_video.resize(height=360)
        annotated_video.preview()


def annotate_video_preview_2(input_file):
        """ Given input_file video, annotated video and preview """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        """ Important Note: VideoFileClip from movepi return BGR format """
        video = video.to_RGB()
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_2)
        annotated_video = annotated_video.resize(height=360)
        annotated_video.preview()


def annotate_video_preview_3(input_file):
        """ Given input_file video, annotated video and preview """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        """ Important Note: VideoFileClip from movepi return BGR format """
        video = video.to_RGB()
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_3)
        annotated_video= annotated_video.resize(height=360)
        annotated_video.preview()

def annotate_video_preview_4(input_file):
        """ Given input_file video, annotated video and preview """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        """ Important Note: VideoFileClip from movepi return BGR format """
        video = video.to_RGB()
        annotated_video = video.fl_image(annotate_image_array_4)
        annotated_video = annotated_video.resize(height=360)
        annotated_video.preview()

def remove_shadows_video_preview(input_file):
        """ Given input_file video, annotated video and preview """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        """ Important Note: VideoFileClip from movepi return BGR format """
        video = video.to_RGB()
        ####video.resize(width=640)
        annotated_video = video.fl_image(remove_shadows_image_array)
        annotated_video= annotated_video.resize(height=360)
        annotated_video.preview()

def annotate_image_1(input_file, output_file):
        """ Given input_file image, save annotated image to output_file """
        annotated_image = annotate_image_array_1(mpimg.imread(input_file))
        plt.imsave(output_file, annotated_image)


def annotate_image_2(input_file, output_file):
        """ Given input_file image, save annotated image to output_file """
        annotated_image = annotate_image_array_2(mpimg.imread(input_file))
        plt.imsave(output_file, annotated_image)


def annotate_image_3(input_file, output_file):
        """ Given input_file image, save annotated image to output_file """
        annotated_image = annotate_image_array_3(mpimg.imread(input_file))
        plt.imsave(output_file, annotated_image)

def annotate_image_4(input_file, output_file):
        """ Given input_file image, save annotated image to output_file """
        if not g.trackbar_enabled:
            annotated_image = annotate_image_array_4(mpimg.imread(input_file))
            plt.imshow(annotated_image)
            plt.show()
            plt.imsave(output_file, annotated_image)
        else:
            # If trackbars are enabled, then loop on the images display in order to
            # take into account the new trackbars values
            from time import sleep
            while True:
                annotate_image_array_4(mpimg.imread(input_file))
                sleep(0.05)

        g.cold_boot = False


def remove_shadows_image(input_file, output_file):
        """ Given input_file image, save annotated image to output_file """
        g.cold_boot = True
        from time import sleep
        while True:
            annotated_image = remove_shadows_image_array(mpimg.imread(input_file))
            #plt.imsave(output_file, annotated_image)
            #plt.imshow(annotated_image)
            #plt.show()
            g.cold_boot = False
            sleep(0.05)


def annotate_video_1(input_file, output_file):
        """ Given input_file video, save annotated video to output_file """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_1)
        annotated_video.preview()
        annotated_video.write_videofile(output_file, audio=False)


def annotate_video_2(input_file, output_file):
        """ Given input_file video, save annotated video to output_file """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_2)
        annotated_video.preview()
        annotated_video.write_videofile(output_file, audio=False)


def annotate_video_3(input_file, output_file):
        """ Given input_file video, save annotated video to output_file """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_3)
        annotated_video.preview()
        annotated_video.write_videofile(output_file, audio=False)

def annotate_video_4(input_file, output_file):
        """ Given input_file video, save annotated video to output_file """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        ####video.resize(width=640)
        annotated_video = video.fl_image(annotate_image_array_4)
        annotated_video.preview()
        annotated_video.write_videofile(output_file, audio=False)

def remove_shadows_video(input_file, output_file):
        """ Given input_file video, save annotated video to output_file """
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_file)
        ####video.resize(width=640)
        annotated_video = video.fl_image(remove_shadows_image_array)
        annotated_video.preview()
        annotated_video.write_videofile(output_file, audio=False)

# End helper functions


def main():
    """Main function"""
    # construct the argument parse and parse the arguments

    # init global variables
    g.init()

    from optparse import OptionParser

    # Configure command line options
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                      help="Input video/image file")
    parser.add_option("-o", "--output_file", dest="output_file",
                      help="Output (destination) video/image file")
    parser.add_option("-m", "--method", dest="method", default='m1',
                      help="m1 -> for method 1 - m2 -> for method 2 ...")
    parser.add_option("-v", "--video", dest="video_mode", default='cv2',
                      help="video_mode choice cv2 or movepi")
    parser.add_option("-P", "--preview_only",
                      action="store_true", default=False,
                      help="preview annoted video/image")
    parser.add_option("-I", "--image_only",
                      action="store_true", dest="image_only", default=False,
                      help="Annotate image (defaults to annotating video)")
    parser.add_option("-T", "--trackbars",
                      dest="TRACKBARS", action="store_true", default=False,
                      help="Enable debug trackbars.")

    # Get and parse command line options
    options, args = parser.parse_args()
    print ('options:',options)
    print ('args:',args)

    input_file = options.input_file
    output_file = options.output_file
    image_only = options.image_only
    preview_only = options.preview_only
    video_mode = options.video_mode
    method = options.method
    g.trackbar_enabled = options.TRACKBARS

    if preview_only:
        if video_mode == 'movepi':
            if method == 'm1':
                annotate_video_preview_1(input_file)
            if method == 'm2':
                annotate_video_preview_2(input_file)
            if method == 'm3':
                annotate_video_preview_3(input_file)
            if method == 'm4':
                annotate_video_preview_4(input_file)
            if method == 'm5':
                remove_shadows_video_preview(input_file)
        if video_mode == 'cv2':
            if method == 'm1':
                lane_detector_method_1(input_file)
            if method == 'm2':
                lane_detector_method_2(input_file)
            if method == 'm3':
                lane_detector_method_3(input_file)
            if method == 'm4':
                lane_detector_method_4(input_file)
            if method == 'm5':
                remove_shadows(input_file)
        sys.exit()

    if image_only:
        if method == 'm1':
            annotate_image_1(input_file, output_file)
        if method == 'm2':
            annotate_image_2(input_file, output_file)
        if method == 'm3':
            annotate_image_3(input_file, output_file)
        if method == 'm4':
            annotate_image_4(input_file, output_file)
        if method == 'm5':
            remove_shadows_image(input_file, output_file)
    else:
        if method == 'm1':
            annotate_video_1(input_file, output_file)
        if method == 'm2':
            annotate_video_2(input_file, output_file)
        if method == 'm3':
            annotate_video_3(input_file, output_file)
        if method == 'm4':
            annotate_video_4(input_file, output_file)
        if method == 'm5':
            remove_shadows_video(input_file, output_file)



# Main script
if __name__ == '__main__':
    main()

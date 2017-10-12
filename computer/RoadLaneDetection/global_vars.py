from __future__ import print_function
from __future__ import division

import numpy as np
import pickle
import laneconfiguration as configuration
import re
import os
import inspect

from screeninfo import get_monitors
from line import Line


""" GLOBAL VARIABLES SETTINGS PART """

def init():
    # Global for Screen resolution and windows pos ...
    global s_height
    global s_win_width
    global s_win_height
    global s_win_height_offset
    m = re.findall(r'\d+',str(get_monitors()))
    if m:
        s_width = int(m[0])
        s_height = int(m[1])
    s_win_width = s_width//6
    s_win_height = s_height //4
    s_win_height_offset = 10
    print("Screen Resolution: (%d x %d) Win Resolution: (%d x %d)" % (s_width, s_height, s_win_width, s_win_height))

    # input video/image size
    global frames, vi_width, vi_height
    frames = 0
    vi_width = 0
    vi_height = 0

    # number of windows for hisrogramme
    global nwindows
    nwindows = configuration.N_WINDOWS
    # Set scale px/meter to default value
    global scale_px_width, scale_px_height
    scale_px_width = 0
    scale_px_height = 0
    lane_size_px = 0

    global cold_boot
    cold_boot = True

    # Global related to trackbar management
    global trackbar_enabled
    trackbar_enabled = False

    # Global related to region of interest trapeze ...
    global trap_bottom_width                      # width of bottom edge of trapezoid, expressed as percentage of image width
    global trap_top_width                         # ditto for top edge of trapezoid
    global trap_height                            # height of the trapezoid expressed as percentage of image height
    global trap_warped_ratio                      # ratio for perspective transform

    trap_bottom_width = configuration.TRAP_BOTTOM_WIDTH
    trap_top_width = configuration.TRAP_TOP_WIDTH
    trap_height = configuration.TRAP_HEIGHT
    trap_warped_ratio = configuration.TRAP_WARPED_RATIO

    """
    Global Variables for method relative to canny + HoughP
    """
    # Gaussian smoothing
    global kernel_size
    kernel_size = 3

    # Canny Edge Detector
    global low_threshold
    global high_threshold
    low_threshold = 50
    high_threshold = 150

    # Hough Transform
    global rho               # (2 -> 0.8) distance resolution in pixels of the Hough grid
    global theta             # angular resolution in radians of the Hough grid
    global threshold         # (15 -> 25) minimum number of votes (intersections in Hough grid cell)
    global min_line_length   # (10 -> 50) minimum number of pixels making up a line
    global max_line_gap      # (20 -> 200) maximum gap in pixels between connectable line segments

    rho = 2
    theta = 1 * np.pi/180
    threshold = 15
    min_line_length = 10
    max_line_gap = 20

    """
    Global Variable for method relative to combined threshold + birdview + curves
    """
    # Filter combination
    global combined_filter_type
    combined_filter_type = configuration.FILTER_SOBEL_MAG_DIR_HLS_HSV

    # Sobel Threshold
    global sobel_th_min
    global sobel_th_max
    sobel_th_min = 50
    sobel_th_max = 255

    # Sobel Kernel Size
    global mag_sobel_kernel_size
    mag_sobel_kernel_size = 3
    global dir_sobel_kernel_size
    dir_sobel_kernel_size = 15

    # HLS threshold
    global hls_s_th_min
    global hls_s_th_max
    hls_s_th_min = 100
    hls_s_th_max = 2555

    # RGB mak
    global rgb_lower_white, rgb_upper_white
    rgb_lower_white = np.array([200, 200, 200])
    rgb_upper_white = np.array([255, 255, 255])

    # HSV mask
    # gimp uses H = 0-360, S = 0-100 and V = 0-100. But OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    global hsv_lower_white, hsv_upper_white
    hsv_lower_white = np.array([0, 0, 184])
    hsv_upper_white = np.array([180, 16, 255])
    global hsv_lower_yellow, hsv_upper_yellow
    hsv_lower_yellow = np.array([20,68,95])
    hsv_upper_yellow = np.array([30,255,255])

    # Line curves global varaibles
    global window_size                      # how many frames for line smoothing
    global left_line
    global right_line
    global detected                         # did the fast line fit detect the lines?
    global left_curve, right_curve          # radius of curvature for left and right lanes
    global left_lane_inds, right_lane_inds  # for calculating curvature

    window_size = 5
    left_line = Line(n=window_size)
    right_line = Line(n=window_size)
    left_line.reset_fit()
    right_line.reset_fit()
    detected = False
    left_curve, right_curve = 0., 0.
    left_lane_inds, right_lane_inds = None, None

    # Detection mode
    global detect_mode, detect_fast_mode_allowed
    detect_fast_mode_allowed = False
    detect_mode = configuration.SLOW_MODE

    # Viz Degraded Mode
    global degraded_viz_mode_allowed ,degraded_viz_mode, degraded_viz_count, degraded_viz_threshold, span_y_ratio_threshold
    degraded_viz_mode_allowed = False
    degraded_viz_mode = False
    degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    degraded_viz_count = 0
    span_y_ratio_threshold = 1/2


    # PID parameters
    global Kp, Ki, Kd
    #Kp = 10
    #Ki = 1
    #Kd = 100
    Kp = 0.1
    Ki = 0.
    Kd = 0.1
    global integral, lastError, derivative
    integral = 0                          # the place where we will store our integral
    lastError = 0                         # the place where we will store the last error value
    derivative = 0                        # the place where we will store the derivative


    """
    Global Variable for camera calibration
    """
    global mtx
    global dist

    with open( os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+'/calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    print ('mtx:', mtx)
    print ('dist:', dist)

    """
    Global Variable for shadow removal
    """
    class ShadowProperties(object):
        light_threshold = {
            'automatic': 1,
            'min':0,
            'max':255
        }
        sat_threshold = {
            'automatic': 1,
            'min': 0,
            'max': 255
        }
        mask_refine_parameters = {
            'kernel_dilate_x': 15,
            'kernel_dilate_y': 15,
            'roi_max': 0}
        mask_blur = {
            'kernel_blur_x': 5,
            'kernel_blur_y':  5,
            'alpha_coef': 200}
        background = {
            'median_blur_ksize': 25,
            'morphology_auto': 1,
            'morphology_kernel_x': 5,
            'morphology_kernel_y': 30}
    global shadows
    shadows = ShadowProperties

    """
    Global Variable for inverse perspective transform
    """
    class IPMProperties(object):
        camera_parameters = {
            'fu': 800,  # 1333,
            'fv': 400,  # 425,
            'alpha': 2,  # 3,
            'cu_ratio': 50,
            'cv_ratio': 0
        }
        camera_parameters_ranges = {
            'fu': [0, 6000],
            'fv': [0, 6000],
            'alpha': [0, 90],
            'cu_ratio': [0, 100],
            'cv_ratio': [0, 100]
        }
        filter_x = {
            'filter_size_x' : 5, #40, #4,
            'sx' : 8 # 36 #8 # values is /10
        }
        filter_y = {
            'filter_size_y': 50,
            'sy': 15 #48 #30 #200, # values is /10
        }
        threshold = {
            'percentile' : 97 #92 #97
        }
        hough_parameters = {
            'min_distance' : 7,
            'min_angle' : 5,
            'max_angle' : 20 # 20
        }
        hough_p_parameters = {
            'threshold' : 25,
            'line_length' : 7,
            'line_gap' : 3
        }
        camera = None
        ground_transform = None

    global ipm
    ipm = IPMProperties

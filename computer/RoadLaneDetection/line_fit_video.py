from __future__ import division

import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import global_vars as g
import laneconfiguration as configuration
import scipy.signal

from combined_thresh import combined_thresh
#from combined_canny import combined_canny
from perspective_transform import perspective_transform
from line_fit import line_fit, tune_fit, final_viz, viz1, viz2, calc_curve, calc_vehicle_offset
from laneconfiguration import *


def parseArguments():
    """
    Parse the command line.
    @return a list of arguments
    """
    parser = argparse.ArgumentParser(
        description="Filter Curves Rendering", )

    parser.add_argument("-T", "--trackbars",
        dest="TRACKBARS", action="store_true", default=False,
      help="Enable debug trackbars.")

    return parser.parse_args()


def nothing(position):
    # print position
    pass


def peakdet(v, delta, x = None):
    """
    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def histogram_calc(image):
    plt.figure(4)
    plt.clf()
    plt.xlabel('pixel')
    plt.ylabel('Sum px')
    plt.ylim(0, image.shape[0])

    bkp_combined_filter_type = g.combined_filter_type

    g.combined_filter_type = FILTER_GRAD_MAG
    img1, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin  = combined_thresh(image)
    binary_warped1, binary_unwarped1, m1, m_inv1, src1, dst1 = perspective_transform(img1)
    histogram = np.sum(binary_warped1[binary_warped1.shape[0]//2:, :], axis=0)
    indexes = scipy.signal.find_peaks_cwt(histogram, np.arange(1, 4), max_distances=np.arange(1, 4)*2)
    indexes = np.array(indexes) - 1
    max_ind, min_ind = peakdet(histogram, 1, x = None)
    plt.plot(histogram,color = 'r')
    plt.plot(*zip(*max_ind), marker='v', color='r', ls='')


    g.combined_filter_type = FILTER_HSV_HSL
    img2, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin = combined_thresh(image)
    binary_warped2, binary_unwarped2, m2, m_inv2, src2, dst2 = perspective_transform(img2)
    histogram = np.sum(binary_warped2[binary_warped2.shape[0]//2:, :], axis=0)
    indexes = scipy.signal.find_peaks_cwt(histogram, np.arange(1, 4), max_distances=np.arange(1, 4)*2)
    indexes = np.array(indexes) - 1
    max_ind, min_ind = peakdet(histogram, 1, x = None)
    plt.plot(histogram,color = 'g')
    plt.plot(*zip(*max_ind), marker='v', color='g', ls='')



    g.combined_filter_type = FILTER_GRAD_MAG_DIR
    img3, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin  = combined_thresh(image)
    binary_warped3, binary_unwarped3, m3, m_inv3, src3, dst3 = perspective_transform(img3)
    histogram = np.sum(binary_warped3[binary_warped3.shape[0]//2:, :], axis=0)
    indexes = scipy.signal.find_peaks_cwt(histogram, np.arange(1, 4), max_distances=np.arange(1, 4)*2)
    indexes = np.array(indexes) - 1
    max_ind, min_ind = peakdet(histogram, 1, x = None)
    plt.plot(histogram,color = 'b')
    plt.plot(*zip(*max_ind), marker='v', color='b', ls='')


    g.combined_filter_type = FILTER_COLOR
    img4, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin  = combined_thresh(image)
    binary_warped4, binary_unwarped4, m4, m_inv4, src4, dst4 = perspective_transform(img4)
    histogram = np.sum(binary_warped4[binary_warped4.shape[0]//2:, :], axis=0)
    indexes = scipy.signal.find_peaks_cwt(histogram, np.arange(1, 4), max_distances=np.arange(1, 4)*2)
    indexes = np.array(indexes) - 1
    max_ind, min_ind = peakdet(histogram, 1, x = None)
    plt.plot(histogram,color = 'yellow')
    plt.plot(*zip(*max_ind), marker='v', color='yellow', ls='')


    g.combined_filter_type = FILTER_EQUALIZ
    img5, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin  = combined_thresh(image)
    binary_warped5, binary_unwarped5, m5, m_inv5, src5, dst5 = perspective_transform(img5)
    histogram = np.sum(binary_warped5[binary_warped5.shape[0]//2:, :], axis=0)
    indexes = scipy.signal.find_peaks_cwt(histogram, np.arange(1, 4), max_distances=np.arange(1, 4)*2)
    indexes = np.array(indexes) - 1
    max_ind, min_ind = peakdet(histogram, 1, x = None)
    plt.plot(histogram,color = 'cyan')
    plt.plot(*zip(*max_ind), marker='v', color='cyan', ls='')


    plt.xlabel('pixel')
    plt.ylim(0, binary_warped2.shape[0])
    plt.ylabel('Sum px')

    redPatch = mpatches.Rectangle((0, 0), 0, 0, color = 'red')
    greenPatch = mpatches.Rectangle((0, 0), 0, 0, color = 'green')
    bluePatch = mpatches.Rectangle((0, 0), 0, 0, color = 'blue')
    yellowPatch = mpatches.Rectangle((0, 0), 0, 0, color = 'yellow')
    cyanPatch = mpatches.Rectangle((0, 0), 0, 0, color = 'cyan')

    plt.legend([redPatch,greenPatch,bluePatch,yellowPatch,cyanPatch],['GRAD_MAG', 'HSV_HSL', 'GRAD_MAG_DIR','COLOR','EQUALIZ'], loc = 'best', frameon = True, fontsize = 10)
    plt.show(block=False)
    plt.pause(0.001)

    g.combined_filter_type = bkp_combined_filter_type


# MoviePy video annotation will call this function
def annotate_image(img_in):
    """
    Annotate the input image with lane line markings
    Returns annotated image
    """

    # Get Frames and Image properties
    g.frames = g.frames + 1
    g.vi_width = img_in.shape[1]
    g.vi_height = img_in.shape[0]

    # Undistort, threshold, perspective transform
    # Transforms an image to compensate for lens distortion.
    # Python: cv2.undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst
    # src - Input (distorted) image.
    # dst - Output (corrected) image that has the same size and type as src .
    # cameraMatrix - Input camera matrix A = Matrix(3,3) [[fx,0,cx], [0,fy,cy] [0,0,1]
    # distCoeffs - Input vector of distortion coefficients  (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]) of 4, 5, or 8 elements.
    #              -->  If the vector is NULL/empty, the zero distortion coefficients are assumed.
    # newCameraMatrix - Camera matrix of the distorted image. By default,
    #              -->  it is the same as cameraMatrix but you may additionally scale and shift the result by using a different matrix.
    if (RLD_ZERO_DISTORSION != True):
        undist = cv2.undistort(img_in, g.mtx, g.dist, None, g.mtx)
        # print ('DISTORSION CORRECTED undist: ', undist)
    else:
        undist = cv2.undistort(img_in, g.mtx, None, None, g.mtx)
        # print ('DISTORSION NOT CORRECTED undist: ', undist)
    # Display undistort image
    if DEBUG_LINE_FIT_VIDEO >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('undist_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('undist_image', g.s_win_width, g.s_win_height)
            cv2.moveWindow('undist_image', 0, g.s_height//2 + g.s_win_height)
        undist_bgr=cv2.cvtColor(undist, cv2.COLOR_RGB2BGR)
        cv2.imshow('undist_image', undist_bgr)

    # Combine all threshold mask on the undistort image
    #histogram_calc(undist)
    img, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin = combined_thresh(undist)
    #img = combined_canny(undist)
    if DEBUG_LINE_FIT_VIDEO >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('undist_comb_thresh', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('undist_comb_thresh', g.s_win_width, g.s_win_height)
            cv2.moveWindow('undist_comb_thresh', 3*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
        cv2.imshow('undist_comb_thresh', img)

    # Apply perpective transformation/warp binary image
    binary_warped, binary_unwarped, m, m_inv, src, dst = perspective_transform(img)
    unwarped_trapez = (np.dstack((binary_unwarped, binary_unwarped, binary_unwarped))*255).astype('uint8')
    if DEBUG_LINE_FIT_VIDEO >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('warped_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('warped_image', g.s_win_width, g.s_win_height)
            cv2.moveWindow('warped_image', 4*g.s_win_width, 3*g.s_win_height)
        cv2.imshow('warped_image', binary_warped)

        if g.cold_boot:
            cv2.namedWindow('unwarped_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('unwarped_image', g.s_win_width, g.s_win_height)
            cv2.moveWindow('unwarped_image', 3*g.s_win_width, 3*g.s_win_height)
            if g.trackbar_enabled:
                cv2.createTrackbar('top', 'unwarped_image', int(g.trap_top_width*100), 100, nothing)
                cv2.createTrackbar('bottom', 'unwarped_image', int(g.trap_bottom_width*100), 100, nothing)
                cv2.createTrackbar('height', 'unwarped_image', int(g.trap_height*100), 100, nothing)
                cv2.setTrackbarPos('top', 'unwarped_image', int(g.trap_top_width*100))
                cv2.setTrackbarPos('bottom', 'unwarped_image', int(g.trap_bottom_width*100))
                cv2.setTrackbarPos('height', 'unwarped_image', int(g.trap_height*100))
        if g.trackbar_enabled:
            l_top = cv2.getTrackbarPos('top', 'unwarped_image')
            g.trap_top_width = float(l_top)/100
            l_bottom = cv2.getTrackbarPos('bottom', 'unwarped_image')
            g.trap_bottom_width = float(l_bottom)/100
            l_height = cv2.getTrackbarPos('height', 'unwarped_image')
            g.trap_height = float(l_height)/100
        cv2.polylines(unwarped_trapez,np.int32([src]), True, (255, 255, 0), 1, 0)
        cv2.imshow('unwarped_image', unwarped_trapez)

    # Perform polynomial fit
    left_fit = None
    right_fit = None

    if not g.detected:
        # Slow line fit
        ret = line_fit(binary_warped)
        #print ('ret:', ret)
        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            g.left_lane_inds = ret['left_lane_inds']
            g.right_lane_inds = ret['right_lane_inds']

            # Get moving average of line fit coefficients
            left_fit = g.left_line.add_fit(left_fit)
            right_fit = g.right_line.add_fit(right_fit)

            # Calculate curvature
            g.left_curve, g.right_curve = calc_curve(g.left_lane_inds, g.right_lane_inds, nonzerox, nonzeroy)
            if g.detect_fast_mode_allowed:
                g.detected = True   # slow line fit always detects the line
            else:
                g.detected = False  # Force the slow mode for ever

            if DEBUG_LINE_FIT_VIDEO >= DEBUG_LEVEL2:
                viz1(binary_warped, ret, save_file=None)
                viz2(binary_warped, ret, save_file=None)
        else:
            if not g.degraded_viz_mode:
                g.detected = False

    else:  # implies g.detected == True
        # Fast line fit
        left_fit = g.left_line.get_fit()
        right_fit = g.right_line.get_fit()
        ret = tune_fit(binary_warped, left_fit, right_fit)

        # Only make updates if we detected lines in current frame
        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            g.left_lane_inds = ret['left_lane_inds']
            g.right_lane_inds = ret['right_lane_inds']

            left_fit = g.left_line.add_fit(left_fit)
            right_fit = g.right_line.add_fit(right_fit)
            g.left_curve, g.right_curve = calc_curve(g.left_lane_inds, g.right_lane_inds, nonzerox, nonzeroy)
            if DEBUG_LINE_FIT_VIDEO >= DEBUG_LEVEL2:
                #viz1(binary_warped, ret, save_file=None)
                viz2(binary_warped, ret, save_file=None)
        else:
            if not g.degraded_viz_mode:
                g.detected = False

    if ret is not None:
        vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

        # Perform final visualization on top of original undistorted image
        # print ('g.detected:',g.detected ,'g.left_curve:',g.left_curve ,'g.right_curve:',g.right_curve)
        # print ('left_fit:',left_fit ,'right_fit:',right_fit)
        result = final_viz(undist, left_fit, right_fit, m_inv, g.left_curve, g.right_curve, vehicle_offset)
    else:
        if g.degraded_viz_mode and g.left_line.len_fit() != 0 and g.right_line.len_fit() != 0:
            if left_fit is None or  right_fit is None:
                left_fit = g.left_line.get_fit()
                right_fit = g.right_line.get_fit()
            vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)
            result = final_viz(undist, left_fit, right_fit, m_inv, g.left_curve, g.right_curve, vehicle_offset)
        else:
            result = undist

    if DEBUG_LINE_FIT_VIDEO >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('final_visu', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('final_visu', 640, 480)
        cv2.putText(result, 'Filter: ' + g.combined_filter_type,
                    (1, int(3 * g.vi_height*VI_TEXT_OFFSET/VI_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, g.vi_height*VI_FONT_SIZE/VI_HEIGHT, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(result, 'Res: ' + str(g.vi_width) + 'x' + str(g.vi_height) + ' - Frame:' + str(g.frames),
                    (1, int(4 * g.vi_height*VI_TEXT_OFFSET/VI_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, g.vi_height*VI_FONT_SIZE/VI_HEIGHT, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(result, 'pX: ' + str(round(g.scale_px_width,2)) + ' mm/px' + ' l_size:' + str(round(g.lane_size_px*g.scale_px_width/1000,2)) + ' m',
                    (1, int(5 * g.vi_height*VI_TEXT_OFFSET/VI_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, g.vi_height*VI_FONT_SIZE/VI_HEIGHT, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(result, g.detect_mode + '  Recovery: ' + str(g.degraded_viz_count),
                    (1, int(6 * g.vi_height*VI_TEXT_OFFSET/VI_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, g.vi_height*VI_FONT_SIZE/VI_HEIGHT, (0, 255, 255), 1, cv2.LINE_AA)
        result_bgr=cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow('final_visu', result_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

    g.cold_boot = False

    return result


def annotate_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    from moviepy.editor import VideoFileClip
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(annotate_image)
    annotated_video= annotated_video.resize(height=360)
    annotated_video.preview()
    ####annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
    # Annotate the video

    # Initialize Global var
    g.init()

    # Parse Argument
    args = parseArguments()
    print('args:', args)
    g.trackbar_enabled = args.TRACKBARS

    """ VIDEO GROUP 1 """
    # ROI + Filter Type settings
    g.trap_bottom_width = configuration.TRAP_BOTTOM_WIDTH
    g.trap_top_width = configuration.TRAP_TOP_WIDTH
    g.trap_height = configuration.TRAP_HEIGHT
    g.trap_warped_ratio = configuration.TRAP_WARPED_RATIO
    g.combined_filter_type = configuration.FILTER_SOBEL_MAG_DIR_HLS_HSV
    g.mag_sobel_kernel_size = 9

    g.rgb_lower_white = np.array([200, 200, 200])
    g.rgb_upper_white = np.array([255, 255, 255])
    g.hsv_lower_white = np.array([0, 0, 184])
    g.hsv_upper_white = np.array([180, 16, 255])
    g.hsv_lower_yellow = np.array([20,86,95])
    g.hsv_upper_yellow = np.array([30,255,255])


    g.detected = False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = True
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/project_video.mp4', 'out.mp4')

    g.detected = False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = True
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/solidYellowLeft.mp4', 'out.mp4')

    g.detected = False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = True
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/solidWhiteRight.mp4', 'out.mp4')


    """ VIDEO GROUP 2 """
    # ROI + Filter Type settings
    g.trap_bottom_width = configuration.TRAP_BOTTOM_WIDTH_CFG1
    g.trap_top_width = configuration.TRAP_TOP_WIDTH_CFG1
    g.trap_height = configuration.TRAP_HEIGHT_CFG1
    g.trap_warped_ratio = configuration.TRAP_WARPED_RATIO_CFG1
    g.combined_filter_type = configuration.FILTER_HSV_HSL
    g.mag_sobel_kernel_size = 13

    g.rgb_lower_white = np.array([130, 200, 200])
    g.rgb_upper_white = np.array([255, 255, 255])
    g.hsv_lower_white = np.array([0, 0, 184])
    g.hsv_upper_white = np.array([180, 16, 255])
    g.hsv_lower_yellow = np.array([20,86,95])
    g.hsv_upper_yellow = np.array([30,255,255])

    g.detected = False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = True
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/challenge_video.mp4', 'out.mp4')


    """ VIDEO GROUP 3 """
    # ROI + Filter Type settings (Toulouse)
    g.trap_bottom_width = configuration.TRAP_BOTTOM_WIDTH_CFG2
    g.trap_top_width = configuration.TRAP_TOP_WIDTH_CFG2
    g.trap_height = configuration.TRAP_HEIGHT_CFG2
    g.trap_warped_ratio = configuration.TRAP_WARPED_RATIO_CFG2
    g.combined_filter_type = configuration.FILTER_GRAD_MAG
    #g.combined_filter_type = configuration.FILTER_GRAD_MAG_HSV
    g.mag_sobel_kernel_size = 3

    g.rgb_lower_white = np.array([110, 110, 110])
    g.rgb_upper_white = np.array([255, 255, 255])
    g.hsv_lower_white = np.array([0, 0, 102])
    g.hsv_upper_white = np.array([180, 16, 255])
    g.hsv_lower_yellow = np.array([20,86,95])
    g.hsv_upper_yellow = np.array([30,255,255])

    g.detected=False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = False
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.span_y_ratio_threshold = 1/3
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/MOVI0034_RocadeArcEnCiel_cropped.mp4', 'out.mp4')
    ####annotate_video('Video/MOVI0036_RocadeArcEnCiel_Gauche_cropped.mp4','out.mp4')


    """ VIDEO GROUP 4 """
    # ROI + Filter Type settings (Toulouse)
    g.trap_bottom_width = configuration.TRAP_BOTTOM_WIDTH_CFG2_1
    g.trap_top_width = configuration.TRAP_TOP_WIDTH_CFG2_1
    g.trap_height = configuration.TRAP_HEIGHT_CFG2_1
    g.trap_warped_ratio = configuration.TRAP_WARPED_RATIO_CFG2_1
    g.combined_filter_type = configuration.FILTER_GRAD_MAG
    g.mag_sobel_kernel_size = 11

    g.rgb_lower_white = np.array([110, 110, 110])
    g.rgb_upper_white = np.array([255, 255, 255])
    g.hsv_lower_white = np.array([0, 0, 102])
    g.hsv_upper_white = np.array([180, 16, 255])
    g.hsv_lower_yellow = np.array([20,86,95])
    g.hsv_upper_yellow = np.array([30,255,255])

    g.detected=False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = False
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.span_y_ratio_threshold = 1/6
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/MOVI0072_TerrePlainADroite.m4v','out.mp4')

    g.detected=False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = False
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/MOVI0063_D948_LignesJaunesEtBlanches.m4v','out.mp4')

    g.detected=False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = False
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    annotate_video('video_in/MOVI0064_D948_LigneBlancheGaucheOmbreGlissiere.m4v','out.mp4')


    """ VIDEO GROUP 5 """
    # ROI + Filter Type settings (Nice)
    g.trap_bottom_width = configuration.TRAP_BOTTOM_WIDTH_CFG3
    g.trap_top_width = configuration.TRAP_TOP_WIDTH_CFG3
    g.trap_height = configuration.TRAP_HEIGHT_CFG3
    g.trap_warped_ratio = configuration.TRAP_WARPED_RATIO_CFG3
    g.combined_filter_type = configuration.FILTER_HSV_HSL

    g.detected=False
    g.cold_boot = True
    g.frames = 0
    g.detect_fast_mode_allowed = False
    g.detect_mode = configuration.SLOW_MODE
    g.degraded_viz_mode_allowed = True
    g.degraded_viz_mode = False
    g.degraded_viz_count = 0
    g.degraded_viz_threshold = configuration.DEGRAD_VIZ_THRESHOLD
    g.left_line.reset_fit()
    g.right_line.reset_fit()
    #annotate_video('Video/original_ts.mp4', 'out.mp4')

    # Show example annotated image on screen for sanity check
    ####img_file = 'test_images/test2.jpg'
    ####img = mpimg.imread(img_file)
    ####result = annotate_image(img)
    ####result = annotate_image(img)
    ####result = annotate_image(img)
    ####plt.imshow(result)
    ####plt.show()

from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
import global_vars as g

from laneconfiguration import *


def line_fit(binary_warped):
    """
    Find and fit lane lines
    """
    g.detect_mode = SLOW_MODE
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # print('histogram:', histogram)
    if DEBUG_PLT_LINE_FIT >= DEBUG_LEVEL1:
        plt.figure(1)
        plt.ion()
        plt.clf()
        plt.title("histogram")
        plt.plot(histogram)
        plt.xlabel('pixel')
        plt.ylim(0, binary_warped.shape[0])
        plt.ylabel('Sum px')
        plt.show(block=False)
        plt.pause(0.001)

    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')

    if DEBUG_LINE_FIT >= DEBUG_LEVEL3:
        if g.cold_boot:
            cv2.namedWindow('out_image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('out_image', g.s_win_width, g.s_win_height)
            cv2.moveWindow('out_image', 5*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
        cv2.imshow('out_image', out_img)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    #print('midpoint:', midpoint, 'histogram.shape[0]:', histogram.shape[0], 'histogram.shape:', histogram.shape)
    leftx_base = np.argmax(histogram[50:midpoint]) + 50
    rightx_base = np.argmax(histogram[midpoint:-50]) + midpoint

    # Choose the number of sliding windows
    g.nwindows = N_WINDOWS
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/g.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #print('nonzero:', nonzero, 'len(nonzero):', len(nonzero) , 'nonzeroy:', nonzeroy, 'len(nonzeroy):', len(nonzeroy), 'nonzerox:', nonzerox, 'len(nonzerox):', len(nonzerox))

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    if leftx_current != 0 and rightx_current != 0:
        # 3,5 meter for each lane for french
        g.scale_px_width = LANE_WIDTH / (rightx_current - leftx_current)
        g.lane_size_px = rightx_base - leftx_base
        #print ('Video Resolution:', binary_warped.shape[1], 'x', binary_warped.shape[0])
        #print ('X(1px)= ', g.scale_px_width, 'rightx_current:', rightx_current, 'leftx_current:', leftx_current)

    # Set the width of the windows +/- margin
    #margin = 100
    margin = int(HIST_WIN_WIDTH / g.scale_px_width)
    #print ('margin:', margin)
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(g.nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # print('win_y_low:',win_y_low ,'win_y_high:',win_y_high ,'win_xleft_low:',win_xleft_low ,'win_xleft_high:',win_xleft_high ,'win_xright_low:',win_xright_low ,'win_xright_high:',win_xright_high)
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

        if DEBUG_LINE_FIT >= DEBUG_LEVEL3:
            if g.cold_boot:
                cv2.namedWindow('out_image_1', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('out_image_1', g.s_win_width, g.s_win_height)
                cv2.moveWindow('out_image_1', 4*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
            cv2.imshow('out_image_1',out_img)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        #print('good_left_inds:', good_left_inds, 'len(good_left_inds):', len(good_left_inds))
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        #print('good_right_inds:', good_right_inds, 'len(good_right_inds):', len(good_right_inds))
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    #print ('leftx:',leftx ,leftx.size, 'lefty:',lefty)
    #print ('rightx',rightx, rightx.size, 'righty:',righty)
    if (leftx.size != 0) & (lefty.size != 0) & (rightx.size !=0) & (righty.size !=0):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if not discard_line_fit_data(g.detect_mode, binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit):
            # Return a dict of relevant variables
            ret = {}
            ret['left_fit'] = left_fit
            ret['right_fit'] = right_fit
            ret['nonzerox'] = nonzerox
            ret['nonzeroy'] = nonzeroy
            ret['out_img'] = out_img
            ret['left_lane_inds'] = left_lane_inds
            ret['right_lane_inds'] = right_lane_inds
        else:
            ret =  None
    else:
        # Return None as no relevant pixel position has been found
        activate_degraded_viz_mode(g.detect_mode)
        ret = None

    return ret


def tune_fit(binary_warped, left_fit, right_fit):
    """
    Given a previously fit line, quickly try to find the line based on previous lines
    """
    g.detect_mode = FAST_MODE

    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #print('nonzeroy:', nonzeroy, 'nonzerox:', nonzerox, 'nonzerox.shape:', nonzerox.shape)
    #margin = 100
    margin = int(HIST_WIN_WIDTH / g.scale_px_width)
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    #print('left_lane_inds:', left_lane_inds, 'right_lane_inds:', right_lane_inds)

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    #print('leftx:', leftx, 'rightx:', rightx)
    #print('lefty:', lefty, 'righty:', righty)

    # If we don't find enough relevant points, return all None (this means error)
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        activate_degraded_viz_mode(g.detect_mode)
        return None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if not discard_line_fit_data(g.detect_mode, binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit):
        # Return a dict of relevant variables
        ret = {}
        ret['left_fit'] = left_fit
        ret['right_fit'] = right_fit
        ret['nonzerox'] = nonzerox
        ret['nonzeroy'] = nonzeroy
        ret['out_img'] = out_img
        ret['left_lane_inds'] = left_lane_inds
        ret['right_lane_inds'] = right_lane_inds
    else:
        ret = None

    return ret


def discard_line_fit_data(mode, binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit):
    """ Affine data line fit """
    discard_data = False
    """ Return immedialty if degraded lode viz not allowed """

    if not g.degraded_viz_mode_allowed:
       return discard_data

    # Check-1 span length y axis
    max_value_lefty = max(lefty)
    min_value_lefty = min(lefty)
    span_lefty = max_value_lefty - min_value_lefty
    max_value_righty = max(righty)
    min_value_righty = min(righty)
    span_righty = max_value_righty - min_value_righty
    #print(mode, 'span_lefty:', span_lefty, 'span_righty:', span_righty)
    if span_lefty < int(binary_warped.shape[0] * g.span_y_ratio_threshold)  or span_righty < int(binary_warped.shape[0] * g.span_y_ratio_threshold):
        g.degraded_viz_count = g.degraded_viz_count + 1
        if g.degraded_viz_count < g.degraded_viz_threshold:
            g.degraded_viz_mode = True
        else:
            g.degraded_viz_mode = False
        print(mode,'BAD: Y SPAN TOO SMALL', 'span_lefty:', span_lefty, 'span_righty:', span_righty, 'degraded mode:', g.degraded_viz_mode, 'degraded count:', g.degraded_viz_count)
        return True

    # Check-2 span length x axis
    # To be completed
    base_leftx = left_fit[0]*(binary_warped.shape[0]-1)**2 + left_fit[1]*(binary_warped.shape[0]-1) + left_fit[2]
    base_rightx = right_fit[0]*(binary_warped.shape[0]-1)**2 + right_fit[1]*(binary_warped.shape[0]-1) + right_fit[2]
    dist_base = int(base_rightx - base_leftx)
    min_dist_base = int(0.5 * g.trap_bottom_width*binary_warped.shape[1]) # 50% threshold of the bottom roi
    if dist_base < min_dist_base:
        g.degraded_viz_count = g.degraded_viz_count + 1
        if g.degraded_viz_count < g.degraded_viz_threshold:
            g.degraded_viz_mode = True
        else:
            g.degraded_viz_mode = False
        print(mode, 'BAD: X SPAN BASE' , 'base_leftx:', int(base_leftx), 'base_rightx:', int(base_rightx), 'dist_base:', dist_base,
              'base_bottom_width:', int(g.trap_bottom_width*binary_warped.shape[1]), 'degraded mode:', g.degraded_viz_mode, 'degraded count:', g.degraded_viz_count)
        return True

    # Check-3 Intersection of Left/Right second degree polynome
    coeff = [left_fit[0] - right_fit[0], left_fit[1] - right_fit[1], left_fit[2] - right_fit[2]]
    res = np.roots(coeff)
    if isinstance(res[0], complex) or isinstance(res[1], complex):
        g.degraded_viz_mode = False
        g.degraded_viz_count = 0
        #print(mode, 'GOOD: LEFT/RIGHT DISTINCT LINES', 'len(res):', len(res), 'res:', res)
    else:
        if 0 <= res[0] < binary_warped.shape[0]  or 0 <= res[1] < binary_warped.shape[0]:
            g.degraded_viz_count = g.degraded_viz_count + 1
            if g.degraded_viz_count < g.degraded_viz_threshold:
                g.degraded_viz_mode = True
            else:
                g.degraded_viz_mode = False
            print(mode, 'BAD: LEFT/RIGHT CROSSED LINES', 'len(res):', len(res), 'res:', res, 'degraded mode:', g.degraded_viz_mode, 'degraded count:', g.degraded_viz_count)
            return True
        else:
            g.degraded_viz_mode = False
            g.degraded_viz_count = 0
            #print(mode, 'GOOD: LEFT/RIGHT DISTINCT LINES', 'len(res):', len(res), 'res:', res)

    return discard_data


def activate_degraded_viz_mode(mode):
   if g.degraded_viz_mode_allowed:
        g.degraded_viz_count = g.degraded_viz_count + 1
        if g.degraded_viz_count < g.degraded_viz_threshold:
            g.degraded_viz_mode = True
        else:
            g.degraded_viz_mode = False
        print(mode, 'BAD: NOT ENOUGH PIXELs', 'degraded mode:', g.degraded_viz_mode, 'degraded count:', g.degraded_viz_count)


def viz1(binary_warped, ret, save_file=None):
    """
    Visualize each sliding window location and predicted lane lines, on binary warped image
    save_file is a string representing where to save the image (if None, then just display)
    """
    # Grab variables from ret dictionary
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    out_img = ret['out_img']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if DEBUG_LINE_FIT >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('result1', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result1', g.s_win_width, g.s_win_height)
            cv2.moveWindow('result1', 4*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
        cv2.namedWindow('result1', cv2.WINDOW_NORMAL) # Need to find service to check if an opencv window is created
        cv2.resizeWindow('result1', g.s_win_width, g.s_win_height)
        cv2.moveWindow('result1', 4*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
        cv2.imshow('result1', out_img)

    if DEBUG_PLT_LINE_FIT >= DEBUG_LEVEL1:
        plt.figure(2)
        plt.clf()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        left_fit_avg = g.left_line.get_fit()
        right_fit_avg = g.right_line.get_fit()
        left_fitx_avg = left_fit_avg[0]*ploty**2 + left_fit_avg[1]*ploty + left_fit_avg[2]
        right_fitx_avg = right_fit_avg[0]*ploty**2 + right_fit_avg[1]*ploty + right_fit_avg[2]
        plt.plot(left_fitx_avg, ploty, color='magenta')
        plt.plot(right_fitx_avg, ploty, color='magenta')

        plt.xlim(0, g.vi_width)
        plt.ylim(g.vi_height,0)
        if save_file is None:
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.savefig(save_file)
        #plt.gcf().clear()


def viz2(binary_warped, ret, save_file=None):
    """
    Visualize the predicted lane lines with margin, on binary warped image
    save_file is a string representing where to save the image (if None, then just display)
    """
    # Grab variables from ret dictionary
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    #margin = 100  # Note: Keep this in sync with *_fit()
    margin = int(HIST_WIN_WIDTH / g.scale_px_width)
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    if DEBUG_LINE_FIT >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result2', g.s_win_width, g.s_win_height)
            cv2.moveWindow('result2', 5*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
        cv2.namedWindow('result2', cv2.WINDOW_NORMAL) # Need to find service to check if an opencv window is created
        cv2.resizeWindow('result2', g.s_win_width, g.s_win_height)
        cv2.moveWindow('result2', 5*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
        cv2.imshow('result2', result)

    if DEBUG_PLT_LINE_FIT >= DEBUG_LEVEL1:
        plt.figure(3)
        plt.ion()
        plt.clf()
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        left_fit_avg = g.left_line.get_fit()
        right_fit_avg = g.right_line.get_fit()
        left_fitx_avg = left_fit_avg[0]*ploty**2 + left_fit_avg[1]*ploty + left_fit_avg[2]
        right_fitx_avg = right_fit_avg[0]*ploty**2 + right_fit_avg[1]*ploty + right_fit_avg[2]
        plt.plot(left_fitx_avg, ploty, color='magenta')
        plt.plot(right_fitx_avg, ploty, color='magenta')

        plt.xlim(0, g.vi_width)
        plt.ylim(g.vi_height,0)
        if save_file is None:
            plt.show()
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.savefig(save_file)
        #plt.gcf().clear()


def calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    """
    Calculate radius of curvature in meters
    """
    y_eval = float(g.vi_height) -1  # Ex 720p video/image, so last (lowest on screen) y index is 719

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30./720 # meters per pixel in y dimension
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension
    xm_per_pix = g.scale_px_width/1000

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    return left_curverad, right_curverad


def calc_vehicle_offset(undist, left_fit, right_fit):
    """
    Calculate vehicle offset from lane center, in meters
    """
    # Calculate vehicle center offset in pixels
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension
    xm_per_pix = g.scale_px_width/1000
    vehicle_offset *= xm_per_pix

    return vehicle_offset


def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
    """
    Final lane line prediction visualized and overlayed on top of original image
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    # warp_zero = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # print ('undist.shape[0]: ',undist.shape[0] ,'undist.shape[1]: ',undist.shape[1])
    # color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions
    color_warp = np.zeros((undist.shape[0], undist.shape[1], 3), dtype='uint8')

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    if DEBUG_LINE_FIT >= DEBUG_LEVEL3:
        cv2.namedWindow('warped_zone', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('warped_zone', g.s_win_width, g.s_win_height)
        cv2.moveWindow('warped_zone', 4*g.s_win_width, g.s_win_height - g.s_win_height//2)
        cv2.imshow('warped_zone', color_warp)


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    if DEBUG_LINE_FIT >= DEBUG_LEVEL3:
        cv2.namedWindow('unwarped_zone', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('unwarped_zone', g.s_win_width, g.s_win_height)
        cv2.moveWindow('unwarped_zone', 5*g.s_win_width, g.s_win_height - g.s_win_height//2)
        cv2.imshow('unwarped_zone', newwarp)

    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Annotate lane curvature values and vehicle offset from center
    avg_curve = (left_curve + right_curve)/2
    label_str = 'Radius of curvature: %.1f m' % avg_curve
    result = cv2.putText(result, label_str, (1, int(1 * g.vi_height*VI_TEXT_OFFSET/VI_HEIGHT)),
                         cv2.FONT_HERSHEY_SIMPLEX, g.vi_height*VI_FONT_SIZE/VI_HEIGHT, (0, 0, 0), 1, cv2.LINE_AA)

    label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
    result = cv2.putText(result, label_str, (1, int(2 * g.vi_height*VI_TEXT_OFFSET/VI_HEIGHT)),
                         cv2.FONT_HERSHEY_SIMPLEX, g.vi_height*VI_FONT_SIZE/VI_HEIGHT, (0, 0, 0), 1, cv2.LINE_AA)

    return result

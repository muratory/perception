from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import global_vars as g

from laneconfiguration import *

def nothing(position):
    # print position
    pass

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Takes an image, gradient orientation, and threshold min/max values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Return the direction of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_thresh(img, thresh=(100, 255)):
    """
    Convert RGB to HLS and threshold to binary image using S channel
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def hsv_thresh(img):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    lower_white = g.rgb_lower_white
    upper_white = g.rgb_upper_white
    white_mask = cv2.inRange(img, lower_white, upper_white)
    white_image = cv2.bitwise_and(img, img, mask=white_mask)

    # Filter White pixels based on HSV color plan
    hsv_w = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_white = g.hsv_lower_white
    upper_white = g.hsv_upper_white
    white_mask = cv2.inRange(hsv_w, lower_white, upper_white)
    white_image = cv2.bitwise_and(img, img, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = g.hsv_lower_yellow
    upper_yellow = g.hsv_upper_yellow
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(img, img, mask=yellow_mask)

    # Combine the two above images
    img1 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    # Combine the two masks
    img1_gray=cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    binary_output =  np.zeros_like(img1_gray)
    binary_output[(img1_gray>0)]=1

    return  binary_output

def filter_colors(img):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 175  # Initial 200 -> 130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(img, lower_white, upper_white)
    white_image = cv2.bitwise_and(img, img, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower_yellow = np.array([90,100,100])
    lower_yellow = np.array([89,95,95])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(img, img, mask=yellow_mask)

    # Combine the two above images
    img1 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    # Combine the two masks
    img1_gray=cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    binary_output =  np.zeros_like(img1_gray)
    binary_output[(img1_gray>0)]=1

    return  binary_output

def filter_equaliz(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    filter_gray = cv2.bilateralFilter(gray,9,50,50)
    gray_adapt_thresh = cv2.adaptiveThreshold(filter_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Opening (erode/dilate = noise cleanup)
    kernel = np.ones((3,3),np.uint8)
    erode_frame = cv2.morphologyEx(gray_adapt_thresh, cv2.MORPH_OPEN, kernel)
    binary_output =  np.zeros_like(erode_frame)
    binary_output[(erode_frame>0)]=1

    return  binary_output


def combined_thresh(img):
    # Print Input image before applying different threshold
    if DEBUG_COMBINED_THRESHOLD >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('input_img',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('input_img', g.s_win_width, g.s_win_height)
            cv2.moveWindow('input_img', 0, g.s_height//2 - g.s_win_height//2)
        img_bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) 
        cv2.imshow('input_img',img_bgr)

    # Compute binary mask based on sobel thresholding 
    abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=g.sobel_th_min, thresh_max=g.sobel_th_max)
    # Display Binary Mask of sobel thresholding
    if DEBUG_COMBINED_THRESHOLD >= DEBUG_LEVEL2:
        #print('abs_bin: ', abs_bin)
        if g.cold_boot:
            cv2.namedWindow('sobel_filter',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('sobel_filter', g.s_win_width, g.s_win_height)
            cv2.moveWindow('sobel_filter', 2*g.s_win_width, 2*g.s_win_height - g.s_win_height//2 + 2*g.s_win_height_offset)
            if g.trackbar_enabled:
                cv2.createTrackbar('th_min', 'sobel_filter', g.sobel_th_min, 255, nothing)
                cv2.createTrackbar('th_max', 'sobel_filter', g.sobel_th_max, 255, nothing)
                cv2.setTrackbarPos('th_min', 'sobel_filter', g.sobel_th_min)
                cv2.setTrackbarPos('th_max', 'sobel_filter', g.sobel_th_max)
        if g.trackbar_enabled:
            g.sobel_th_min = cv2.getTrackbarPos('th_min', 'sobel_filter')
            g.sobel_th_max = cv2.getTrackbarPos('th_max', 'sobel_filter')
        cv2.imshow('sobel_filter',abs_bin*255)

    # Compute binary mask based on magnitude of gradient
    mag_bin = mag_thresh(img, sobel_kernel=g.mag_sobel_kernel_size, mag_thresh=(50, 255))
    # Display binary mask of magnitude of gradient
    if DEBUG_COMBINED_THRESHOLD >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('mag_thresh', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('mag_thresh', g.s_win_width, g.s_win_height)
            cv2.moveWindow('mag_thresh', g.s_win_width, 0)
            if g.trackbar_enabled:
                cv2.createTrackbar('k_sz', 'mag_thresh', g.mag_sobel_kernel_size, 31, nothing)
                cv2.setTrackbarPos('k_sz', 'mag_thresh', g.mag_sobel_kernel_size)
        if g.trackbar_enabled:
            k_size = cv2.getTrackbarPos('k_sz', 'mag_thresh')
            if (k_size % 2 == 1):
                g.mag_sobel_kernel_size = k_size
            else:
                g.mag_sobel_kernel_size = k_size + 1
            cv2.setTrackbarPos('k_sz', 'mag_thresh', g.mag_sobel_kernel_size)
        cv2.imshow('mag_thresh', mag_bin*255)

    # Compute binary mask based on direction of gradient based on a sobel kernel
    dir_bin = dir_threshold(img, sobel_kernel=g.dir_sobel_kernel_size, thresh=(0.7, 1.3))
    # Display binary mask result
    if DEBUG_COMBINED_THRESHOLD >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('dir_thresh',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('dir_thresh', g.s_win_width, g.s_win_height)
            cv2.moveWindow('dir_thresh', g.s_win_width , g.s_win_height + g.s_win_height_offset)
            if g.trackbar_enabled:
                cv2.createTrackbar('k_sz', 'dir_thresh', g.dir_sobel_kernel_size, 31, nothing)
                cv2.setTrackbarPos('k_sz', 'dir_thresh', g.dir_sobel_kernel_size)
        if g.trackbar_enabled:
            k_size = cv2.getTrackbarPos('k_sz', 'dir_thresh')
            if (k_size % 2 == 1):
                g.dir_sobel_kernel_size = k_size
            else:
                g.dir_sobel_kernel_size = k_size + 1
            cv2.setTrackbarPos('k_sz', 'dir_thresh', g.dir_sobel_kernel_size)
        cv2.imshow('dir_thresh', dir_bin*255)

        if g.cold_boot:
            cv2.namedWindow('mag_and_dir',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('mag_and_dir', g.s_win_width, g.s_win_height)
            cv2.moveWindow('mag_and_dir', 2*g.s_win_width, g.s_win_height - g.s_win_height//2)
        combined_mag_and_dir= np.zeros_like(dir_bin)
        combined_mag_and_dir[((mag_bin == 1) & (dir_bin == 1))] = 1
        cv2.imshow('mag_and_dir',combined_mag_and_dir*255)

    # Compute binary mask based on hls thresholding/mask
    hls_bin = hls_thresh(img, thresh=(g.hls_s_th_min,g.hls_s_th_max))
    # Display binary mask of hls thresholding/mask
    if DEBUG_COMBINED_THRESHOLD >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('hls_bin',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hls_bin', g.s_win_width, g.s_win_height)
            cv2.moveWindow('hls_bin', g.s_win_width, 2*g.s_win_height + 2*g.s_win_height_offset)
            if g.trackbar_enabled:
                cv2.createTrackbar('s_min', 'hls_bin', g.hls_s_th_min, 255, nothing)
                cv2.createTrackbar('s_max', 'hls_bin', g.hls_s_th_max, 255, nothing)
                cv2.setTrackbarPos('s_min', 'hls_bin', g.hls_s_th_min)
                cv2.setTrackbarPos('s_max', 'hls_bin', g.hls_s_th_max)
        if g.trackbar_enabled:
            g.hls_s_th_min = cv2.getTrackbarPos('s_min', 'hls_bin')
            g.hls_s_th_max = cv2.getTrackbarPos('s_max', 'hls_bin')
            if g.hls_s_th_min > g.hls_s_th_max:
                g.hls_s_th_max = g.hls_s_th_min
            if g.hls_s_th_max < g.hls_s_th_min:
                g.hls_s_th_min = g.hls_s_th_max
            cv2.setTrackbarPos('s_min', 'hls_bin', g.hls_s_th_min)
            cv2.setTrackbarPos('s_max', 'hls_bin', g.hls_s_th_max)
        cv2.imshow('hls_bin',hls_bin*255)

    # Compute binary mask based on hsv thresholding/mask
    hsv_bin = hsv_thresh(img)
    # Display binary mask of hsv thresholding/mask
    if DEBUG_COMBINED_THRESHOLD >= DEBUG_LEVEL2:
        if g.cold_boot:
            cv2.namedWindow('hsv_bin',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hsv_bin', g.s_win_width, g.s_win_height)
            cv2.moveWindow('hsv_bin', g.s_win_width , 3*g.s_win_height + 2*g.s_win_height_offset)
        cv2.imshow('hsv_bin',hsv_bin*255)

        if g.cold_boot:
            cv2.namedWindow('hls_or_hsv',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hls_or_hsv', g.s_win_width, g.s_win_height)
            cv2.moveWindow('hls_or_hsv', 2*g.s_win_width, 3*g.s_win_height)
        combined_hls_and_hsv= np.zeros_like(dir_bin)
        combined_hls_and_hsv[((hls_bin == 1) | (hsv_bin == 1))] = 1
        cv2.imshow('hls_or_hsv',combined_hls_and_hsv*255)

    # Compute binary mask based on color filter
    #col_bin = filter_colors(img)

    # Compute binary mask based on equalization filter
    #equ_bin = filter_equaliz(img)
    # Display binary mask based on equalization

    combined = np.zeros_like(dir_bin)
    if g.combined_filter_type == 'GRAD_MAG':
        combined[(mag_bin == 1  )] = 1
    if g.combined_filter_type == 'GRAD_MAG_DIR':
        combined[(mag_bin == 1) & (dir_bin == 1)] = 1

    if g.combined_filter_type == 'GRAD_MAG_HSV':
        combined[(mag_bin == 1) | (hsv_bin == 1)] = 1

    if g.combined_filter_type == 'HLS_OR_HSV':
        combined[((hls_bin == 1) | (hsv_bin == 1))] = 1

    if g.combined_filter_type == 'COLOR':
        print('NOT ACTIVATED')
        #combined[(col_bin == 1) ] = 1
    if g.combined_filter_type == 'EQUALIZ':
        print('NOT ACTIVATED')
        #combined[(equ_bin == 1) ] = 1

    if g.combined_filter_type == 'MAG_DIR_HSV':
        combined[(((mag_bin == 1) & (dir_bin == 1))) | (hsv_bin == 1)] = 1

    if g.combined_filter_type == 'MAG_DIR_HLS_HSV':
        combined[(((mag_bin == 1) & (dir_bin == 1))) | ((hls_bin == 1) | (hsv_bin == 1))] = 1

    if g.combined_filter_type == 'DEFAULT' or g.combined_filter_type == 'SOBEL_MAG_DIR_HLS_HSV':
        combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | ((hls_bin == 1) | (hsv_bin == 1))] = 1

    return combined, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin  # DEBUG


if __name__ == '__main__':
    img_file = 'test_images/straight_lines1.jpg'
    img_file = 'test_images/test5.jpg'

    g.init()
    img = mpimg.imread(img_file)

    img = cv2.undistort(img, g.mtx, g.dist, None, g.mtx)

    combined, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin = combined_thresh(img)

    plt.subplot(2, 3, 1)
    plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 2)
    plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 5)
    plt.imshow(img)
    plt.subplot(2, 3, 6)
    plt.imshow(combined, cmap='gray', vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()

from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
import glob


def parseArguments():
    """
    Parse the command line.
    @return a list of arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute Calibration Parameters For Camera.", )

    parser.add_argument("-f", "--folder", dest="folder", required=True,
                        help="Chess Board Reference Images Used For Calibration")

    parser.add_argument("-a", "--alpha", dest="alpha", default = 0.05, required=False,
                        help="choose alpha value from range [0. : 1.] for Optimal Camerai Matrix")

    return parser.parse_args()


def calibrate_camera(folder):
    # Mapping each calibration image to number of checkerboard corners
    # Everything is (9,6) for now

    # List of object points and corners for calibration
    objp_list = []
    corners_list = []
    
    # Go through all images and find corners
    for image_path in glob.glob("*.bmp"):
        print image_path
        #chess board is at least a 9x6 . need to choose here a pair,unpair number
        nx, ny = (7,6)
        # print ('k: ', k, 'nx: ',nx ,'ny: ',ny)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, save & draw corners
        if ret == True:
            # Save object points and corresponding corners
            #print ('objp: ', objp, 'corners: ', corners)
            objp_list.append(objp)
            corners_list.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            plt.show(block=False)
            plt.pause(0.001)

            print('Found corners for %s' % image_path)
        else:
            print('Warning: ret = %s for %s' % (ret, image_path))

    # Calibrate camera and undistort a test image
    img = cv2.imread(folder + '/calibration1.bmp')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size,None,None)

    return mtx, dist


if __name__ == '__main__':

    args = parseArguments()
    calib_folder = args.folder
    alpha = float(args.alpha)
    print('args:',args)
    print('Chess Board Reference Images Folder:', calib_folder)

    mtx, dist = calibrate_camera(calib_folder)
    save_dict = {'mtx': mtx, 'dist': dist}
    # to keep compatibility with python 2 add protocol=2
    with open(calib_folder+'/calibrate_gps_camera.p', 'wb') as f:
        pickle.dump(save_dict, f, protocol=2)

    # Undistort example calibration image
    img = mpimg.imread(calib_folder + '/calibration1.bmp')

    # refine the camera Matrix / Alpha can be be chnaged from 0 to 1 in our case 0.1 is enough
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.namedWindow('Before Crop', cv2.WINDOW_NORMAL)
    cv2.imshow('Before Crop', dst)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # save image
    cv2.imwrite(calib_folder+'/undistort_gps_calibration.png', dst)

    plt.imshow(dst)
    plt.show(block=False)
    plt.pause(0.001)

    plt.savefig(calib_folder+'/undistort_gps_calibration1.png')

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', dst)
    key = cv2.waitKey(0) & 0xF

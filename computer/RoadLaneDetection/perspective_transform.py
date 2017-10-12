from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import global_vars as g
from combined_thresh import combined_thresh


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    poly_pts = np.zeros((4, 2), dtype = "int32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    print ('s:',s)
    poly_pts[0] = pts[np.argmin(s)]
    poly_pts[2] = pts[np.argmax(s)]
    print ('poly_pts:', poly_pts)
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    print ('diff:', diff)
    poly_pts[1] = pts[np.argmin(diff)]
    poly_pts[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return poly_pts


def perspective_transform(img):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[(int((img.shape[1] * (1 - g.trap_bottom_width)) // 2), int(img.shape[0]))],
        [(int(img.shape[1] - (img.shape[1] * (1 - g.trap_bottom_width)) // 2), int(img.shape[0]))],
        [(int(img.shape[1] - (img.shape[1] * (1 - g.trap_top_width)) // 2), int(img.shape[0] - img.shape[0] * g.trap_height))],
        [(int((img.shape[1] * (1 - g.trap_top_width)) // 2), int(img.shape[0] - img.shape[0] * g.trap_height))]])
    # print ('src:',src)

    dst = np.float32(
        [[(int((img.shape[1] * (1 - g.trap_warped_ratio)) // 2), int(img.shape[0]))],
        [(int(img.shape[1] - (img.shape[1] * (1 - g.trap_warped_ratio)) // 2), int(img.shape[0]))],
        [(int(img.shape[1] - (img.shape[1] * (1 - g.trap_warped_ratio)) // 2), int(0))],
        [(int((img.shape[1] * (1 - g.trap_warped_ratio)) // 2), int(0))]])
    # print ('dst:',dst)

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
    cv2.polylines(unwarped,np.int32([src]), True, (255, 255, 255))

    return warped, unwarped, m, m_inv, src, dst


if __name__ == '__main__':
    img_file = 'test_images/test5.jpg'

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)

    warped, unwarped, m, m_inv = perspective_transform(img)

    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.show()

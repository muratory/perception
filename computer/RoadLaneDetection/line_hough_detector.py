# import the necessary packages
import numpy as np
import argparse
import glob
import cv2


def _auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged


def _canny_filter(image,mode): 
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    if mode == 'WIDE_THRESHOLD':
        img = cv2.Canny(blurred, 10, 200)
    elif mode == 'TIGHT_THRESHOLD':
        img = cv2.Canny(blurred, 225, 250)
    elif mode == 'AUTO_THRESHOLD':
        img = _auto_canny(blurred)

    return img


#ref: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
def hough_line_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    ####cv2.imshow("edges",edges)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 175)
    return lines


def hough_lineP_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    ####cv2.imshow("edgesP",edges)

    minLineLength = 100
    maxLineGap = 10

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength, maxLineGap)
    return lines


 
def main():
    """Main function"""

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
            help="path to input dataset of images")
    args = vars(ap.parse_args())

    # loop over the images
    for image_path in glob.glob(args["images"] + "/*.jpg"):
        image_orig = cv2.imread(image_path)

        image = cv2.imread(image_path)
        imageP = cv2.imread(image_path)

        lines = hough_line_detector(image)
        print ('lines:', lines, 'lines.shape:', lines.shape)
        linesP = hough_lineP_detector(imageP)
        print ('linesP:', linesP, 'linesP.shape:', linesP.shape)

        # show the images
        for i in range(lines.shape[0]):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)

                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for i in range(linesP.shape[0]):
            for x1, y1, x2, y2 in linesP[i]:
                cv2.line(imageP, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Line Dectector",  np.hstack([image, imageP]))

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break


if __name__ == '__main__':
    main()
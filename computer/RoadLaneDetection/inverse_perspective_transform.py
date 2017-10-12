import numpy as np
import cv2
from matplotlib import pyplot as plt

from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import global_vars as g
from laneconfiguration import *
from util import display_debug_image

'''
    A. Inverse Perspective Mapping
'''

class Camera():
    def __init__(self, img_width, img_height, h=1600, fu=400, fv=300, alpha=14, beta=0, cu_ratio=50, cv_ratio=50):
        self.img_width, self.img_height, = img_width, img_height

        # By default cu, cv point to the center of the image (ratio 50 %)
        cu = self.img_width * cu_ratio / 100.0
        cv = self.img_height * cv_ratio / 100.0

        #h = 1600.0  # height (m)
        #fu = 400.0  # horizontal focal length
        #fv = 300.0  # vertical focal lenght
        alpha = np.deg2rad(alpha) # pitch angle
        beta = np.deg2rad(beta) # yaw angle

        c1 = np.cos(alpha)
        s1 = np.sin(alpha)
        c2 = np.cos(beta)
        s2 = np.sin(beta)

        self.Tig = h * np.array([[-c2 / fu, s1 * s2 / fv, (cu / fu) * c2 - (cv / fv) * s1 * s2 - c1 * s2, 0],
                                 [s2 / fu, s1 * c2 / fv, -(cu / fu) * s2 - (cv / fv) * s1 * c2 - c1 * c2, 0],
                                 [0, c1 / fv, -(cv / fv) * c1 + s1, 0],
                                 [0, -c1 / (h * fv), (cv * c1) / (h * fv) - s1 / h, 0]])

        self.Tgi = np.array([[fu * c2 + cu * c1 * s2, cu * c1 * c2 - s2 * fu, -cu * s1, 0],
                             [s2 * (cv * c1 - fv * s1), c2 * (cv * c1 - fv * s1), -fv * c1 - cv * s1, 0],
                             [c1 * s2, c1 * c2, -s1, 0],
                             [c1 * s2, c1 * c2, -s1, 0]])


        self.corners = np.array([[0, 0],
                                 [self.img_width, 0],
                                 [self.img_width, self.img_height],
                                 [0, self.img_height]])

        if DEBUG_LEVEL >= DEBUG_LEVEL2:
            print("camera image corners coordinates (pixels):")
            print(self.corners)

class GroundImage():

    def __init__(self, camera, target_width=200):
        self.camera = camera
        self.target_width = target_width

        ground_corners = self.image_corners_to_ground_reference()

        if DEBUG_LEVEL >= DEBUG_LEVEL2:
            print("Projection of corners on the ground (real coordinates):")
            print(ground_corners.astype(np.int32)[0:4,0:2])

        # compute coordinates inside the ground corners
        ground_coordinates = self.compute_ground_real_coordinates(ground_corners)

        if DEBUG_LEVEL >= DEBUG_LEVEL2:
            print("Ground coordinates extrema (real coordinates):")
            print("  x: ", ground_coordinates[0, 0], ground_coordinates[0, -1])
            print("  y: ", ground_coordinates[1, 0], ground_coordinates[1, -1])

        # map ground pixels to camera image pixels
        self.map_ground_coordinates_to_camera_image_coordinates(ground_coordinates)

        if DEBUG_LEVEL >= DEBUG_LEVEL2:
            print("Corresponding corners coordinates in source image (pixels):")
            print(self.coords_i_mat[0, 0, 0:2].astype(np.int32))
            print(self.coords_i_mat[0, self.target_width-1, 0:2].astype(np.int32))
            print(self.coords_i_mat[self.target_height-1, 0, 0:2].astype(np.int32))
            print(self.coords_i_mat[self.target_height-1, self.target_width-1, 0:2].astype(np.int32))

        # prepare output image
        self.Pg = np.zeros((self.target_height, self.target_width))

    # compute image corners from Camera Reference to Ground Reference
    def image_corners_to_ground_reference(self):
        assert(self.camera.corners.shape == (4, 2))

        # homogeneous coordinates for corners coords
        image_corners_HC = np.c_[self.camera.corners, np.ones((4, 2))]

        # Use Tig transform matrix on image coords to get ground coords
        ground_corners = np.matmul(self.camera.Tig, image_corners_HC.T)
        ground_corners = ground_corners.T

        # Divide by scaling factor
        v = ground_corners[:,2]
        ground_corners = ground_corners/v[:,None]

        return ground_corners[:,0:2]

    def compute_ground_real_coordinates(self, corners_real):

        # The corners of the ground image represent a trapezoidal shape.
        # Here we only select an inner rectangle inside this trapeze, with the
        # same height and the width corresponding to the minimum width of the
        # trapeze (i.e. foreground of the camera) multiplied by a scale factor.

        ground_min_y, ground_max_y = np.min(corners_real[:,1]), np.max(corners_real[:,1])
        ground_height = ground_max_y - ground_min_y

        # The 2 last corners are in the foreground
        ground_min_x, ground_max_x = corners_real[2:,0]
        ground_width = ground_max_x - ground_min_x

        # Scale the width to get a wider field of view
        width_scale_factor = 6
        ground_min_x = ground_min_x * width_scale_factor
        ground_width = ground_width * width_scale_factor

        ground_x_y_ratio = ground_height / ground_width

        # Find the englobing rectangular shape
        self.target_height = int(self.target_width * ground_x_y_ratio / 3)

        # Ground image pixels
        # coords_g_pixels : coords des pixels dans l'image ground

        # Generate grid of coordinates for the target ground image
        coords_g_pixels = np.mgrid[0:self.target_height, 0:self.target_width].reshape(2,-1)
        coords_g_pixels = coords_g_pixels[::-1] # inversion ligne 1 et 2 pour que les X soient en premier et les Y en 2 eme
        coords_g_pixels = np.r_[coords_g_pixels, np.ones((1, self.target_height * self.target_width))]

        '''
        Homothetie matrice: pour passer dans le repere ground reel
        coords_g : coords ground reelles
        '''
        pixels_coords_to_real_coords_mat = \
            np.array([[ground_width/self.target_width,                           0,  ground_min_x],
                      [0,                         ground_height/self.target_height,  ground_min_y]])


        coords_g_real = np.matmul(pixels_coords_to_real_coords_mat, coords_g_pixels)
        coords_g_real = np.r_[coords_g_real, np.ones((2, self.target_height * self.target_width))]
        return coords_g_real

    '''
    coords_i : passage repere camera
    '''

    def map_ground_coordinates_to_camera_image_coordinates(self, coords_g_real):
        coords_i = np.matmul(self.camera.Tgi, coords_g_real)
        coords_i_div = coords_i.T
        v = coords_i_div[:, 2]
        coords_i_div = coords_i_div / v[:, None]

        coords_i_mat = coords_i_div.reshape((self.target_height, self.target_width, 4))

        # Only take X and Y (2 other dims are just 1 and 1)
        coords_i_mat = coords_i_mat[:, :, 0:2]

        x = coords_i_mat[:, :, 0]
        y = coords_i_mat[:, :, 1]

        # flip X
        x = self.camera.img_width - x

        coords_i_mat[:, :, 0] = x
        coords_i_mat[:, :, 1] = y
        self.coords_i_mat = coords_i_mat

        # Reduce to frame size only
        self.coords_i_frame = np.zeros_like(coords_i_mat, dtype=int)

        # Reset points outside of the image coordinates
        x[x < 0] = 0
        x[x >= self.camera.img_width] = 0

        y[y < 0] = 0
        y[y >= self.camera.img_height] = 0

        self.coords_i_frame[:, :, 0] = x
        self.coords_i_frame[:, :, 1] = y

    '''
    coords camera vers image camera
    '''
    def generate_ground_image(self, Pi):
        #assert Pi.shape == (self.camera.img_height, self.camera.img_width)
        interpolate = True
        for j in range(self.target_height):
            for i in range(self.target_width):
                x, y = self.coords_i_mat[j, i]
                if 0 <= x < self.camera.img_width and 0 <= y < self.camera.img_height:

                    # interpolation
                    if interpolate:
                        ratio_y = y - np.floor(y)
                        x, y = int(x), int(y)
                        current_color = int(Pi[y, x])
                        below_color = int(Pi[y+1, x])
                        interpolated_color = current_color + (below_color - current_color) * ratio_y
                        self.Pg[j,i] = interpolated_color
                    else:
                        self.Pg[j,i] = Pi[y,x]
        return self.Pg

    def generate_ground_image_fast(self, Pi):
        x = self.coords_i_frame[:, :, 0]
        y = self.coords_i_frame[:, :, 1]

        self.Pg = Pi[y, x]
        return self.Pg

    def generate_camera_image(self, ground_view, image_origine):
        x = self.coords_i_frame[:, :, 0]
        y = self.coords_i_frame[:, :, 1]

        Pi = np.zeros((image_origine.shape[0], image_origine.shape[1]))
        Pi[y, x] = ground_view
        return Pi

    def generate_camera_image_lines(self, lines_list, image_origine):
        Pi = np.zeros((image_origine.shape[0], image_origine.shape[1], 3))
        for line in lines_list:
            valid_line = line[(line[:, 0] < self.coords_i_mat.shape[0]) &
                              (line[:, 0] >= 0) &
                              (line[:, 1] < self.coords_i_mat.shape[1]) &
                              (line[:, 1] >= 0)]
            line_coords_camera = self.coords_i_mat[valid_line[:, 0], valid_line[:, 1]].astype(int)
            cv2.polylines(Pi, [line_coords_camera], 0, (0, 0, 255))

        return Pi

    def generate_camera_image_lanes(self, lanes_list, y, image_origine):
        Pi = np.zeros((image_origine.shape[0], image_origine.shape[1], 3), np.uint8)
        import colorsys
        h_colors = [
            64,  # jaune
            116, # green
            240, # bleu
            300, # violet
            180, # cyan
        ]
        for i, lane in enumerate(reversed(lanes_list)):
            line_left_x, line_right_x = lane.skeleton - lane.width/2, lane.skeleton + lane.width/2

            # reconstruct line with y information
            line_left = np.array(list(zip(y, line_left_x)), dtype=int)
            line_right = np.array(list(zip(y, line_right_x)), dtype=int)

            valid_left_line = line_left[(line_left[:, 0] < self.coords_i_mat.shape[0]) &
                                       (line_left[:, 0] >= 0) &
                                       (line_left[:, 1] < self.coords_i_mat.shape[1]) &
                                       (line_left[:, 1] >= 0)]

            valid_right_line = line_right[(line_right[:, 0] < self.coords_i_mat.shape[0]) &
                                          (line_right[:, 0] >= 0) &
                                          (line_right[:, 1] < self.coords_i_mat.shape[1]) &
                                          (line_right[:, 1] >= 0)]

            line_left_coords_camera = self.coords_i_mat[valid_left_line[:, 0], valid_left_line[:, 1]].astype(int)
            line_right_coords_camera = self.coords_i_mat[valid_right_line[:, 0], valid_right_line[:, 1]].astype(int)
            line_right_reversed = np.array(list(reversed(line_right_coords_camera)))
            pts = np.vstack((line_left_coords_camera, line_right_reversed))

            # Colors
            lane_proba = np.mean(lane.probas.get_values())
            #light = np.interp(lane_proba, [0, 1], [1.0, 0.5])
            light = 0.5
            h = h_colors[i % len(h_colors)]/360.0
            rgb = np.array(colorsys.hls_to_rgb(h, light, 1.0))
            [r, g, b] = (rgb * 255).astype(int)

            cv2.fillPoly(Pi, [pts], (b, g, r))

        return Pi


'''
    B. Filterning & Thresholding
'''

def filter_image(pg):
    ### filtre horizontal ###

    filter_size_x = g.ipm.filter_x['filter_size_x'] # 25 pixels
    sx = float(g.ipm.filter_x['sx']) / 10.0
    x = np.arange(-filter_size_x / 2, filter_size_x / 2 + 1)
    kernel_x = [(1 / sx ** 2) * np.exp(-x ** 2 / (2 * sx ** 2)) * (1 - x ** 2 / (sx ** 2)) for x in x]
    try:
        pg_fu = np.apply_along_axis(lambda m: np.convolve(m, kernel_x), axis=1, arr=pg)
    except ValueError as e:
        pg_fu = np.zeros_like(pg)

    display_debug_image(img=pg_fu, frame_name='3.1. Horizontal Filter', position=(1, 0),
                        param_dict=g.ipm.filter_x)

    # Filtre vertical #
    filter_size_y = g.ipm.filter_y['filter_size_y'] # pixels
    sy = float(g.ipm.filter_y['sy']) / 10.0
    y = np.arange(-filter_size_y / 2, filter_size_y / 2 + 1)
    kernel_y = [np.exp(-y ** 2 / (2 * sy ** 2)) for y in y]
    pg_fv = np.apply_along_axis(lambda m: np.convolve(m, kernel_y), axis=0, arr=pg)
    # Normalize
    pg_fv = pg_fv/np.max(pg_fv)
    display_debug_image(img=pg_fv, frame_name='3.2. Vertical Filter', position=(1, 2),
                        param_dict=g.ipm.filter_y)

    # Combinaison des 2 filtres #
    pg_filtered = np.apply_along_axis(lambda m: np.convolve(m, kernel_y), axis=0, arr=pg_fu)

    # Crop and zero border
    pg_filtered = pg_filtered[filter_size_y/2:-filter_size_y/2, filter_size_x/2:-filter_size_x/2]

    border = 5
    pg_filtered[:border,:] = pg_filtered[-border:,:] = pg_filtered[:,:border] = pg_filtered[:,-border:] = 0

    return pg_filtered


def threshold_image(pg_filtered, q = 97.5):
    percentile = np.percentile(pg_filtered, q)
    #print("percentile:", percentile)

    pg_threshold = pg_filtered
    pg_threshold[pg_threshold < percentile] = 0

    return pg_threshold


'''
    C. Line Detection
'''
def compute_hough_lines(image):
    height = image.shape[0]
    #y_offset = height/3*2 # only look bottom part of image
    hough_top_point_ratio = 0.5
    y_offset = int(height * hough_top_point_ratio)
    h, theta, d = hough_line(image[y_offset:,:])

    points_list = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d,
                                                min_distance=g.ipm.hough_parameters['min_distance'],
                                                min_angle=g.ipm.hough_parameters['min_angle'])):

        if np.abs(angle) <= np.deg2rad(g.ipm.hough_parameters['max_angle']):
            y0 = int((dist - 0 * np.cos(angle)) / np.sin(angle))
            y1 = int((dist - image.shape[1] * np.cos(angle)) / np.sin(angle))
            points_list.append([(0, y0 + y_offset), (image.shape[1], y1 + y_offset)])


    return points_list

def compute_probabilistic_hough_lines(image):
    lines = probabilistic_hough_line(image,
                                     threshold=g.ipm.hough_p_parameters['threshold'],
                                     line_length=g.ipm.hough_p_parameters['line_length'],
                                     line_gap=g.ipm.hough_p_parameters['line_gap'])
    return lines


'''
    D. RANSAC
'''

def debug_ransac_poly(image, roi_image, x, y):
    from sklearn.metrics import mean_squared_error

    lines_list = []
    x_plot = np.linspace(0, image.shape[0]).astype(float)
    colors = ['red', 'green', 'blue']
    plt.plot(x[:, 0], y, 'b+')
    for polynomial_degree in [1, 2, 3]:
        model = make_pipeline(PolynomialFeatures(polynomial_degree),
                              RANSACRegressor(residual_threshold=np.mean(np.abs(y - np.mean(y)))))
        try:
            model.fit(x, y)
            mse = mean_squared_error(model.predict(x), y)

            y_plot = model.predict(x_plot[:, np.newaxis])

            plt.plot(x_plot, y_plot, color=colors[polynomial_degree - 1],
                     linewidth=3, label='polynomial_degree %d: error = %.3f' % (polynomial_degree, mse))

            lines_list.append(zip(y_plot.ravel().astype(int), x_plot.astype(int)))
        except ValueError as err:
            ransac = model.steps[1][1]
            lines_list.append([])

    legend_title = 'Error of Mean\nAbsolute Deviation'
    plt.legend(loc='upper right', frameon=False, title=legend_title,
               prop=dict(size='x-small'))

    roi_image[roi_image > 0] = 255
    roi_image_color = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)

    for polynomial_degree, line in enumerate(lines_list):
        if colors[polynomial_degree] == 'red':
            c = (0, 0, 255)
        elif colors[polynomial_degree] == 'green':
            c = (0, 255, 0)
        else:
            c = (255, 0, 0)
        for p1, p2 in list(zip(line, line[1:])):
            cv2.line(roi_image_color, p1, p2, c, 1)
    roi_image_color[roi_image > 0] = [255, 255, 255]
    display_debug_image(roi_image_color, frame_name='7. RANSAC', position=(5, 3))

    plt.show()

def ransac_fit(image, points_list):
    contours_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours_image[image > 0] = [255, 255, 255]

    # Create contour around each line found by hough method.
    # These lines are represented by 2 points at x=0 and x=image_width
    contour_points_list = []
    for p1, p2 in points_list:
        a = float(p2[1] - p1[1]) / (p2[0] - p1[0])
        b = image.shape[0]

        # Find points at y=image_height * 1/3 and y=image_height
        ransac_top_point_ratio = 0.5
        ptop = [int((b/2 - p1[1]) / a), int(b * ransac_top_point_ratio)]
        pbottom = [int((b - p1[1]) / a), b]

        top_offset = np.array([5, 0])  # px offset in X direction
        bottom_offset = np.array([2, 0])  # px offset in X direction

        contour_points = np.array(
            [pbottom - bottom_offset, ptop - top_offset, ptop + top_offset, pbottom + bottom_offset])
        contour_points_list.append(contour_points)

        c = np.random.randint(0, 126,3) + 127
        cv2.polylines(contours_image, [contour_points], isClosed=False, color=c, thickness=1)

    display_debug_image(contours_image, frame_name='7. RANSAC contours', position=(5, 0))

    # Extract ROI inside the contours found, and try to fit a degree2 line using RANSAC method on the
    # points found
    lines_list = []
    lines_fitted = []
    for contour_points in contour_points_list:
        # ROI : around the current line
        roi_mask = np.zeros_like(image)

        cv2.fillConvexPoly(roi_mask, contour_points, 1)
        roi_mask = roi_mask.astype(bool)

        # Select part of the image in the ROI
        roi_image = np.zeros_like(image)
        roi_image[roi_mask] = image[roi_mask]

        #roi_image = roi_image.astype(float)
        #display_debug_image(roi_image, frame_name='7. RANSAC', position=(5, 3))

        x, y = np.where(roi_image > 0)
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))

        test_poly = False
        if test_poly:
            debug_ransac_poly(image, roi_image, x, y)
        else:
            # seuil par defaut avec des median ne fonctionne pas car il peut
            # sortir des 0 !
            residual_threshold = np.mean(np.abs(y - np.mean(y)))
            model = make_pipeline(PolynomialFeatures(2),
                                  RANSACRegressor(residual_threshold=residual_threshold))
            try:
                model.fit(x, y)
                x_ransac = np.linspace(0, image.shape[0], num=image.shape[0] * 2)
                y_ransac = model.predict(x_ransac[:, np.newaxis])
                lines_fitted.append(True)
            except ValueError:
                # Ransac failed, Fallback to hough line values
                x_ransac = x.ravel()
                y_ransac = y
                lines_fitted.append(False)
            lines_list.append(np.vstack((x_ransac, y_ransac.ravel())).astype(int).T)

    # Display the ransac results
    image[image > 0] = 255
    image_color = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for fitted, line in zip(lines_fitted, lines_list):
        c = (0, 0, 255) if fitted else (255, 0, 0)
        cv2.polylines(image_color, [np.fliplr(line)], False, c, 1)

    display_debug_image(image_color, frame_name='7. RANSAC', position=(5, 3))

    return image_color, lines_list

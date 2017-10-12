import numpy as np
from util import *
import global_vars as g

def get_shadow_mask(l, s, light_threshold, sat_threshold):
    """
    Retrieve a mask corresponding to the shadows of the image.

    :param l: light channel
    :param s: saturation channel
    :param light_threshold: threshold to use to filter the light channel
    :param sat_threshold: threshold to use to filter the saturation channel
    :return: mask
    """

    # Get theshold parameters depending on if weare working in automatic mode or not
    if g.shadows.light_threshold['automatic'] > 0:
        l_min, l_max = 0, light_threshold
    else:
        l_min = g.shadows.light_threshold['min']
        l_max = g.shadows.light_threshold['max']

    if g.shadows.sat_threshold['automatic'] > 0:
        s_min, s_max = sat_threshold, 255
    else:
        s_min = g.shadows.sat_threshold['min']
        s_max = g.shadows.sat_threshold['max']

    # Threshold on light channel
    l_filtered = cv2.inRange(l, l_min, l_max)

    # Threshold on saturation channel
    s_filtered = cv2.inRange(s, s_min, s_max)

    display_debug_image(img=l_filtered, frame_name='l_filtered', position=(0, 0),
                        param_dict=g.shadows.light_threshold)
    display_debug_image(img=s_filtered, frame_name='s_filtered', position=(0, 1),
                        param_dict=g.shadows.sat_threshold)

    # Combine L & S masks
    mask = cv2.bitwise_and(l_filtered, s_filtered)
    return mask

def refine_mask(l, mask):
    """
    Given a mask in parameter and the light channel, try to refine the mask to get
    a more accurate result.

    :param l: light channel
    :param mask: original mask to refine
    :return: new mask
    """

    # Dilate the mask
    kernel = np.ones((g.shadows.mask_refine_parameters['kernel_dilate_y'],
                      g.shadows.mask_refine_parameters['kernel_dilate_x']), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # apply the mask on light channel, and refine pixels in this area
    roi = l[mask != 0]

    # threshold on the result to refine the original mask
    if g.shadows.mask_refine_parameters['roi_max'] == 0:
        roi_max = int(np.mean(roi) + np.std(roi)*1.5)
    else:
        roi_max = g.shadows.mask_refine_parameters['roi_max']

    mask = cv2.inRange(l, 0, roi_max)

    display_debug_image(img=mask, frame_name='Mask_refined', position=(0, 2),
                        param_dict=g.shadows.mask_refine_parameters)

    return mask

def create_background(img, mask, mode):
    """
    Background propagation: remove the part of the input image where there are shadows
    and fill the hole by propagating surrounding colors.

    :param img: input image
    :param mask: mask localizing shadows
    :param mode:
        - 0 : light mode, erosion to remove black values and propagate lighter values
        - 1 : saturation mode, dilatation to remove white values and propagate darker values
    :return: the original image with colors propagated at the location pointed by the mask
    """

    str_mode = 'light' if mode == 0 else 'sat'

    # Median blur to remove the white lines
    median_blur_ksize = g.shadows.background['median_blur_ksize'] * 2 + 1
    img_blured = cv2.medianBlur(img, median_blur_ksize)
    display_debug_image(img=img_blured, frame_name='Median_blur_' + str_mode, position=(2, mode*2))

    # Fill the shadows: dilate/erode the image depending on if we work on light
    # or saturation, in order to remove dark/white zones
    op = cv2.MORPH_DILATE if mode == 0 else cv2.MORPH_ERODE

    morphology_auto = g.shadows.background['morphology_auto'] > 0

    if morphology_auto:
        # automatic mode: find morphology kernel size from the image properties
        # mask ratio: proportion of white pixels in the mask
        mask_ratio = np.sum(mask / 255) / np.size(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, int(300 * mask_ratio)))
    else:
        kernel_x = g.shadows.background['morphology_kernel_x']
        kernel_y = g.shadows.background['morphology_kernel_y']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
    background = cv2.morphologyEx(img_blured, op, kernel)

    background_blured = cv2.blur(background, (11, 11))
    display_debug_image(img=background_blured, frame_name='Background_' + str_mode, position=(3, mode*2))

    img_background = img.copy()
    img_background[mask != 0] = background_blured[mask != 0]
    display_debug_image(img=img_background, frame_name='Foreground_background_' + str_mode, position=(4, mode*2))

    return img_background

''' Mask transparency (alpha channel) & Vignette '''
def generate_vignette(rows, cols):
    """
    Generate a vignette filter.

    :param rows: number of rows of the image
    :param cols: number of columns of the image
    :return: a gaussian filter centered in X and at 80% Y
    """
    kernel_x = cv2.getGaussianKernel(cols, cols / 6)
    kernel_y = cv2.getGaussianKernel(int(rows * 1.8), rows / 3)
    kernel_y = kernel_y[0:rows]
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.max(kernel)
    return mask

def get_alpha_from_mask(mask):
    """
    Generate a transparency mask, to focus on the center/lower part of the image.

    :param mask: non transparent input mask
    :return: a transparent mask
    """
    kernel_x = g.shadows.mask_blur['kernel_blur_x']
    kernel_y = g.shadows.mask_blur['kernel_blur_y']
    alpha_coef = g.shadows.mask_blur['alpha_coef']/255.0

    if kernel_x > 0 and kernel_y > 0:
        alpha = cv2.blur(mask, (kernel_y, kernel_x)) / 255.0
    else:
        alpha = mask/255.0

    # Transparency
    vignette = generate_vignette(*alpha.shape)
    alpha = alpha * alpha_coef * vignette

    display_debug_image(img=alpha, frame_name='Alpha', param_dict=g.shadows.mask_blur, position=(5,1))

    return alpha

''' Fusion '''
def fusion(img, img_shadow_fixed, alpha):
    """
    Fusion the original image with the fixed image (no shadows), using a transparency mask
    :param img: original image (light channel or saturation channel)
    :param img_shadow_fixed: image without shadows
    :param alpha: transparency mask to use for the fusion
    :return: fusion of the images
    """
    shadow_part = cv2.multiply(img_shadow_fixed / 255.0, alpha)
    light_part = cv2.multiply(img / 255.0, 1.0 - alpha)

    fusion = cv2.add(shadow_part, light_part)
    fusion = ((fusion / np.max(fusion))*255).astype(np.uint8)
    return fusion

def remove_shadow(image):
    """
    Remove shadows of an image.

    :param image: image containing shadows
    :return: image without shadows
    """
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        original = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        display_debug_image(original, 'original', position=(0, 4))

    # RGB to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)

    # Convert L and S channels to uint8
    if h.dtype != 'uint8':
        l = (l/l.max() * 255).astype(np.uint8)
        s = (s/s.max() * 255).astype(np.uint8)

    light_threshold = int(l.mean())
    sat_threshold = int(np.median(s))

    # No shadows: exit
    if light_threshold > 100:
        return image

    # Threshold the light & saturation channels to get a shadow mask
    mask = get_shadow_mask(l, s, light_threshold, sat_threshold)
    mask = refine_mask(l, mask)

    display_debug_trackers('Background_propatation', param_dict = g.shadows.background, position=(1, 1))
    l_background = create_background(l, mask, mode=0)
    s_background = create_background(s, mask, mode=1)

    # Blur the mask to soften sharp borders
    alpha = get_alpha_from_mask(mask)

    # Fusion the fixed light with the original light, using alpha channel
    l_fusion = fusion(l, l_background, alpha)
    s_fusion = fusion(s, s_background, alpha)

    # Convert the 2 channels back to original float32
    if h.dtype != 'uint8':
        l_fusion = l_fusion.astype(np.float32)/255.0
        s_fusion = s_fusion.astype(np.float32)/255.0

    # Merge to final image
    hls_fixed = cv2.merge((h, l_fusion, s_fusion))
    rgb = cv2.cvtColor(hls_fixed, cv2.COLOR_HLS2RGB)

    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        bgr = cv2.cvtColor(hls_fixed, cv2.COLOR_HLS2BGR)
        display_debug_image(bgr, 'Final', position=(5, 2))

    return rgb

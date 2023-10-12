import numpy as np
import cv2
from skimage.feature import hog
from skimage import measure, morphology

def preprocess_image(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(imgray, 90, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = np.array(255-im_bw, dtype=bool)
    cleaned = morphology.remove_small_objects(im_bw, min_size=20, connectivity=2)
    return np.array(cleaned, dtype=int)

def extract_regions(label_image):
    regions = measure.regionprops(label_image)
    return regions

def compute_hog_features(region, k):
    digit = region.image
    padding = np.zeros([digit.shape[0], digit.shape[1] + k], dtype='float64')
    padding[:, k//2:padding.shape[1] - k//2] = digit
    re_digit = cv2.resize(padding, (28, 28), interpolation=cv2.INTER_AREA)
    re_digit = cv2.dilate(re_digit, (3, 3))
    roi_hog_fd = hog(re_digit, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    return np.array([roi_hog_fd], 'float64')

def labeled_user_image(image, k=0):
    cleaned = preprocess_image(image)
    label, n = measure.label(cleaned, neighbors=8, background=255, return_num=True, connectivity=2)
    print("numbers of numbers on image:", n)
    
    regions = extract_regions(label)
    numbers, ph, rect = [], [], []
    for region in regions:
        numbers.append(compute_hog_features(region, k))
        ph.append(region.image)
        minr, minc, maxr, maxc = region.bbox
        rect.append([(minc, minr), (maxc - minc), (maxr - minr)])
    
    return numbers, ph, rect

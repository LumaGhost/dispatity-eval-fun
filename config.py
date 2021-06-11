import cv2 as cv
import numpy as np


'''
folder should contain entries from the middlebury 2014 dataset
i.e. a collection of folders with the following contents:
    img0.png (the left camera image)
    img1.png, imgE1.png, imgL1.png (right camera image in different lighting conditions)
    disp0.pfm (floating point ground truth horizontal disparity relative to the left image)
    note: currently for simplicity only disp0 is supported
'''
ALL_DATASETS = "./datasets/middlebury/2014/"

'''
lighting: "default", "E", or "L" to specifiy which image1 version to load from the dataset
'''
LIGHTING = "default"

'''
bad_threshold: disparity difference from the ground truth where pixels will be
    considered "bad" for the purpose of calculating the % of "bad pixels"
'''
BAD_THRESHOLD = 1.5

def calc_dispariry(im1, im2):
    '''
    this function should expect two matricies representing the left and right image
    and return a matrix representing the disparity map calculated from the two images
    note: invalid pixels should be expressed as inf/nan. all pixels that are not inf
    or nan will be considered valid disparities for the purpose of average and other calculations
    '''
    stereo = cv.StereoSGBM_create(numDisparities=300, blockSize=8)
    disp = stereo.compute(im1,im2).astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.inf
    return disp
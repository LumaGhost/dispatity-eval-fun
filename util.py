import sys
import re
from struct import *
import numpy as np
import cv2 as cv

def load_pfm(path):
    '''
    copy pasted from 
    https://gist.github.com/chpatrick/8935738 (:
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    '''
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(path,"rb") as file:
        header = file.readline().decode('utf8').rstrip()
        if header == 'PF':
            color = True    
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.flipud(np.reshape(data, shape))


def displayable_pfm(data):
    '''
    https://gist.github.com/chpatrick/8935738#gistcomment-2765922
    normalize the floating point values of a pfm so that its displayable
    as a grayscale imagen with opencv
    '''
    max_val_pct=0.1
    data = np.where(data == np.inf, -1, data)
    max_val = np.max(data)
    max_val += max_val * max_val_pct
    data = np.where(data == -1, max_val, data)
    normalized = cv.normalize(data, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
    return cv.resize(normalized, dsize=(0, 0), fx=0.2, fy=0.2)
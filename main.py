import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from collections import namedtuple

import sys
import re
from struct import *
import re
import sys

## https://gist.github.com/chpatrick/8935738
'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(path):
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

# shamlessly https://stackoverflow.com/a/48840964
def read_pfm_old(path):
    debug = True
    with open(path,"rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type=f.readline().decode('latin-1')
        if "PF" in type:
            channels=3
        elif "Pf" in type:
            channels=1
        else:
            print("ERROR: Not a valid PFM file",file=sys.stderr)
            sys.exit(1)
        if(debug):
            print("DEBUG: channels={0}".format(channels))

        # Line 2: width height
        line=f.readline().decode('latin-1')
        width,height=re.findall('\d+',line)
        width=int(width)
        height=int(height)
        if(debug):
            print("DEBUG: width={0}, height={1}".format(width,height))

        # Line 3: +ve number means big endian, negative means little endian
        line=f.readline().decode('latin-1')
        BigEndian=True
        if "-" in line:
            BigEndian=False
        if(debug):
            print("DEBUG: BigEndian={0}".format(BigEndian))

        # Slurp all binary data
        samples = width*height*channels;
        buffer  = f.read(samples*4)

        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt=">"
        else:
            fmt="<"
        fmt= fmt + str(samples) + "f"
        img = unpack(fmt,buffer)
    print(width*height)
    print(len(img))
    arr = np.array(img, dtype=np.float32, ndmin=2, order='F')
    arr = np.reshape(arr, (height, width))
    print(len(arr))
    print(arr.shape)
    return cv.rotate(arr, cv.ROTATE_180)

# https://gist.github.com/chpatrick/8935738#gistcomment-2765922
def displayable_pfm(data):
    max_val_pct=0.1
    data = np.where(data == np.inf, -1, data)
    max_val = np.max(data)
    max_val += max_val * max_val_pct
    data = np.where(data == -1, max_val, data)
    normalized = cv.normalize(data, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
    return cv.resize(normalized, dsize=(0, 0), fx=0.2, fy=0.2)

ALL_DATASETS = "./datasets/middlebury/2014/"

BenchConfig = namedtuple('BenchConfig', ['lighting','bad_threshold'])

# metrics and record keeping during benchmarking
BenchResults = namedtuple('BenchResults', ['percent_bad',
                                           'percent_invalid',
                                           'avg_diff'])

def img1_name(lighting):
    if lighting == "" or lighting == "default":
        return "im1.png"
    elif lighting.upper() == "E":
        return "im1E.png"
    elif lighting.upper() == "L":
        return "im1L.png"
    else:
        raise ValueError("unexpected lighting option: {}".format(lighting))

'''
just the avg diff between ground truth disparity at each pixel 
(average out of all the valid pixels i.e. not black/inf).

the percent of invalid (inf/black/no disparity) pixels out of total pixels

and the percent of pixels where the ground truth disp difference is greater than some threshold 
(again out of total pixels, and these are called "bad pixels"). 

they also have the percent of pixels that are invalid or bad. i.e. bad+invalid /count
'''
def process_folder(*, path, disparity_func, config):
    im0 = cv.imread(os.path.join(path, "im0.png"))
    im1 = cv.imread(os.path.join(path, img1_name(config.lighting)))
    # IMREAD_UNCHANGED gives the same result
    # true_disp = cv.imread(os.path.join(path, "disp0.pfm"), cv.IMREAD_ANYDEPTH)
    true_disp = load_pfm(os.path.join(path, "disp0.pfm"))
    # cv.imshow("true_disp", displayable_pfm(true_disp))
    # cv.imshow("im0", im0)
    # cv.imshow("im1", im1)
    # cv.waitKey(0)

    disp = disparity_func(im0, im1)

    assert disp.shape == true_disp.shape

    disp_diff = np.absolute(np.subtract(disp, true_disp))
    
    bad_pixels = np.count_nonzero(disp_diff < config.bad_threshold)
    inf_pixels = np.count_nonzero(np.isinf(disp))
    nan_pixels = np.count_nonzero(np.isnan(disp))
    invalid_pixels = inf_pixels + nan_pixels
    
    percent_bad = 100*(bad_pixels / disp.size)
    percent_invalid = 100*(invalid_pixels / disp.size)
    avg_diff = np.average(disp_diff[np.logical_and(~np.isnan(disp_diff),
                                                    ~np.isinf(disp_diff))])

    # print("precent bad: {}".format(percent_bad))
    # print("percent invalid: {}".format(percent_invalid))
    # print("avg diff: {}".format(avg_diff))
    
    # cv.imshow("disp", displayable_pfm(disp))
    # cv.waitKey(0)

    return BenchResults(percent_bad=percent_bad,
                        percent_invalid=percent_invalid,
                        avg_diff=avg_diff)

def iterate_files(*, disparity_func, config):
    all_folders = os.listdir(ALL_DATASETS)
    entry_count = len(all_folders)
    percent_bad = []
    percent_invalid = []
    avg_diff = []
    for entry in all_folders:
        full_path = os.path.join(ALL_DATASETS, entry)
        print("reading {}".format(full_path))
        if '.' in entry:
            print("skipping {}".format(full_path))
            continue
        results = process_folder(path=full_path, 
                                disparity_func=disparity_func,
                                config=config)
        percent_bad.append(results.percent_bad)
        percent_invalid.append(results.percent_invalid)
        avg_diff.append(results.avg_diff)
    return BenchResults(percent_bad=np.average(percent_bad),
                        percent_invalid=np.average(percent_invalid),
                        avg_diff=np.average(avg_diff))


def calc_dispariry(im1, im2):
    stereo = cv.StereoSGBM_create(numDisparities=300, blockSize=8)
    disp = stereo.compute(im1,im2).astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.inf
    return disp

if __name__ == "__main__":
    config = BenchConfig(lighting="default", bad_threshold=1.5)
    results = iterate_files(disparity_func=calc_dispariry, config=config)
    print(results)

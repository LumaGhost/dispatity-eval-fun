import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from collections import namedtuple
import util
import config


'''
grouping the config parameters supllied by the user
'''
BenchConfig = namedtuple('BenchConfig', ['lighting','bad_threshold'])

'''
metrics returned by the benchmark
percent bad: percent of pixels with a disparity difference larger than
    the "bad threshold" specified in the config
percent invalid: percent of pixels with nan/inf disparity
avg diff: average disparity difference from the ground truth
'''
BenchResults = namedtuple('BenchResults', ['percent_bad',
                                           'percent_invalid',
                                           'avg_diff'])


'''
maps the lighting config paramter to the image1 name that should be loaded
'''
def img1_name(lighting):
    if lighting == "" or lighting == "default":
        return "im1.png"
    elif lighting.upper() == "E":
        return "im1E.png"
    elif lighting.upper() == "L":
        return "im1L.png"
    else:
        raise ValueError("unexpected lighting option: {}".format(lighting))


def process_folder(*, path, disparity_func, conf):
    '''
    - calculate dispairity between im0 and the im1 version specified by the config
    using the input disparity function
    - note: assertion failure if the calculated disparty and ground truth disparity
    are not the same shape
    - calculate the average element wise absolute difference from the ground truth (excluding nan and inf)
    - calculate the percent of pixels with a disparity difference below the "bad threshold"
    - calculate the percent of "invalid" disparity results i.e. nan or inf
    '''
    im0 = cv.imread(os.path.join(path, "im0.png"))
    im1 = cv.imread(os.path.join(path, img1_name(conf.lighting)))
    true_disp = util.load_pfm(os.path.join(path, "disp0.pfm"))

    disp = disparity_func(im0, im1)

    assert disp.shape == true_disp.shape

    disp_diff = np.absolute(np.subtract(disp, true_disp))
    
    bad_pixels = np.count_nonzero(disp_diff < conf.bad_threshold)
    inf_pixels = np.count_nonzero(np.isinf(disp))
    nan_pixels = np.count_nonzero(np.isnan(disp))
    invalid_pixels = inf_pixels + nan_pixels
    
    percent_bad = 100*(bad_pixels / disp.size)
    percent_invalid = 100*(invalid_pixels / disp.size)
    avg_diff = np.average(disp_diff[np.logical_and(~np.isnan(disp_diff),
                                                    ~np.isinf(disp_diff))])

    return BenchResults(percent_bad=percent_bad,
                        percent_invalid=percent_invalid,
                        avg_diff=avg_diff)


def run_benchmark(*, disparity_func, conf):
    '''
    iterate though any folders in config.ALL_DATASETS
    load im0 and either im1 im1E or im1L depending on the input config
    calculate the disparity map using the input disparity func
    return BenchResults averaged over all folders
    '''
    all_results = []
    for root, dirs, _ in os.walk(config.ALL_DATASETS):
        for name in dirs:
            full_path = os.path.join(root, name)
            print("reading {}".format(full_path))
            results = process_folder(path=full_path, 
                                    disparity_func=disparity_func,
                                    conf=conf)
            all_results.append(results)
    percent_bad, percent_invalid, avg_diff = list(zip(*all_results))
    return BenchResults(percent_bad=np.average(percent_bad),
                        percent_invalid=np.average(percent_invalid),
                        avg_diff=np.average(avg_diff))


if __name__ == "__main__":
    results = run_benchmark(disparity_func=config.calc_dispariry, 
                            conf=BenchConfig(lighting=config.LIGHTING,
                                             bad_threshold=config.BAD_THRESHOLD))
    print(results)

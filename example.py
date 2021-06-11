import numpy as np
import cv2 as cv
import argparse
import disp_eval

NUM_DISPARITIES = 304
BLOCK_SIZE = 9
def sgbm(im1, im2):
    stereo = cv.StereoSGBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
    disp = stereo.compute(im1,im2).astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.inf
    return disp

def to_8uc1(im):
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)

def bm(im1, im2):
    stereo = cv.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
    disp = stereo.compute(to_8uc1(im1),to_8uc1(im2)).astype(float) / 16.0
    disp[disp <= 0.0] = np.inf
    return disp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Configuration (:')
    parser.add_argument('--lighting', default="default", type=str,
                        help='see config.py')
    parser.add_argument('--all_datasets', default="./datasets/middlebury/2014/", type=str,
                        help='see config.py')
    parser.add_argument('--bad_threshold', default=1.5, type=float,
                        help='see config.py')
    args = parser.parse_args()
    
    bench_conf= disp_eval.BenchConfig(lighting=args.lighting,
                                      bad_threshold=args.bad_threshold,
                                      all_datasets=args.all_datasets)
    print("config: {}".format(bench_conf))
    sgbm_results = disp_eval.run_benchmark(disparity_func=sgbm, 
                                          conf=bench_conf)
    print("sgbm: {}".format(sgbm_results))
    bm_results = disp_eval.run_benchmark(disparity_func=bm, 
                                          conf=bench_conf)
    print("bm: {}".format(bm_results))



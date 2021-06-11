
### Intro

hello (: this is a project i made for fun and nostalgia and to demonstrate my python capabilities.
this projects supports benchmarking disparity algorithms against the middlebury 2014 dataset.
the user supplies the disparity algorithm they want to test and a configuration for the
benchmark and this project provdes functionality for running the benchmark and obtaining results.

### Usage

install:
- `git clone https://github.com/LumaGhost/dispatity-eval-fun.git`
- `cd disparty-eval-fun`
- `pip install -r requirements.txt`

examples:
- `python example.py -h`

to use as a library:
- import disp_eval

to use as a script:
- open `config.py` and complete the configuration
- `python disp_eval.py`


### Examples

this example will benchmark the opencv block matching algorithm against any datasets stored in `./datasets/`
and display the results

```

import numpy as np
import cv2 as cv
import disp_eval


def bm(im1, im2):
    stereo = cv.StereoBM_create()
    disp = stereo.compute(cv.cvtColor(im1, cv.COLOR_BGR2GRAY),
                          cv.cvtColor(im2, cv.COLOR_BGR2GRAY)).astype(float) / 16.0
    disp[disp <= 0.0] = np.inf
    return disp

if __name__ == "__main__":
    
    bench_conf= disp_eval.BenchConfig(lighting="default",
                                      bad_threshold=1.5,
                                      all_datasets="./datasets/")
    bm_results = disp_eval.run_benchmark(disparity_func=bm, 
                                          conf=bench_conf)
    print("bm: {}".format(bm_results))

```

a more involved example is provided in `example.py`

### Questions

why the middlebury dataset? only because i've worked with it before. i also think a lot of the images are really nice to look at (: 

### Contributing

i am not planning on accepting contributions. this is just something i did for fun (:
feedback is fine though. i am still pretty new to python (:
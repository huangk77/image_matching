#ifndef __EPSIF__
#define __EPSIF__
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#include <valarray>
#include <sys/time.h>

#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static const float CONST_ITR_SCALE = 0.50;
static const int CONST_FILTER_SIZE = 9;
static const int CONST_FILTER_SIZE_HALF = 4;
using namespace std;
using namespace cv;


// void ep2d_simple_gray(float *output, float *input, const int M, const int N, float threshold, const int iter_max);
void EPSIF(const cv::Mat &src, cv::Mat &dst, float threshold = 0.45, const int iter_max = 3);
// void ep2d_simple_color(float *output, float *input, float threshold = 0.45, const int iter_max = 3);

#endif
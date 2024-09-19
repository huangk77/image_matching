#ifndef __FCOMMON__
#define __FCOMMON__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <thread>
#include <vector>
#include <valarray>
#include <sys/time.h>
#include <thread>

#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"

const cv::Mat gaussian_W_5 = (cv::Mat_<float>(5, 5) << 0.016531580643701, 0.029701870689091, 0.036108291846035, 0.029701870689091, 0.016531580643701,
                              0.029701870689091, 0.053364596008407, 0.064874850041854, 0.053364596008407, 0.029701870689091,
                              0.036108291846035, 0.064874850041854, 0.078867760327278, 0.064874850041854, 0.036108291846035,
                              0.029701870689091, 0.053364596008407, 0.0648748500418541, 0.0533645960084072, 0.0297018706890914,
                              0.0165315806437010, 0.0297018706890914, 0.0361082918460354, 0.0297018706890914, 0.0165315806437010);

const cv::Mat kernel_prewitt_x = (cv::Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
const cv::Mat kernel_prewitt_y = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);

const cv::Mat kernel_sobel_h1 = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
const cv::Mat kernel_sobel_h2 = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

// typedef struct
// {
//     std::vector<cv::Mat> Scalespace;
//     std::vector<cv::Mat> E_Scalespace;
//     std::vector<cv::Mat> Max_Scalespace;
//     std::vector<cv::Mat> Min_Scalespace;
// } WSSF_Feat_InParams;

typedef struct
{
  cv::Mat descriptor;
  cv::Mat locs;
} descriptor;

typedef struct
{
  float x;
  float y;
  float layers;
  float attr_1; // 关键点属性
  float attr_2; // 关键点属性 预留
  float attr_3; // 关键点属性 预留
} WSSFKeyPts;

// typedef struct
// {
//     std::vector<cv::Mat> Bolb_space;
//     std::vector<cv::Mat> Corner_space;
//     std::vector<cv::Mat> Bolb_gradient_cell;
//     std::vector<cv::Mat> Corner_gradient_cell;
//     std::vector<cv::Mat> Bolb_angle_cell;
//     std::vector<cv::Mat> Corner_angle_cell;
// } WSSF_Feat_OutParams;

double what_time_is_it_now();
void meshgrid(const cv::Range &xRange, const cv::Range &yRange, cv::Mat &x_grid, cv::Mat &y_grid, float norm_cols = 1, float norm_rows = 1);
void steerable_gaussians2(cv::InputArray src, cv::OutputArray dst, int filter = 5, int sigmas = 5);
void applyATAN2ToMat(cv::Mat &src1, cv::Mat &src2, cv::Mat &dst);

#endif
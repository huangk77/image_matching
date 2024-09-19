#ifndef __CREATE_IMAGE_SPACE__
#define __CREATE_IMAGE_SPACE__

#include "Fcommon.hpp"
#include "phasecong3.hpp"
#include "ep2d_simple_9.hpp"

void Create_Image_space(const cv::Mat &im,
                        std::vector<cv::Mat> &Nonelinear_Scalespace,
                        std::vector<cv::Mat> &E_Scalespace,
                        std::vector<cv::Mat> &Max_Scalespace,
                        std::vector<cv::Mat> &Min_Scalespace,
                        std::vector<cv::Mat> &Phase_Scalespace,
                        bool Scale_Invariance = true,
                        int nOctaves = 3,
                        float ScaleValue = 2,
                        float ratio = 1.2599,
                        float sigma_1 = 1.6,
                        float filter = 5);

#endif
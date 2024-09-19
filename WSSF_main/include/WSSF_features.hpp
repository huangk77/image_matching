#ifndef __WSSF_features__
#define __WSSF_features__

#include "Fcommon.hpp"
using namespace std;
using namespace cv;

void WSSF_features(const std::vector<cv::Mat> &nonelinear_space,
                   const std::vector<cv::Mat> &E_space,
                   const std::vector<cv::Mat> &Max_space,
                   const std::vector<cv::Mat> &Min_space,
                   const std::vector<cv::Mat> &Phase_space,
                   // output
                   std::vector<WSSFKeyPts> &position_1,
                   std::vector<WSSFKeyPts> &position_2,
                   std::vector<cv::Mat> &Bolb_gradient_cell,
                   std::vector<cv::Mat> &Corner_gradient_cell,
                   std::vector<cv::Mat> &Bolb_angle_cell,
                   std::vector<cv::Mat> &Corner_angle_cell,
                   bool Scale_Invariance = false,
                   int nOctaves = 3,
                   float sigma_1 = 1.6,
                   float ratio = 1.2599
                   );

#endif
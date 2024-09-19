#ifndef __GLOH_DESC__
#define __GLOH_DESC__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "Fcommon.hpp"

void GLOH_descriptors(const std::vector<cv::Mat> &gradient, const std::vector<cv::Mat> &angle, const std::vector<WSSFKeyPts> &key_point_vector, descriptor &des_example, const int Path_Block = 48, float ratio = 1.2599, float sigma_1 = 1.6);
/*cv::Mat &descriptors_des,std::vector<WSSFKeyPts>cv::Mat &descriptors_locs*/

void GLOH_descriptors_multi(const std::vector<cv::Mat> &gradient, const std::vector<cv::Mat> &angle, const std::vector<WSSFKeyPts> &key_point_vector, descriptor &des_example, const int Path_Block = 48, float ratio = 1.2599, float sigma_1 = 1.6);

#endif
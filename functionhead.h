#ifndef _FUNCTIONHEAD_H
#define _FUNCTIONHEAD_H



// Create_Image_space
int Create_Image_space(
const cv::Mat& im,
int nOctaves,
int Scale_Invariance,
double ScaleValue, 
double ratio,
double sigma_1,
double filter,
std::vector<cv::Mat>& Nonelinear_Scalespace,
std::vector<cv::Mat>& E_Scalespace,
std::vector<cv::Mat>& Max_Scalespace,
std::vector<cv::Mat>& Min_Scalespace,
std::vector<cv::Mat>& Phase_Scalespace
);

//WSSF_features
int WSSF_features(
const std::vector<cv::Mat>& nonelinear_space,
const std::vector<cv::Mat>& E_space,
const std::vector<cv::Mat>& Max_space,
const std::vector<cv::Mat>& Min_space,
const std::vector<cv::Mat>& Phase_space,
double sigma_1,
double ratio,
int Scale_Invariance,
int nOctaves,
cv::Mat& position_1,
cv::Mat& position_2,
std::vector<cv::Mat>& Bolb_gradient_cell,
std::vector<cv::Mat>& Corner_gradient_cell,
std::vector<cv::Mat>& Bolb_angle_cell,
std::vector<cv::Mat>& Corner_angle_cell
);


//GLOH_descriptors
int GLOH_descriptors(
const std::vector<cv::Mat>& gradient,
const std::vector<cv::Mat>& angle,
const cv::Mat& key_point_array,
const cv::Mat& Path_Block,
double ratio,
double sigma_1,
std::vector<cv::Mat>& descriptors_des,
std::vector<cv::Mat>& descriptors_locs
);




//WSSF_gradient_feature
int WSSF_gradient_feature(
const std::vector<cv::Mat>& Scalespace,
const std::vector<cv::Mat>& E_Scalespace,
const std::vector<cv::Mat>& Max_Scalespace,
const std::vector<cv::Mat>& Min_Scalespace,
const std::vector<cv::Mat>& Phase_Scalespace,
int Scale_Invariance,
int nOctaves,
std::vector<cv::Mat>& Bolb_space,
std::vector<cv::Mat>& Corner_space,
std::vector<cv::Mat>& Bolb_gradient_cell,
std::vector<cv::Mat>& Corner_gradient_cell,
std::vector<cv::Mat>& Bolb_angle_cell,
std::vector<cv::Mat>& Corner_angle_cell
);




//FeatureDetection
int FeatureDetection(
const std::vector<cv::Mat>& Bolb_space,
const std::vector<cv::Mat>& Corner_space,
int layers,
int npt1,
int npt2, 
double sigma_1,
double ratio,
cv::Mat& Blob_key_point_array,
cv::Mat& Corner_key_point_array
);




//WSSF_selectMax_NMS
int WSSF_selectMax_NMS(
const cv::Mat& AFPts,
int window,
cv::Mat& keypoints
);




//kptsOrientation
int kptsOrientation(
const cv::Mat& key,
const std::vector<cv::Mat>& gradient,
const std::vector<cv::Mat>& angle,
const std::vector<cv::Mat>& nonelinear_space,
double sigma_1,
double ratio,
cv::Mat& key_point_array
);

#endif
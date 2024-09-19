#include "Fcommon.hpp"
#include "phasecong3.hpp"
#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include "ep2d_simple_9.hpp"
#include "WSSF_features.hpp"
#include "Create_Image_space.hpp"
#include "GLOH_description.hpp"

using namespace cv;
using namespace cv::ximgproc;

int LogLevel;
int main()
{
    // cv::Mat img1 = cv::imread("../images/bird_512.png", IMREAD_GRAYSCALE);

    initLogger(DEBUG);

    cv::Mat img1 = imread("../images/bird_512.png", IMREAD_GRAYSCALE);
    cv::Mat img2 = imread("../images/bird_512.png", IMREAD_GRAYSCALE);

    cv::resize(img1, img1, Size(), 0.8, 0.8);
    cv::resize(img2, img2, Size(), 0.8, 0.8);

    img1.convertTo(img1, CV_32FC1);
    img2.convertTo(img2, CV_32FC1);
    cv::normalize(img1, img1, 1.0, 0, cv::NORM_MINMAX);
    cv::normalize(img2, img2, 1.0, 0, cv::NORM_MINMAX);

    std::vector<cv::Mat> Nonelinear_Scalespace_1, Nonelinear_Scalespace_2;
    std::vector<cv::Mat> E_Scalespace_1, E_Scalespace_2;
    std::vector<cv::Mat> Max_Scalespace_1, Max_Scalespace_2;
    std::vector<cv::Mat> Min_Scalespace_1, Min_Scalespace_2;
    std::vector<cv::Mat> Phase_Scalespace_1, Phase_Scalespace_2;

    std::vector<WSSFKeyPts> Bolb_KeyPts_1, Bolb_KeyPts_2;
    std::vector<WSSFKeyPts> Corner_KeyPts_1, Corner_KeyPts_2;
    std::vector<cv::Mat> Bolb_gradient_1, Bolb_gradient_2;
    std::vector<cv::Mat> Corner_gradient_1, Corner_gradient_2;
    std::vector<cv::Mat> Bolb_angle_1, Bolb_angle_2;
    std::vector<cv::Mat> Corner_angle_1, Corner_angle_2;

    descriptor Bolb_descriptors_1, Corner_descriptors_1;
    descriptor Bolb_descriptors_2, Corner_descriptors_2;

    double t1, t2;
    t1 = what_time_is_it_now();

    Create_Image_space(img1, Nonelinear_Scalespace_1, E_Scalespace_1, Max_Scalespace_1, Min_Scalespace_1, Phase_Scalespace_1, true);
    Create_Image_space(img2, Nonelinear_Scalespace_2, E_Scalespace_2, Max_Scalespace_2, Min_Scalespace_2, Phase_Scalespace_2, true);

    t2 = what_time_is_it_now();

    LOG(NOTICE, "构造影像尺度空间花费时间 = %.3f ms", t2 - t1);
    // for (int i = 0; i < Nonelinear_Scalespace_1.size(); i++)
    // {
    //     LOG(DEBUG, "size = [%d, %d]", Nonelinear_Scalespace_1.at(i).rows, Nonelinear_Scalespace_1.at(i).cols);
    // }

    t1 = what_time_is_it_now();

    WSSF_features(Nonelinear_Scalespace_1, E_Scalespace_1, Max_Scalespace_1, Min_Scalespace_1, Phase_Scalespace_1, Bolb_KeyPts_1, Corner_KeyPts_1, Bolb_gradient_1, Corner_gradient_1, Bolb_angle_1, Corner_angle_1, true);
    WSSF_features(Nonelinear_Scalespace_2, E_Scalespace_2, Max_Scalespace_2, Min_Scalespace_2, Phase_Scalespace_2, Bolb_KeyPts_2, Corner_KeyPts_2, Bolb_gradient_2, Corner_gradient_2, Bolb_angle_2, Corner_angle_2, true);
    
    t2 = what_time_is_it_now();

    LOG(NOTICE, "特征点提取花费时间 = %.3f ms", t2 - t1);
    LOG(NOTICE, "特征点个数 = %d", Bolb_KeyPts_1.size());

    t1 = what_time_is_it_now();

    GLOH_descriptors(Bolb_gradient_1, Bolb_angle_1, Bolb_KeyPts_1, Bolb_descriptors_1);
    GLOH_descriptors(Corner_gradient_1, Corner_angle_1, Corner_KeyPts_1, Corner_descriptors_1);

    GLOH_descriptors(Bolb_gradient_2, Bolb_angle_2, Bolb_KeyPts_2, Bolb_descriptors_2);
    GLOH_descriptors(Corner_gradient_2, Corner_angle_2, Corner_KeyPts_2, Corner_descriptors_2);

    t2 = what_time_is_it_now();
    LOG(NOTICE, "特征描述子花费时间 = %.3f ms\n", t2 - t1);

    // LOG(NOTICE, "Bolb_descriptors_1 size = [%d, %d]", Bolb_descriptors_1.descriptor.rows, Bolb_descriptors_1.descriptor.cols);
    // LOG(NOTICE, "Corner_descriptors_1 size = [%d, %d]", Corner_descriptors_1.descriptor.rows, Corner_descriptors_1.descriptor.cols);
    // LOG(NOTICE, "Bolb_descriptors_2 size = [%d, %d]", Bolb_descriptors_2.descriptor.rows, Bolb_descriptors_2.descriptor.cols);
    // LOG(NOTICE, "Corner_descriptors_2 size = [%d, %d]", Corner_descriptors_2.descriptor.rows, Corner_descriptors_2.descriptor.cols);

    cv::BFMatcher bfmatches;
    std::vector<cv::DMatch> Blob_matches_point;
    std::vector<cv::DMatch> Corner_matches_point;

    t1 = what_time_is_it_now();
    bfmatches.match(Bolb_descriptors_1.descriptor, Bolb_descriptors_2.descriptor, Blob_matches_point);
    bfmatches.match(Corner_descriptors_1.descriptor, Corner_descriptors_2.descriptor, Corner_matches_point);

    std::vector<int> queryIdxs(Blob_matches_point.size()), trainIdxs(Blob_matches_point.size());
    for (size_t i = 0; i < Blob_matches_point.size(); i++)
    {
        queryIdxs[i] = Blob_matches_point[i].queryIdx; // 取出查询图片中匹配的点对的索引即id号；那么queryIdxs、trainIdxs都为257
        trainIdxs[i] = Blob_matches_point[i].trainIdx; // 取出训练图片中匹配的点对的索引即id号；
    }

    t2 = what_time_is_it_now();
    LOG(NOTICE, "暴力匹配所花时间 = %.3f ms\n", t2 - t1);

    LOG(NOTICE, "Blob_matches_point count = %d", Blob_matches_point.size());
    LOG(NOTICE, "Corner_matches_point count = %d", Corner_matches_point.size());

    // // 绘制匹配线
    // Mat resultImg;
    // drawMatches(img1, keypoints_obj, img2, keypoints_scene, matches, resultImg,
    // 			Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return 0;
}
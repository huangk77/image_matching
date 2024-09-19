#include "WSSF_features.hpp"

static void get_WSSF_gradient_feature(
    // input
    const std::vector<cv::Mat> &Scalespace_ls,
    const std::vector<cv::Mat> &E_Scalespace_ls,
    const std::vector<cv::Mat> &Max_Scalespace_ls,
    int layers,
    // output
    std::vector<cv::Mat> &Bolb_space,
    std::vector<cv::Mat> &Corner_space,
    std::vector<cv::Mat> &Bolb_gradient_cell,
    std::vector<cv::Mat> &Corner_gradient_cell,
    std::vector<cv::Mat> &Bolb_angle_cell,
    std::vector<cv::Mat> &Corner_angle_cell)
{
    // int M = Scalespace_ls.at(0).rows;
    // int N = Scalespace_ls.at(0).cols;

    for (int j = 0; j < layers; j++)
    {
        cv::Mat Max_Scalespace = Max_Scalespace_ls.at(j).clone();
        cv::Mat E_Scalespace = E_Scalespace_ls.at(j).clone();
        cv::Mat Scalespace = Scalespace_ls.at(j).clone();

        int M = Scalespace.rows;
        int N = Scalespace.cols;

        cv::Mat Cornerspace;
        cv::Mat Bolbspace = Max_Scalespace + 2 * E_Scalespace;

        steerable_gaussians2(Bolbspace, Bolbspace, 5, 5);

        Bolb_space.push_back(Max_Scalespace.clone());

        cv::normalize(Scalespace, Cornerspace, 1.0, 0);
        steerable_gaussians2(Cornerspace, Cornerspace, 5, 5);

        Corner_space.push_back(Max_Scalespace.clone());

        // Bolb
        cv::Mat gradient_x_Bolb_1, gradient_y_Bolb_1, gradient_Bolb_1;
        cv::Mat Bolb_angle = cv::Mat::zeros(M, N, CV_32FC1);

        cv::Mat Bolb_angle_mask;

        cv::filter2D(Bolbspace, gradient_x_Bolb_1, Bolbspace.depth(), kernel_sobel_h1);
        cv::filter2D(Bolbspace, gradient_y_Bolb_1, Bolbspace.depth(), kernel_sobel_h2);
        cv::add(gradient_x_Bolb_1.mul(gradient_x_Bolb_1), gradient_y_Bolb_1.mul(gradient_y_Bolb_1), gradient_Bolb_1);
        cv::sqrt(gradient_Bolb_1, gradient_Bolb_1);

        Bolb_gradient_cell.push_back(gradient_Bolb_1.clone());

        applyATAN2ToMat(gradient_y_Bolb_1, gradient_x_Bolb_1, Bolb_angle);
        Bolb_angle = Bolb_angle * 180.0 / M_PI;
        Bolb_angle_mask = (Bolb_angle < 0);
        cv::add(Bolb_angle, cv::Mat(Bolb_angle.size(), Bolb_angle.type(), cv::Scalar(360.0)), Bolb_angle, Bolb_angle_mask);

        Bolb_angle_cell.push_back(Bolb_angle.clone());

        // Corner
        cv::Mat gradient_x_Corner_1, gradient_y_Corner_1, gradient_Corner_1;
        cv::Mat gradient_x_Corner_2, gradient_y_Corner_2, gradient_Corner_2;
        cv::Mat Corner_angle = cv::Mat::zeros(M, N, CV_32FC1);
        cv::Mat Corner_angle_mask;

        cv::filter2D(Cornerspace, gradient_x_Corner_1, Cornerspace.depth(), kernel_sobel_h1);
        cv::filter2D(Cornerspace, gradient_y_Corner_1, Cornerspace.depth(), kernel_sobel_h2);
        cv::add(gradient_x_Corner_1.mul(gradient_x_Corner_1), gradient_y_Corner_1.mul(gradient_y_Corner_1), gradient_Corner_1);

        cv::filter2D(gradient_Corner_1, gradient_x_Corner_2, gradient_Corner_1.depth(), kernel_sobel_h1);
        cv::filter2D(gradient_Corner_1, gradient_y_Corner_2, gradient_Corner_1.depth(), kernel_sobel_h2);
        cv::add(gradient_x_Corner_2.mul(gradient_x_Corner_2), gradient_y_Corner_2.mul(gradient_y_Corner_2), gradient_Corner_2);

        Corner_gradient_cell.push_back(gradient_Corner_2.clone());

        applyATAN2ToMat(gradient_y_Bolb_1, gradient_x_Bolb_1, Corner_angle);
        Corner_angle = Corner_angle * 180 / M_PI;
        Corner_angle_mask = (Corner_angle < 0);
        cv::add(Corner_angle, cv::Mat(Corner_angle.size(), Corner_angle.type(), cv::Scalar(360.0)), Corner_angle, Corner_angle_mask);

        Corner_angle_cell.push_back(Corner_angle.clone());
    }
}

static void detectKAZEFeatures(cv::Mat &images, std::vector<cv::KeyPoint> &feat_kpts)
{
    // 检测特征点(非线性)
    std::vector<cv::KeyPoint> temp_point;
    cv::Ptr<cv::AKAZE> detector = AKAZE::create();
    detector->setThreshold(0.0001);
    detector->setNOctaveLayers(4);
    detector->setNOctaves(3);
    detector->setMaxPoints(5000);

    // std::vector<cv::KeyPoint> keypoints;
    detector->detect(images, temp_point);

    // 使用unordered_map存储坐标点和对应最大response的KeyPoint
    std::unordered_map<std::string, cv::KeyPoint> unique_keypoints_map;

    for (auto &kp : temp_point)
    {
        kp.pt.x = floor(kp.pt.x);
        kp.pt.y = floor(kp.pt.y);
        std::string key = std::to_string((int)(kp.pt.x)) + "_" + std::to_string((int)(kp.pt.y));

        // 如果map中没有这个坐标点，或者当前关键点的response更大，更新map
        if (unique_keypoints_map.find(key) == unique_keypoints_map.end() || kp.response > unique_keypoints_map[key].response)
        {
            unique_keypoints_map[key] = kp;
        }
    }

    // 将map中的结果转换回vector
    for (const auto &pair : unique_keypoints_map)
    {
        feat_kpts.push_back(pair.second);
    }
}

static void WSSF_selectMax_NMS(const std::vector<cv::KeyPoint> &feat_kpts, std::vector<cv::KeyPoint> &feat_kpts_nms, float window)
{
    // cout << feat_kpts_nms.size() << endl;
    // assert(feat_kpts_nms.size() == 0);
    float r = window / 2.0;
    int Numbers = feat_kpts.size();
    if (window != 0)
    {
        for (int i = 0; i < Numbers; i++)
        {
            for (int j = i + 1; j < Numbers; j++)
            {
                float distance = std::sqrt(std::pow(feat_kpts.at(i).pt.x - feat_kpts.at(j).pt.x, 2) + std::pow(feat_kpts.at(i).pt.y - feat_kpts.at(j).pt.y, 2));
                if (distance <= r * r)
                {
                    if (feat_kpts.at(i).response < feat_kpts.at(j).response)
                    {
                        feat_kpts_nms.push_back(feat_kpts.at(j));
                    }
                    else
                    {
                        feat_kpts_nms.push_back(feat_kpts.at(i));
                    }
                }
            }
        }
    }
}

static Point zhixin(const Mat &sub_nonelinear_space)
{
    Moments m = moments(sub_nonelinear_space, true);

    // Calculate the x and y coordinates of the centroid
    double x_zhixin = m.m10 / m.m00;
    double y_zhixin = m.m01 / m.m00;

    // Return the centroid as a Point object
    return Point(x_zhixin, y_zhixin);
}

// Function to calculate orientation
static double calculate_orientation_zhixin(int x, int y, int radius, const Mat &nonelinear_space)
{
    int radius_x_left = x - radius;
    int radius_x_right = x + radius;
    int radius_y_up = y - radius;
    int radius_y_down = y + radius;

    // Define region of interest (ROI) boundaries
    if (radius_x_left <= 0)
    {
        radius_x_left = 1;
    }

    if (radius_y_up <= 0)
    {
        radius_y_up = 1;
    }

    if (radius_x_right >= nonelinear_space.cols)
    {
        radius_x_right = nonelinear_space.cols - 1;
        radius_x_left = radius_x_left - 1;
    }

    if (radius_y_down >= nonelinear_space.rows)
    {
        radius_y_down = nonelinear_space.rows - 1;
        radius_y_up = radius_y_up - 1;
    }

    // Extract submatrix (ROI - Region of Interest)
    Rect roi(radius_x_left, radius_y_up, radius_x_right - radius_x_left + 1, radius_y_down - radius_y_up + 1);
    Mat sub_nonlinear_space = nonelinear_space(roi);

    // Calculate the central point within the submatrix
    Point center_zhixin = zhixin(sub_nonlinear_space);
    double angle = atan2(center_zhixin.y, center_zhixin.x);

    // Check for NaN angle
    if (isnan(angle))
    {
        angle = 0;
    }
    else
    {
        angle = angle * 180 / CV_PI; // Convert radians to degrees
    }

    // Optional: Convert the angle to a bin (if needed)
    /*
    int n = 12;
    int bin = round(angle * n / 360);
    if (bin >= n) bin -= n;
    if (bin < 0) bin += n;
    angle = bin;
    */

    return angle;
}

static void calculate_oritation_hist(Mat &hist, double &max_value, int x, int y, int radius, const Mat &gradient,
                                     const Mat &angle, const Mat &nonelinear_space, int n, const Mat &Sa, double sigma_1, double ratio, int layer)
{
    // Calculate sigma for Gaussian weight
    double sigma = 1.5 * sigma_1 * std::pow(ratio, layer - 1);

    // Determine ROI bounds and ensure they are within image limits
    int radius_x_left = x - radius;
    int radius_x_right = x + radius;
    int radius_y_up = y - radius;
    int radius_y_down = y + radius;
    
    if (radius_x_left <= 0)
    {
        radius_x_left = 1;
    }

    if (radius_y_up <= 0)
    {
        radius_y_up = 1;
    }

    if (radius_x_right >= gradient.cols)
    {
        radius_x_right = gradient.cols - 1;
        radius_x_left = radius_x_left - 1;
    }

    if (radius_y_down >= gradient.rows)
    {
        radius_y_down = gradient.rows - 1;
        radius_y_up = radius_y_up - 1;
    }

    // Extract submatrices for gradient and angle
    cv::Rect roi(radius_x_left, radius_y_up, radius_x_right - radius_x_left + 1, radius_y_down - radius_y_up + 1);
    Mat sub_gradient = gradient(roi);
    Mat sub_angle = angle(roi);

    // Create meshgrid for X and Y
    cv::Mat XX, YY;
    cv::Range x_range = cv::Range(-(x - radius_x_left), radius_x_right - x + 1);
    cv::Range y_range = cv::Range(-(y - radius_y_up), radius_y_down - y + 1);
    meshgrid(x_range, y_range, XX, YY, 1, 1);

    // LOG(DEBUG, "meshgrid_x = %d,%d meshgrid_y = %d,%d", XX.rows, XX.cols, YY.rows, YY.cols);

    // Compute Gaussian weights
    Mat gaussian_weight;

    cv::exp(-(XX.mul(XX) + YY.mul(YY)) / (2 * sigma * sigma), gaussian_weight);
    // cout << "gaussian_weight size =  " << gaussian_weight.size() << endl;
    // cout << "sub_gradient size =  " << sub_gradient.size() << endl;
    cv::Mat W1 = sub_gradient.mul(gaussian_weight);
    // cout << "Sa size =  " << Sa.size() << endl;
    cv::Mat W = Sa.mul(W1);

    // cout << "W size =  " << W.size() << endl;
    // multiply(Sa(roi), W1, W);

    // cout << "Binning angles" << endl;
    // Binning angles
    Mat bin;
    Mat srcMatrix = sub_angle * n / 360.0;
    srcMatrix.convertTo(bin, CV_8UC1, 1.0, 0.0);
    bin.convertTo(bin, CV_32FC1, 1.0, 0.0);

    // bin set
    cv::Mat mask = (bin >= n);
    cv::add(bin, cv::Mat(bin.size(), bin.type(), cv::Scalar(-n)), bin, mask);
    mask = (bin < 0);
    cv::add(bin, cv::Mat(bin.size(), bin.type(), cv::Scalar(n)), bin, mask);

    // Compute histogram
    Mat temp_hist = Mat::zeros(1, n, CV_32FC1);
    // LOG(DEBUG, "bin size = [%d, %d]", bin.rows, bin.cols);
    // LOG(DEBUG, "W size = [%d, %d]", W.rows, W.cols);
    for (int i = 0; i < n; ++i)
    {
        // Create a mask where bin == i+1 (because MATLAB is 1-based index)
        Mat mask = (bin == (i + 1));
        if (!mask.empty())
        {
            // Sum the elements in W where the mask is true
            mask.convertTo(mask, W.type(), 1.0 / 255.0, 0);
            // cout << mask << endl;

            cv::Scalar sum_val = cv::sum(W.mul(mask));
            // cout << sum_val << endl;
            // Store the sum in the corresponding bin of temp_hist
            temp_hist.at<float>(0, i) = sum_val[0]; // sum_val[0] contains the sum because W is a single-channel matrix
        }
    }
    
    // Smooth histogram
    // Mat hist = Mat::zeros(1, n, CV_32FC1);
    hist.at<float>(0) = (temp_hist.at<float>(0, n - 2) + temp_hist.at<float>(0, 2)) / 16.0 +
                        4 * (temp_hist.at<float>(0, n - 1) + temp_hist.at<float>(0, 1)) / 16.0 +
                        temp_hist.at<float>(0, 0) * 6 / 16.0;

    // Smooth the second bin
    hist.at<float>(1) = (temp_hist.at<float>(0, n - 1) + temp_hist.at<float>(0, 3)) / 16.0 +
                        4 * (temp_hist.at<float>(0, 0) + temp_hist.at<float>(0, 2)) / 16.0 +
                        temp_hist.at<float>(0, 1) * 6 / 16.0;

    // Smooth the middle bins
    for (int i = 2; i < n - 2; ++i)
    {
        hist.at<float>(0, i) = (temp_hist.at<float>(0, i - 2) + temp_hist.at<float>(0, i + 2)) / 16.0 +
                               4 * (temp_hist.at<float>(0, i - 1) + temp_hist.at<float>(0, i + 1)) / 16.0 +
                               temp_hist.at<float>(0, i) * 6 / 16.0;
    }

    // Smooth the second-to-last bin
    hist.at<float>(0, n - 2) = (temp_hist.at<float>(0, n - 4) + temp_hist.at<float>(0, 0)) / 16.0 +
                               4 * (temp_hist.at<float>(0, n - 3) + temp_hist.at<float>(0, n - 1)) / 16.0 +
                               temp_hist.at<float>(0, n - 2) * 6 / 16.0;

    // Smooth the last bin
    hist.at<float>(0, n - 1) = (temp_hist.at<float>(0, n - 3) + temp_hist.at<float>(0, 1)) / 16.0 +
                               4 * (temp_hist.at<float>(0, n - 2) + temp_hist.at<float>(0, 0)) / 16.0 +
                               temp_hist.at<float>(0, n - 1) * 6 / 16.0;
    // Find maximum value in the histogram
    cv::minMaxLoc(hist, nullptr, &max_value, nullptr, nullptr);
}

static void orientation(vector<double> &ANG, int x, int y,
                        const cv::Mat gradientImg,
                        const cv::Mat gradientAng,
                        const cv::Mat Scalespace,
                        double radius, int patch_size_zhixin, double sigma_1,
                        double ratio, int layer, int n, double ORI_PEAK_RATIO)
{
    if (Scalespace.channels() != 1)
    {
        cvtColor(Scalespace, Scalespace, COLOR_BGR2GRAY);
    }

    // 半径为r的圆
    int diameter = 2 * radius + 1;
    cv::Mat Sa = cv::Mat::zeros(diameter, diameter, CV_32FC1);
    cv::circle(Sa, cv::Point(radius, radius), radius, cv::Scalar(1.0), -1);

    // cv::imwrite("Sa.tiff", Sa);
    // std::cout << "save Sa done!" << std::endl;

    // 统计梯度直方图，并给出最大方向上的值
    // Mat hist;
    Mat hist = Mat::zeros(1, n, CV_32FC1);
    double max_value;
    calculate_oritation_hist(hist, max_value, x, y, radius, gradientImg, gradientAng, Scalespace, n, Sa, sigma_1, ratio, layer);
    double mag_thr = max_value * ORI_PEAK_RATIO;
    double angle_zhixin = calculate_orientation_zhixin(x, y, radius, Scalespace);

    // 对关键点邻域的梯度大小进行高斯加权。每相邻三个bin采用高斯加权，根据Lowe的建议，模板采用[0.25,0.5,0.25]。方向包括主方向和辅助方向
    ANG[0] = angle_zhixin;
    int k1, k2;
    double bin;
    for (int k = 1; k < n; k++)
    {
        if (k == 1)
        {
            k1 = n;
        }
        else
            k1 = k - 1;

        if (k == n)
            k2 = 1;
        else
            k2 = k + 1;

        if ((hist.at<double>(k - 1) > hist.at<double>(k1 - 1)) && (hist.at<double>(k - 1) > hist.at<double>(k2 - 1)) && (hist.at<double>(k - 1) > mag_thr))
        {
            bin = k - 1 + 5 * (hist.at<double>(k1 - 1) - hist.at<double>(k2 - 1)) / (hist.at<double>(k1 - 1) + hist.at<double>(k2 - 1) - 2 * hist.at<double>(k - 1));
            if (bin < 0)
                bin = n + bin;
            else
            {
                if (bin >= n)
                    bin = bin - n;
            }
            double angle = ((360 / n) * bin); // 0-360
            ANG[1] = angle;
        }
    }
}

static void kptsOrientation(const vector<WSSFKeyPts> &key,
                            const vector<Mat> &gradient,
                            const vector<Mat> &gradientAng,
                            const vector<Mat> &nonelinear_space,
                            std::vector<WSSFKeyPts> &key_point_array,
                            float sigma_1,
                            float ratio)
{
    int HIST_BIN = 36;
    double SIFT_ORI_PEAK_RATIO = 0.95;
    int key_number = 0;

    for (int i = 0; i < key.size(); i++)
    {
        WSSFKeyPts key_temp;
        key_temp = key.at(i);

        float x = key_temp.x;
        float y = key_temp.y;
        float layer = key_temp.layers;
        int diameter = (int)(16 * layer);
        float radius_zhixin = 8 * layer;

        if ((diameter % 2) != 0)
        {
            diameter++;
        }

        float x1 = fmax(0, x - floor((float)(diameter) / 2.0));
        float y1 = fmax(0, y - floor((float)(diameter) / 2.0));
        float x2 = fmin(x + floor((float)(diameter) / 2.0), gradient.at(layer - 1).cols);
        float y2 = fmin(y + floor((float)(diameter) / 2.0), gradient.at(layer - 1).rows);

        if (((int)(y2 - y1) != diameter) || ((int)(x2 - x1) != diameter))
        {
            continue;
        }
        else
        {
            std::vector<double> angle(2);
            // cout << "begin orientation" << endl;
            orientation(angle, x, y, gradient.at(layer - 1), gradientAng.at(layer - 1), nonelinear_space.at(layer - 1), (float)(diameter) / 2.0, radius_zhixin, sigma_1, ratio, layer, HIST_BIN, SIFT_ORI_PEAK_RATIO);
            // cout << "end orientation" << endl;
            for (int j = 1; j < angle.size(); j++)
            {
                // std::cout << "here" << std::endl;
                WSSFKeyPts kpts;
                key_number = key_number + 1;
                kpts.x = x;
                kpts.y = y;
                kpts.layers = layer;
                kpts.attr_1 = angle.at(j - 1);
                key_point_array.push_back(kpts);
                // key_point_array[key_number - 1].x = x;
                // key_point_array[key_number - 1].y = y;
                // key_point_array[key_number - 1].layers = layer;
                // key_point_array[key_number - 1].attr_1 = angle[j - 1];
            }
        }
    }
}

static void FeatureDetection(const std::vector<cv::Mat> &Bolb_space,
                             const std::vector<cv::Mat> &Corner_space,
                             std::vector<WSSFKeyPts> &Blob_key_point_array,
                             std::vector<WSSFKeyPts> &Corner_key_point_array,
                             int layers,
                             float sigma_1,
                             float ratio,
                             int npt1 = 5000,
                             int npt2 = 5000)
{
    int Blob_key_number = 1;
    int Corner_key_number = 1;

    double minValue, maxValue; // 最大值，最小值
    cv::Point minIdx, maxIdx;  // 最小值坐标，最大值坐标

    std::vector<cv::KeyPoint> Bolb_kpts_cv, Bolb_kpts_cv_nms, Corner_kpts_cv, Corner_kpts_cv_nms;
    // WSSFKeyPts Bolb_kpts, Corner_kpts;

    for (int l = 0; l < layers; l++)
    {
        cv::Mat im1 = Bolb_space.at(l).clone();
        cv::Mat im2 = Corner_space.at(l).clone();

        cv::minMaxLoc(im1, &minValue, &maxValue, NULL, NULL);
        im1 = (im1 - minValue) / (maxValue - minValue);
        detectKAZEFeatures(im1, Bolb_kpts_cv);

        WSSF_selectMax_NMS(Bolb_kpts_cv, Bolb_kpts_cv_nms, 5);

        cv::minMaxLoc(im2, &minValue, &maxValue, NULL, NULL);
        im2 = (im2 - minValue) / (maxValue - minValue);
        detectKAZEFeatures(im2, Corner_kpts_cv);

        WSSF_selectMax_NMS(Corner_kpts_cv, Corner_kpts_cv_nms, 5);

        for (const auto &item : Bolb_kpts_cv_nms)
        {
            // std::cout << item.size << std::endl;
            if (item.size < 5)
            {
                WSSFKeyPts Bolb_kpts;
                Bolb_kpts.x = item.pt.x;
                Bolb_kpts.y = item.pt.y;
                Bolb_kpts.layers = l + 1;
                Bolb_kpts.attr_1 = item.response;
                Blob_key_point_array.push_back(Bolb_kpts);
            }
        }

        for (const auto &item : Corner_kpts_cv_nms)
        {
            // std::cout << item.size << std::endl;
            if (item.size < 5)
            {
                WSSFKeyPts Corner_kpts;
                Corner_kpts.x = item.pt.x;
                Corner_kpts.y = item.pt.y;
                Corner_kpts.layers = l + 1;
                Corner_kpts.attr_1 = item.response;
                Corner_key_point_array.push_back(Corner_kpts);
            }
        }
    }
}

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
                   bool Scale_Invariance,
                   int nOctaves,
                   float sigma_1,
                   float ratio)
{
    std::vector<cv::Mat> Bolb_space;
    std::vector<cv::Mat> Corner_space;

    std::vector<WSSFKeyPts> Blob_key_point_array;
    std::vector<WSSFKeyPts> Corner_key_point_array;

    int _octaves = 0;

    if (Scale_Invariance == false)
    {
        _octaves = 1;
    }
    else
    {
        _octaves = nOctaves;
    }
    get_WSSF_gradient_feature(nonelinear_space, E_space, Max_space, _octaves,
                              Bolb_space, Corner_space, Bolb_gradient_cell, Corner_gradient_cell, Bolb_angle_cell, Corner_angle_cell);
    FeatureDetection(Bolb_space, Corner_space, Blob_key_point_array, Corner_key_point_array, _octaves, sigma_1, ratio);

    // std::cout << "Blob_key_point_array number = " << Blob_key_point_array.size() << std::endl;
    // std::cout << "Corner_key_point_array number = " << Corner_key_point_array.size() << std::endl;

    kptsOrientation(Blob_key_point_array, Bolb_gradient_cell, Bolb_angle_cell, nonelinear_space, position_1, sigma_1, ratio);
    kptsOrientation(Corner_key_point_array, Corner_gradient_cell, Corner_angle_cell, nonelinear_space, position_2, sigma_1, ratio);
}

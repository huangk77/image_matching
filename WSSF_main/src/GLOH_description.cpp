#include "GLOH_description.hpp"

cv::Mat calc_log_polar_descriptor(const cv::Mat &gradient, const cv::Mat &angle, int x, int y,
                                  float main_angle, int d, int n, float Path_Block, int circle_count)
{
    // Calculate cos and sin of main_angle
    float cos_t = cos(-main_angle * CV_PI / 180.0);
    float sin_t = sin(-main_angle * CV_PI / 180.0);

    int M = gradient.rows;
    int N = gradient.cols;
    int radius = (int)(round(Path_Block));

    int radius_x_left = x - radius;
    int radius_x_right = x + radius;
    int radius_y_up = y - radius;
    int radius_y_down = y + radius;
    // Define region of interest (ROI) boundaries
    if (radius_x_left <= 0)
    {
        radius_x_left = 1;
    }

    if (radius_x_right > N)
    {
        radius_x_right = N - 1;
    }

    if (radius_y_up <= 0)
    {
        radius_y_up = 1;
    }

    if (radius_y_down > M)
    {
        radius_y_down = M - 1;
    }
    // std::cout << M << " " << N << std::endl;
    // std::cout << radius_x_left << std::endl;
    // std::cout << radius_y_up << std::endl;
    // std::cout << radius_x_right - radius_x_left + 1 << std::endl;
    // std::cout << radius_y_down - radius_y_up + 1 << std::endl;
    // std::cout << "\n"
    //           << std::endl;

    // int radius_x_left = std::max(x - radius, 0);
    // int radius_x_right = std::min(x + radius, N - 1);
    // int radius_y_up = std::max(y - radius, 0);
    // int radius_y_down = std::min(y + radius, M - 1);

    // Extract submatrices (ROI)
    cv::Rect roi(radius_x_left, radius_y_up, radius_x_right - radius_x_left, radius_y_down - radius_y_up);
    cv::Mat sub_gradient = gradient(roi);
    // std::cout<<"gradient=" <<gradient <<endl;
    cv::Mat sub_angle = angle(roi);

    // Normalize angle
    // std::cout << sub_angle << std::endl;
    // std::cout << main_angle << std::endl;
    sub_angle = (sub_angle - main_angle) * n / 360;

    // sub_angle.setTo(sub_angle + n, sub_angle <= 0)
    for (int i = 0; i < sub_angle.rows; ++i)
    {
        for (int j = 0; j < sub_angle.cols; ++j)
        {
            sub_angle.at<float>(i, j) = std::round(sub_angle.at<float>(i, j));
            if (sub_angle.at<float>(i, j) <= 0)
            {
                sub_angle.at<float>(i, j) += n;
            }
        }
    }
    sub_angle.setTo(n, sub_angle == 0);
    // std::cout << sub_angle << std::endl;
    //  Calculate xrange
    cv::Range X = cv::Range(-(x - radius_x_left), (radius_x_right - x));
    cv::Range Y = cv::Range(-(y - radius_y_up), (radius_y_down - y));
    cv::Mat XX, YY;
    meshgrid(X, Y, XX, YY, 1.0, 1.0);
    // Create meshgrid for X and Y

    // Rotate coordinates
    cv::Mat c_rot = XX * cos_t - YY * sin_t;
    cv::Mat r_rot = XX * sin_t + YY * cos_t;
    cv::Mat log_amp;
    cv::Mat log_amplitude;
    cv::pow((c_rot.mul(c_rot) + r_rot.mul(r_rot)), 0.5, log_amp);
    cv::log(log_amp, log_amplitude);
    log_amplitude = log_amplitude / log(2.0);
    // Calculate log-polar coordinates
    cv::Mat log_angle = cv::Mat::zeros(r_rot.size(), r_rot.type());
    applyATAN2ToMat(r_rot, c_rot, log_angle);
    log_angle = log_angle / CV_PI * 180;
    // log_angle.setTo(log_angle + 360, log_angle < 0);
    for (int i = 0; i < log_angle.rows; ++i)
    {
        for (int j = 0; j < log_angle.cols; ++j)
        {
            if (log_angle.at<float>(i, j) < 0)
            {
                log_angle.at<float>(i, j) += 360;
            }
            // Normalize angle
            log_angle.at<float>(i, j) = std::round(log_angle.at<float>(i, j) * d / 360);
            // log_angle.setTo(log_angle + d, log_angle <= 0);
            // log_angle.setTo(log_angle - d, log_angle > d);
            if (log_angle.at<float>(i, j) <= 0)
            {
                log_angle.at<float>(i, j) += d;
            }
            else if (log_angle.at<float>(i, j) > d)
            {
                log_angle.at<float>(i, j) += (-1.0 * d);
            }
        }
    }
    // std::cout << " Calculate radius for amplitude bins" << std::endl;
    // Calculate radius for amplitude bins
    double double_max_radius;
    minMaxLoc(log_amplitude, nullptr, &double_max_radius, nullptr, nullptr);
    float max_radius = float(double_max_radius);
    cv::Mat temp_hist = cv::Mat::zeros(1, (circle_count * d + 1) * n, CV_32FC1);
    int row = log_angle.rows;
    int col = log_angle.cols;
    if (circle_count == 2)
    {
        float r1 = max_radius * 0.25 * 0.73;
        float r2 = max_radius * 0.73;

        log_amplitude.setTo(1, log_amplitude <= r1);
        log_amplitude.setTo(2, ((log_amplitude > r1) & (log_amplitude <= r2)));
        log_amplitude.setTo(3, log_amplitude > r2);
    }
    else
    {
        float r1 = log2(max_radius * 0.3006);
        float r2 = log2(max_radius * 0.7071);
        float r3 = log2(max_radius * 0.866);

        log_amplitude.setTo(1, log_amplitude <= r1);
        log_amplitude.setTo(2, (log_amplitude > r1) & (log_amplitude <= r2));
        log_amplitude.setTo(3, (log_amplitude > r2) & (log_amplitude <= r3));
        log_amplitude.setTo(4, log_amplitude > r3);
    }

    // Histogram calculation
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            int angle_bin = log_angle.at<float>(i, j);
            int amplitude_bin = log_amplitude.at<float>(i, j);
            int bin_vertical = sub_angle.at<float>(i, j);
            float Mag = sub_gradient.at<float>(i, j);

            if (amplitude_bin == 1)
            {
                temp_hist.at<float>(0, bin_vertical) = temp_hist.at<float>(0, bin_vertical) + Mag;
            }
            else
            {
                int index = ((amplitude_bin - 2) * d + angle_bin - 1) * n + bin_vertical - 1 + n;
                temp_hist.at<float>(0, index) += Mag;
            }
        }
    }

    // Normalize the histogram
    cv::Mat normalized_hist;

    // Step 1: Normalize the histogram
    float norm = cv::norm(temp_hist, cv::NORM_L2);
    if (norm != 0.0)
    {
        normalized_hist = temp_hist / norm;
    }
    else
    {
        normalized_hist = temp_hist.clone(); // Handle the case where norm is 0
    }
    // Step 2: Truncate values above 0.2
    threshold(normalized_hist, normalized_hist, 0.2, 0.2, cv::THRESH_TRUNC);
    // Step 3: Normalize the histogram again
    norm = cv::norm(normalized_hist, cv::NORM_L2);
    if (norm != 0.0)
    {
        normalized_hist = normalized_hist / norm;
    }
    cv::Mat descriptor = normalized_hist.t();

    // std::cout << descriptor.size() << std::endl;
    return descriptor;
}

void GLOH_descriptors(const std::vector<cv::Mat> &gradient, const std::vector<cv::Mat> &angle, const std::vector<WSSFKeyPts> &key_point_vector, descriptor &des_example, const int Path_Block, float ratio, float sigma_1) /*cv::Mat &descriptors_des,std::vector<WSSFKeyPts>cv::Mat &descriptors_locs*/
{
    int LOG_POLAR_WIDTH = 16;
    int LOG_POLAR__HIST_BINS = 12;
    int circle_count = 2;

    int M = key_point_vector.size();
    int d = LOG_POLAR_WIDTH;
    int n = LOG_POLAR__HIST_BINS;

    int desc_dimension = (d * circle_count + 1) * n; // 16 * 2 + 1 + 1 = 34
    cv::Mat WFSS_descriptors = cv::Mat::zeros(desc_dimension, M, CV_32FC1);
    // std::cout << WFSS_descriptors.size() << std::endl;
    cv::Mat locs = cv::Mat::zeros(4, M, CV_32FC1);

    // parrellel for loop
    for (int i = 0; i < M; i++)
    {
        float x = key_point_vector.at(i).x;
        float y = key_point_vector.at(i).y;
        int layer = key_point_vector.at(i).layers;
        float main_angle = key_point_vector.at(i).attr_1;

        locs.at<float>(0, i) = x;
        locs.at<float>(1, i) = y;
        locs.at<float>(2, i) = (float)layer;
        locs.at<float>(3, i) = main_angle;

        cv::Mat current_gradient = gradient.at(layer - 1);
        cv::Mat current_angle = angle.at(layer - 1);
        // std::cout << "begin " << i << std::endl;
        // WFSS_descriptors.col(i) = calc_log_polar_descriptor(current_gradient, current_angle, x, y, main_angle, d, n, Path_Block, circle_count);
        cv::Mat temp = calc_log_polar_descriptor(current_gradient, current_angle, x, y, main_angle, d, n, Path_Block, circle_count);
        // std::cout << "done" << std::endl;
    }
    des_example.descriptor = WFSS_descriptors;
    des_example.locs = locs;
}

void GLOH_descriptors_multi(const std::vector<cv::Mat> &gradient, const std::vector<cv::Mat> &angle, const std::vector<WSSFKeyPts> &key_point_vector, descriptor &des_example, const int Path_Block, float ratio, float sigma_1)
{
#if 0
    int LOG_POLAR_WIDTH = 16;
    int LOG_POLAR__HIST_BINS = 12;
    int circle_count = 2;

    int M = key_point_vector.size();
    int d = LOG_POLAR_WIDTH;
    int n = LOG_POLAR__HIST_BINS;

    int desc_dimension = (d * circle_count + 1) * n; // 16 * 2 + 1 + 1 = 34
    cv::Mat WFSS_descriptors = cv::Mat::zeros(desc_dimension, M, CV_32FC1);
    cv::Mat locs = cv::Mat::zeros(4, M, CV_32FC1);

    // int MaxThread = std::max(std::thread::hardware_concurrency(), 1u);
    int global_count = 0;
    int MaxThread = 8;
    std::vector<std::thread> worker;
    printf("MaxThread = %d\n", MaxThread);
    for (int ThreadNum = 0; ThreadNum < MaxThread; ThreadNum++)
    {
        worker.emplace_back([&](int id)
                            {
                                
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(id, &mask);
                if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
                    std::cerr << "set thread affinity failed" << std::endl;
                printf("Bind process on CPU %d\n", id);
                int begin_loop = M / MaxThread * id + std::min(M % MaxThread, id);
                int end_loop = M / MaxThread * (id+1) + std::min(M % MaxThread, id+1);
                printf("begin = %d, end = %d\n", begin_loop, end_loop);

                for (int i = begin_loop; i < end_loop; ++i) {
                    float x = key_point_vector.at(i).x;
                    float y = key_point_vector.at(i).y;
                    int layer = key_point_vector.at(i).layers;
                    float main_angle = key_point_vector.at(i).attr_1;

                    locs.at<float>(0, i) = x;
                    locs.at<float>(1, i) = y;
                    locs.at<float>(2, i) = (float)layer;
                    locs.at<float>(3, i) = main_angle;

                    cv::Mat current_gradient = gradient.at(layer - 1);
                    cv::Mat current_angle = angle.at(layer - 1);
                    // WFSS_descriptors.col(i) = cv::Mat(desc_dimension, 1, WFSS_descriptors.type(),cv::Scalar(id));
                    WFSS_descriptors.col(i) = calc_log_polar_descriptor(current_gradient, current_angle, x, y, main_angle, d, n, Path_Block, circle_count);
                    // printf("done!\n");
                    // printf("global_count = %d id = %d\n", global_count++, id);
                } }, ThreadNum);
    }
    for (auto &t : worker)
        t.join();

    des_example.descriptor = WFSS_descriptors;
    des_example.locs = locs;
#endif
}
#include "Fcommon.hpp"

double what_time_is_it_now()
{
    // 单位: ms
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        return 0;
    }
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * .001;
}

void meshgrid(const cv::Range &xRange, const cv::Range &yRange, cv::Mat &x_grid, cv::Mat &y_grid, float norm_cols, float norm_rows)
{
    cv::Mat xRow = cv::Mat(1, xRange.end - xRange.start, CV_32FC1);
    cv::Mat yCol = cv::Mat(yRange.end - yRange.start, 1, CV_32FC1);

    for (int i = 0; i < xRow.cols; i++)
    {
        xRow.at<float>(0, i) = static_cast<float>(xRange.start + i) / norm_cols;
    }

    for (int i = 0; i < yCol.rows; i++)
    {
        yCol.at<float>(i, 0) = static_cast<float>(yRange.start + i) / norm_rows;
    }

    // std::cout << "xRow = " << xRow << std::endl;
    // std::cout << "yCol = " << yCol << std::endl;

    // Repeat the vectors to form 2D grid
    cv::repeat(xRow, yCol.rows, 1, x_grid);
    cv::repeat(yCol, 1, xRow.cols, y_grid);
}

void steerable_gaussians2(cv::InputArray src, cv::OutputArray dst, int filter, int sigmas)
{
    cv::Mat X1 = src.getMat();

    CV_Assert(X1.type() == CV_32FC1 && X1.channels() == 1);

    int angles = 6;
    float angle_step = M_PI / (float)angles;

    std::vector<cv::Mat> G;

    // Construct steerable Gaussians
    for (int i = 0; i < 1; i++)
    {
        int Wx = (filter < 1) ? 1 : filter;
        int Wy = (filter < 1) ? 1 : filter;

        cv::Mat X = cv::Mat::zeros(Wx, Wy, CV_32FC1);
        cv::Mat Y = cv::Mat::zeros(Wx, Wy, CV_32FC1);
        cv::Range xrange(-5, 6);
        cv::Range yrange(-5, 6);
        meshgrid(xrange, yrange, X, Y);
        cv::Mat X_2, Y_2;
        cv::pow(X, 2, X_2);
        cv::pow(Y, 2, Y_2);

        cv::Mat g0;
        cv::exp(-1 * (X_2 + Y_2) / (2 * std::pow(sigmas, 2)), g0);
        g0 = g0 / (sigmas * sqrt(2 * M_PI));

        cv::Mat temp_X = g0.mul(X_2);
        cv::Mat temp_Y = g0.mul(Y_2);
        cv::Mat temp_XY = X.mul(Y);

        cv::Mat G2a = -1 * g0 / pow(sigmas, 2) + temp_X / std::pow(sigmas, 4);
        cv::Mat G2b = g0.mul(temp_XY) / std::pow(sigmas, 4);
        cv::Mat G2c = -1 * g0 / pow(sigmas, 2) + temp_Y / std::pow(sigmas, 4);

        for (int j = 0; j < angles; j++)
        {
            float angle = j * angle_step;
            cv::Mat temp_G = std::pow(cos(angle), 2) * G2a + std::pow(sin(angle), 2) * G2c - 2 * cos(angle) * sin(angle) * G2b;
            G.push_back(temp_G.clone());
        }
    }

    // Perform filtering
    cv::Mat tmp = cv::Mat::zeros(src.size(), src.type());
    for (int i = 0; i < angles; i++)
    {
        cv::Mat fil_out;
        cv::Mat fil_perform = G.at(i).clone();
        cv::filter2D(X1, fil_out, X1.depth(), fil_perform);
        tmp = tmp + fil_out;
    }

    dst.create(src.size(), src.type());
    cv::Mat Y1 = dst.getMat();
    Y1.setTo(0);
    
    Y1 = Y1 + tmp;
    // cv::imwrite("X1.tiff",X1);
    // cv::imwrite("tmp.tiff", tmp);
    // cv::imwrite("Y1.tiff", Y1);
    // for (int i = 0; i < G.size(); i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     std::cout << G.at(i) << std::endl;
    // }
}

void applyATAN2ToMat(cv::Mat &src1, cv::Mat &src2, cv::Mat &dst)
{
    /**
     * 输入参数 src1: 计算sin，cos，atan2,exp时有效
     * 输入参数 src2: 仅在计算atan2时有效
     *
     * 输出dst：计算结果
     */

    CV_Assert(!src1.empty() && !src2.empty() && !dst.empty());
    CV_Assert(src1.cols == src2.cols && src1.rows == src2.rows && dst.rows == src1.rows && dst.cols == src1.cols);
    CV_Assert(src1.type() == src2.type() && src1.type() == dst.type());
    CV_Assert(src1.channels() == src2.channels());
    CV_Assert(src1.channels() == 1);

    // cout << "atan2 function" << endl;
    // cout << "src1 size = " << src1.size() << endl;
    // cout << "src1 type = " << src1.type() << endl;

    // 判断矩阵的类型
    int type = src1.type();

    switch (type)
    {
    case CV_32FC1: // 单通道32位浮点型
        for (int i = 0; i < src1.rows; i++)
        {
            for (int j = 0; j < src1.cols; j++)
            {
                // dst.at<float>(i, j) = std::atan2(src1.at<float>(i, j), src2.at<float>(i, j));
                float temp = cv::fastAtan2(src1.at<float>(i, j), src2.at<float>(i, j)) / 180.0 * M_PI;
                if (temp >= M_PI)
                {
                    temp = temp - 2 * M_PI;
                }
                dst.at<float>(i, j) = temp;
            }
        }
        break;

    case CV_64FC1: // 单通道64位浮点型
        for (int i = 0; i < src1.rows; i++)
        {
            for (int j = 0; j < src1.cols; j++)
            {
                // dst.at<double>(i, j) = std::atan2(src1.at<double>(i, j), src2.at<double>(i, j));
                double temp = cv::fastAtan2(src1.at<double>(i, j), src2.at<double>(i, j)) / 180.0 * M_PI;
                if (temp >= M_PI)
                {
                    temp = temp - 2 * M_PI;
                }
                dst.at<double>(i, j) = temp;
            }
        }
        break;

    default:
        std::cerr << "Unsupported matrix type." << std::endl;
        break;
    }
}
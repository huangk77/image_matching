#include "phasecong3.hpp"

#ifndef BUILD_LIBS
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
#endif

PhaseCongFeature::PhaseCongFeature(int proc_w, int proc_h, int nscale, int norient, int minwave, float mult, float sigmaf, float k, float cutOff, int noiseMothod)
{
    _nscale = nscale;
    _norient = norient;
    _mult = mult;
    _minWaveLength = minwave;
    _sigmaOnf = sigmaf;
    _k = k;
    _cutOff = cutOff;
    _noiseMethod = noiseMothod;
    _rayleigh_nbins = 50;
    _epsilon = .0001;

    _row = proc_h;
    _col = proc_w;

    // Initialize zero matrix and other matrices
    zero = cv::Mat::zeros(_row, _col, CV_32FC1);
    covx2 = zero.clone();
    covy2 = zero.clone();
    covxy = zero.clone();
    EnergyV_1 = zero.clone();
    EnergyV_2 = zero.clone();
    EnergyV_3 = zero.clone();

    // Calculate xrange
    cv::Range xRange;
    if (_col % 2)
    {
        xRange = cv::Range(-(_col - 1) / 2, (_col - 1) / 2 + 1); // Equivalent to -5:5 in MATLAB
    }
    else
    {
        xRange = cv::Range(-(_col) / 2, (_col - 1) / 2 + 1);
    }

    // Calculate yrange
    cv::Range yRange;
    if (_row % 2)
    {
        yRange = cv::Range(-(_row - 1) / 2, (_row - 1) / 2 + 1);
    }
    else
    {
        yRange = cv::Range(-(_row) / 2, (_row - 1) / 2 + 1);
    }

    cv::Mat grid_x, grid_y;
    meshgrid(xRange, yRange, grid_x, grid_y, _col, _row);
    // std::cout << grid_y << std::endl;
    // std::cout << grid_x << std::endl;
    cv::Mat _theta(grid_x.size(), grid_x.type());
    grid_y = grid_y * -1;
    applyATAN2ToMat(grid_y, grid_x, _theta);
    // std::cout << _theta << std::endl;
    ifftshift(_theta);

    cv::pow(grid_x, 2, grid_x);
    cv::pow(grid_y, 2, grid_y);
    cv::add(grid_x, grid_y, _radius);
    cv::sqrt(_radius, _radius);
    ifftshift(_radius);

    _radius.at<float>(0, 0) = 1.0;

    // cout << "_radius = "<< endl;
    // cout << _radius << endl;

    cv::polarToCart(cv::Mat::ones(_theta.size(), _theta.type()), _theta, costheta, sintheta);

    // std::cout << _theta << std::endl;
    // std::cout << costheta << std::endl;
    // std::cout << sintheta << std::endl;

    lowpassfilter(lp, _row, _col, 0.45, 15);
    // cout << lp << endl;

    float div_tmp = 2 * std::pow((std::log(_sigmaOnf)), 2);
    // cout << div_tmp << endl;
    for (int i = 0; i < _nscale; i++)
    {
        // cout << "i = " << i << endl;
        cv::Mat temp;
        float fo = 1 / (_minWaveLength * std::pow((_mult), (i)));
        temp = _radius / fo;
        cv::log(temp, temp);
        cv::pow(temp, 2, temp);
        temp = -1 * temp;
        temp = temp / div_tmp;
        cv::exp(temp, temp);
        temp = temp.mul(lp);
        temp.at<float>(0, 0) = 0;
        // cout << temp << endl;
        logGabor.push_back(temp.clone());
    }

    cv::Mat pi_mat(_row, _col, CV_32FC1, cv::Scalar(M_PI));
    cv::Mat _ds_tmp(_row, _col, CV_32FC1);
    cv::Mat _dc_tmp(_row, _col, CV_32FC1);
    cv::Mat _dtheta_tmp(_row, _col, CV_32FC1);
    cv::Mat _cos_tmp(_row, _col, CV_32FC1);
    cv::Mat _sin_tmp(_row, _col, CV_32FC1);
    for (int o = 0; o < _norient; o++)
    {
        float angl = o * M_PI / _norient;
        // cout << "o = " << o << " angle = " << angl << endl;
        _ds_tmp = sintheta * cos(angl) - costheta * sin(angl);
        _dc_tmp = costheta * cos(angl) + sintheta * sin(angl);
        // std::cout << "ds" << std::endl;
        // std::cout << _ds_tmp << std::endl;
        // std::cout << "dc" << std::endl;
        // std::cout << _dc_tmp << std::endl;
        applyATAN2ToMat(_ds_tmp, _dc_tmp, _dtheta_tmp);
        _dtheta_tmp = cv::abs(_dtheta_tmp);
        // std::cout << _dtheta_tmp << std::endl;
        // cout << "atan done" << endl;
        cv::min(_dtheta_tmp * _norient / 2, pi_mat, _dtheta_tmp);
        // cout << _dtheta_tmp << endl;
        cv::polarToCart(cv::Mat::ones(_dtheta_tmp.size(), _dtheta_tmp.type()), _dtheta_tmp, _cos_tmp, _sin_tmp);
        // cout << _cos_tmp << endl;
        // cout << "polarToCart done" << endl;
        _dtheta_tmp = (_cos_tmp + 1) / 2;
        // cout << _dtheta_tmp << endl;
        spread.push_back(_dtheta_tmp.clone());
    }

    // cv::Mat egde_range(1, 51, CV_32FC1, cv::Scalar(0.0));
    edge_base.create(1, 51, CV_32FC1);
    for (int i = 0; i < 51; i++)
    {
        edge_base.at<float>(0, i) = static_cast<float>(i) / 50;
    }
    // std::cout << edge_base << std::endl;
}

void PhaseCongFeature::meshgrid(const cv::Range &xRange, const cv::Range &yRange, cv::Mat &x_grid, cv::Mat &y_grid, float norm_cols, float norm_rows)
{
    // Create x and y matrices based on the ranges
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

void PhaseCongFeature::fft2d(const Mat &src, Mat &Fourier)
{
#if 0
    cv::Mat InImage = _src.getMat();
    cv::Mat outImage = _dst.getMat();
    CV_Assert(!InImage.empty());
    CV_Assert(!InImage.channels() == 1);
    // CV_Assert(outImage.channels() == 2 && outImage.rows == InImage.rows && outImage.cols == InImage.cols);
    // CV_Assert(outImage.type() == CV_32FC2);

    namedWindow("The original image", WINDOW_NORMAL);
    imshow("The original image", InImage);

    // Extending image
    int m = getOptimalDFTSize(InImage.rows);
    int n = getOptimalDFTSize(InImage.cols);
    copyMakeBorder(InImage, InImage, 0, m - InImage.rows, 0, n - InImage.cols, BORDER_CONSTANT, Scalar(0));

    // Fourier transform
    // cv::Mat mFourier(InImage.rows + m, InImage.cols + n, CV_32FC2, Scalar(0, 0));
    outImage.create(InImage.rows + m, InImage.cols + n, CV_32FC2);
    cv::Mat mForFourier[] = {Mat_<float>(InImage), cv::Mat::zeros(InImage.size(), CV_32F)};
    cv::Mat mSrc;

    t1 = what_time_is_it_now();

    cv::merge(mForFourier, 2, mSrc);
    cv::dft(mSrc, outImage);

    t2 = what_time_is_it_now();
    printf("time = %.3f\n", t2 - t1);

    // channels[0] is the real part of Fourier transform,channels[1] is the imaginary part of Fourier transform
    vector<cv::Mat> channels;
    cv::split(outImage, channels);
    cv::Mat mRe = channels[0];
    cv::Mat mIm = channels[1];

    // Calculate the amplitude
    cv::Mat mAmplitude;
    cv::magnitude(mRe, mIm, mAmplitude);

    // Logarithmic scale
    mAmplitude += Scalar(1);
    cv::log(mAmplitude, mAmplitude);

    // The normalized
    cv::normalize(mAmplitude, mAmplitude, 0, 255, NORM_MINMAX);

    cv::Mat mResult(InImage.rows, InImage.cols, CV_8UC1, Scalar(0));
    for (int i = 0; i < InImage.rows; i++)
    {
        uchar *pResult = mResult.ptr<uchar>(i);
        float *pAmplitude = mAmplitude.ptr<float>(i);
        for (int j = 0; j < InImage.cols; j++)
        {
            pResult[j] = (uchar)pAmplitude[j];
        }
    }

    cv::Mat mQuadrant1 = mResult(Rect(mResult.cols / 2, 0, mResult.cols / 2, mResult.rows / 2));                // ROI区域的右上
    cv::Mat mQuadrant2 = mResult(Rect(0, 0, mResult.cols / 2, mResult.rows / 2));                               // ROI区域的左上
    cv::Mat mQuadrant3 = mResult(Rect(0, mResult.rows / 2, mResult.cols / 2, mResult.rows / 2));                // ROI区域的左下
    cv::Mat mQuadrant4 = mResult(Rect(mResult.cols / 2, mResult.rows / 2, mResult.cols / 2, mResult.rows / 2)); // ROI区域的右下

    cv::Mat mChange1 = mQuadrant1.clone();
    mQuadrant3.copyTo(mQuadrant1);
    mChange1.copyTo(mQuadrant3);

    Mat mChange2 = mQuadrant2.clone();
    mQuadrant4.copyTo(mQuadrant2);
    mChange2.copyTo(mQuadrant4);
    // cv::imwrite("mResult.png", mResult);
#endif
    int mat_type = src.type();
    // cout << "mat_type: " << mat_type << endl;
    assert(mat_type < 15); // Unsupported Mat datatype

    if (mat_type < 7) // Channels 1
    {
        Mat planes[] = {Mat_<float>(src), Mat::zeros(src.size(), CV_32F)};
        merge(planes, 2, Fourier);
        cv::dft(Fourier, Fourier);
    }
    else // 7 < mat_type < 15  Channels 2
    {
        Mat tmp;
        cv::dft(src, tmp);
        vector<Mat> planes;
        split(tmp, planes);
        magnitude(planes[0], planes[1], planes[0]); // Change complex to magnitude
        Fourier = planes[0];
    }
    // std::cout << Fourier.channels() << std::endl;
}

void PhaseCongFeature::ifft2d(const Mat &src, Mat &Fourier_Amp, Mat &Fourier_Real, Mat &Fourier_Imag)
{
    int mat_type = src.type();
    assert(mat_type < 15); // Unsupported Mat datatype

    if (mat_type < 7) //  Channels 1
    {
        Mat planes[] = {Mat_<float>(src), Mat::zeros(src.size(), CV_32F)};
        merge(planes, 2, Fourier_Amp);
        dft(Fourier_Amp, Fourier_Amp, DFT_INVERSE + DFT_SCALE, 0);
        Fourier_Real = Fourier_Amp.clone();
        Fourier_Imag = Fourier_Imag.clone();
    }
    else // 7 < mat_type <15  //  Channels 2
    {
        Mat tmp;

        dft(src, tmp, DFT_INVERSE + DFT_SCALE, 0);

        vector<Mat> planes;
        split(tmp, planes);
        Fourier_Real = planes[0].clone();
        Fourier_Imag = planes[1].clone();
        magnitude(planes[0], planes[1], planes[0]); // Change complex to magnitude
        Fourier_Amp = planes[0];
    }
}

void PhaseCongFeature::circshift(Mat &out, const Point &delta)
{
    Size sz = out.size();

    // error checking
    assert(sz.height > 0 && sz.width > 0);

    // no need to shift
    if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
        return;

    // delta transform
    int x = delta.x;
    int y = delta.y;
    if (x > 0)
        x = x % sz.width;
    if (y > 0)
        y = y % sz.height;
    if (x < 0)
        x = x % sz.width + sz.width;
    if (y < 0)
        y = y % sz.height + sz.height;

    // in case of multiple dimensions
    vector<Mat> planes;
    split(out, planes);

    for (size_t i = 0; i < planes.size(); i++)
    {
        // vertical
        Mat tmp0, tmp1, tmp2, tmp3;
        Mat q0(planes[i], Rect(0, 0, sz.width, sz.height - y));
        Mat q1(planes[i], Rect(0, sz.height - y, sz.width, y));
        q0.copyTo(tmp0);
        q1.copyTo(tmp1);
        tmp0.copyTo(planes[i](Rect(0, y, sz.width, sz.height - y)));
        tmp1.copyTo(planes[i](Rect(0, 0, sz.width, y)));

        // horizontal
        Mat q2(planes[i], Rect(0, 0, sz.width - x, sz.height));
        Mat q3(planes[i], Rect(sz.width - x, 0, x, sz.height));
        q2.copyTo(tmp2);
        q3.copyTo(tmp3);
        tmp2.copyTo(planes[i](Rect(x, 0, sz.width - x, sz.height)));
        tmp3.copyTo(planes[i](Rect(0, 0, x, sz.height)));
    }

    merge(planes, out);
}

void PhaseCongFeature::fftshift(Mat &out)
{
    Size sz = out.size();
    Point pt(0, 0);
    pt.x = (int)floor(sz.width / 2.0);
    pt.y = (int)floor(sz.height / 2.0);
    circshift(out, pt);
}

void PhaseCongFeature::ifftshift(Mat &out)
{
    Size sz = out.size();
    Point pt(0, 0);
    pt.x = (int)ceil(sz.width / 2.0);
    pt.y = (int)ceil(sz.height / 2.0);
    circshift(out, pt);
}

void PhaseCongFeature::applyATAN2ToMat(cv::Mat &src1, cv::Mat &src2, cv::Mat &dst)
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

void PhaseCongFeature::lowpassfilter(cv::Mat &lp, int rows, int cols, float cutOff, int n)
{
    assert(cutOff >= 0 && cutOff < 0.5);
    assert(n > 1);
    int norm_rows, norm_cols;

    // cout << "lowpass filter" << endl;
    if (lp.empty())
    {
        lp.create(rows, cols, CV_32FC1);
    }

    // Calculate xrange
    cv::Range xRange;
    if (cols % 2)
    {
        xRange = cv::Range(-(cols - 1) / 2, (cols - 1) / 2 + 1); // Equivalent to -5:5 in MATLAB
        norm_cols = cols - 1;
    }
    else
    {
        xRange = cv::Range(-(cols) / 2, (cols / 2 - 1) + 1);
        norm_cols = cols;
    }

    // Calculate yrange
    cv::Range yRange;
    if (rows % 2)
    {
        yRange = cv::Range(-(rows - 1) / 2, (rows - 1) / 2 + 1);
        norm_rows = rows - 1;
    }
    else
    {
        yRange = cv::Range(-(rows) / 2, (rows / 2 - 1) + 1);
        norm_rows = rows;
    }

    cv::Mat grid_x, grid_y;
    cv::Mat radius;
    meshgrid(xRange, yRange, grid_x, grid_y, norm_cols, norm_rows);

    cv::pow(grid_x, 2, grid_x);
    cv::pow(grid_y, 2, grid_y);

    // cout << "grid_x" << grid_x << endl;
    // cout << "grid_y" << grid_y << endl;

    cv::add(grid_x, grid_y, radius);
    cv::sqrt(radius, radius);
    radius = radius / cutOff;
    cv::pow(radius, 2 * n, radius);
    radius = radius + cv::Scalar(1);
    cv::pow(radius, -1, radius);
    ifftshift(radius);
    radius.convertTo(radius, CV_32FC1);
    lp = radius.clone();
}

float PhaseCongFeature::rayleighmode(cv::Mat &src)
{
    double minValue, maxValue; // 最大值，最小值
    cv::Point minIdx, maxIdx;  // 最小值坐标，最大值坐标
    cv::minMaxLoc(src, &minValue, &maxValue, NULL, NULL);

    // cout << "maxValue = " << maxValue << endl;
    cv::Mat edges = edge_base * maxValue;
    // cout << "edges = " << edges << endl;

    cv::Mat hist;                // 定义输出Mat类型
    int dims = 1;                // 设置直方图维度
    const int histSize[] = {51}; // 直方图每一个维度划分的柱条的数目
    // 每一个维度取值范围
    float pranges[] = {0, maxValue}; // 取值区间
    const float *ranges[] = {pranges};

    cv::calcHist(&src, 1, 0, Mat(), hist, dims, histSize, ranges, true, false); // 计算直方图
    // std::cout << hist.size() << std::endl;
    cv::minMaxLoc(hist, &minValue, &maxValue, &minIdx, &maxIdx);

    // cout << hist.size() << endl;
    // cout << hist << endl;
    // cout << minIdx << endl;
    // cout << minValue << endl;
    // cout << maxIdx << endl;
    // cout << maxValue << endl;

    float tau = (edges.at<float>(0, maxIdx.y) + edges.at<float>(0, maxIdx.y + 1)) / 2;

    // std::cout << tau << std::endl;

    return tau;
}

void PhaseCongFeature::getFeature(cv::InputArray input, cv::OutputArray output_M, cv::OutputArray output_m, cv::OutputArray output_or)
{
    assert(spread.size() == _norient);
    cv::Mat src = input.getMat();

    cv::Mat imagefft;
    CV_Assert(src.channels() == 1);
    CV_Assert(src.rows == _row && src.cols == _col);
    if (src.type() != CV_32FC1)
    {
        src.convertTo(src, CV_32FC1);
        std::cout << "Converting image to float32." << std::endl;
    }
    output_M.create(src.size(), src.type());
    output_m.create(src.size(), src.type());
    output_or.create(src.size(), src.type());

    cv::Mat _M = output_M.getMat();
    cv::Mat _m = output_m.getMat();
    cv::Mat _or = output_or.getMat();

    // t1 = what_time_is_it_now();
    fft2d(src, imagefft);
    // t2 = what_time_is_it_now();
    // std::cout << "fft2d time = " << t2 - t1 << std::endl;

    cv::Mat sumE_ThisOrient = zero.clone();
    cv::Mat sumO_ThisOrient = zero.clone();
    cv::Mat sumAn_ThisOrient = zero.clone();
    cv::Mat Energy = zero.clone();

    EnergyV_1 = zero.clone();
    EnergyV_2 = zero.clone();
    EnergyV_3 = zero.clone();
    XEnergy = zero.clone();

    double ifft_t1 = 0, ifft_t2 = 0, total_fft_time = 0;
    t1 = what_time_is_it_now();
    for (int o = 1; o <= _norient; o++)
    {

        cv::Mat EO, Amp, mReal, mImag, max_Amp;
        std::vector<cv::Mat> mReal_vec;
        std::vector<cv::Mat> mImag_vec;
        float tau = 0;
        float angl = (o - 1) * M_PI / _norient;

        sumE_ThisOrient.setTo(0);
        sumO_ThisOrient.setTo(0);
        sumAn_ThisOrient.setTo(0);
        Energy.setTo(0);

        for (int s = 1; s <= _nscale; s++)
        {

            cv::Mat filter = logGabor[s - 1].mul(spread[o - 1]);
            std::vector<cv::Mat> planes;
            cv::split(imagefft, planes);
            planes[0] = planes[0].mul(filter);
            planes[1] = planes[1].mul(filter);
            cv::merge(planes, EO);

            ifft_t1 = what_time_is_it_now();
            ifft2d(EO, Amp, mReal, mImag);
            ifft_t2 = what_time_is_it_now();
            total_fft_time += (ifft_t2 - ifft_t1);

            mReal_vec.push_back(mReal.clone());
            mImag_vec.push_back(mImag.clone());
            cv::add(sumAn_ThisOrient, Amp, sumAn_ThisOrient);
            cv::add(sumE_ThisOrient, mReal, sumE_ThisOrient);
            cv::add(sumO_ThisOrient, mImag, sumO_ThisOrient);
            if (s == 1)
            {
                // cout << "loop o = " << o << endl;
                // std::cout << sumAn_ThisOrient << std::endl;
                // med = medianMat(sumAn_ThisOrient, 4096);

                tau = rayleighmode(sumAn_ThisOrient);

                // cout << "tau = " << tau << endl;
                max_Amp = Amp.clone();
            }
            else
            {
                cv::max(max_Amp, Amp, max_Amp);
            }
        }

        cv::add(EnergyV_1, sumE_ThisOrient, EnergyV_1);
        cv::add(EnergyV_2, sumO_ThisOrient * cos(angl), EnergyV_2);
        cv::add(EnergyV_3, sumO_ThisOrient * sin(angl), EnergyV_3);
        cv::Mat E_tmp, O_tmp;
        cv::pow(sumE_ThisOrient, 2, E_tmp);
        cv::pow(sumO_ThisOrient, 2, O_tmp);
        cv::add(E_tmp, O_tmp, XEnergy);
        cv::sqrt(XEnergy, XEnergy);
        XEnergy = XEnergy + _epsilon;

        cv::divide(sumE_ThisOrient, XEnergy, MeanE);
        cv::divide(sumO_ThisOrient, XEnergy, MeanO);

        for (int s = 0; s < _nscale; s++)
        {
            cv::Mat E = mReal_vec.at(s);
            cv::Mat O = mImag_vec.at(s);
            cv::Mat mEE = E.mul(MeanE);
            cv::Mat mOO = O.mul(MeanO);
            cv::Mat mEO = E.mul(MeanO);
            cv::Mat mOE = O.mul(MeanE);
            cv::Mat mabs = cv::abs(mEO - mOE);
            cv::Mat add_tmp;
            Energy = Energy + mEE + mOO - mabs;
        }

        float totalTau = tau * (1 - std::pow(1.0 / _mult, _nscale)) / (1 - (1 / _mult));
        float EstNoiseEnergyMean = totalTau * std::sqrt(M_PI / 2);
        float EstNoiseEnergySigma = totalTau * sqrt((4 - M_PI) / 2);
        float T = EstNoiseEnergyMean + _k * EstNoiseEnergySigma;

        cv::max(Energy - T, zero, Energy);

        // cout << "Energy = " << Energy << endl;

        cv::Mat width, weight, tmp_exp;
        cv::divide(sumAn_ThisOrient, max_Amp + _epsilon, width);

        width = (width - 1) / (_nscale - 1);

        cv::exp((_cutOff - width) * _g, tmp_exp);

        weight = 1.0 / (1.0 + tmp_exp);

        cv::Mat PC = weight.mul(Energy);
        PC = PC / sumAn_ThisOrient;
        cv::Mat covx = PC * cos(angl);
        cv::Mat covy = PC * sin(angl);
        covx2 = covx2 + covx.mul(covx);
        covy2 = covy2 + covy.mul(covy);
        covxy = covxy + covx.mul(covy);

    }
    t2 = what_time_is_it_now();
    cout << "main loop time = " << t2 - t1 << endl;
    cout << "ifft time = " << total_fft_time << endl;
    cout << "fft percentage = " << total_fft_time / (t2 - t1) * 100 << "%" << endl;

    // 3ms to end
    covx2 = covx2 / (_norient / 2);
    covy2 = covy2 / (_norient / 2);
    covxy = 4 * covxy / _norient;
    cv::Mat denom = covxy.mul(covxy) + (covx2 - covy2).mul((covx2 - covy2));
    cv::sqrt(denom, denom);
    denom = denom + _epsilon;

    cv::Mat tmp_M = (covy2 + covx2 + denom) / 2;
    cv::Mat tmp_m = (covy2 + covx2 - denom) / 2;

    // Or 正确
    cv::Mat tmp_or = cv::Mat::zeros(EnergyV_2.size(), EnergyV_2.type());
    applyATAN2ToMat(EnergyV_3, EnergyV_2, tmp_or);
    cv::Mat or_mask = (tmp_or < 0);
    // std::cout << or_mask << std::endl;
    // std::cout << tmp_or << std::endl;
    cv::add(tmp_or, cv::Mat(tmp_or.size(), tmp_or.type(), cv::Scalar(M_PI)), tmp_or, or_mask);
    // std::cout << tmp_or << std::endl;
    tmp_or = tmp_or * 180.0 / M_PI;

    tmp_or.convertTo(tmp_or, CV_16SC1);
    tmp_or.convertTo(tmp_or, CV_32FC1);

    // std::cout << tmp_M.size() << std::endl;
    // cout << tmp_M << endl;
    // cout << tmp_m << endl;
    // cout << tmp_or << endl;

    tmp_M.copyTo(_M);
    tmp_m.copyTo(_m);
    tmp_or.copyTo(_or);
}

void PhaseCongFeature::DebugOut(cv::InputArray input, cv::OutputArray output_M, cv::OutputArray output_m, cv::OutputArray output_or)
{
#if 0
    // cv::Mat input = (cv::Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    // cv::Mat output;
    cv::Mat src = input.getMat();
    // cv::Mat dst = output_M.getMat();
    cv::Mat result;
    CV_Assert(src.channels() == 1);

    if (src.type() != CV_32FC1)
    {
        src.convertTo(src, CV_32FC1);
        std::cout << "Converting image to float32." << std::endl;
    }

    // src.copyTo(dst);
    t1 = what_time_is_it_now();
    // ifftshift(dst);
    fft2d(src, result);
    t2 = what_time_is_it_now();
    std::cout << "alg time: " << t2 - t1 << " ms" << std::endl;

    vector<cv::Mat> channels;
    cv::split(result, channels);
    cv::Mat mag;
    magnitude(channels[0], channels[1], mag);

    cv::imwrite("mRe.tiff", channels[0]);
    cv::imwrite("mIm.tiff", channels[1]);

    mag += cv::Scalar(1);
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 255, NORM_MINMAX);

    // mag.convertTo(mag, CV_8UC1);

    cv::Mat mResult(mag.rows, mag.cols, CV_8UC1, Scalar(0));
    for (int i = 0; i < mag.rows; i++)
    {
        uchar *pResult = mResult.ptr<uchar>(i);
        float *pAmplitude = mag.ptr<float>(i);
        for (int j = 0; j < mag.cols; j++)
        {
            pResult[j] = (uchar)pAmplitude[j];
        }
    }

    fftshift(mResult);
    output_M.create(mResult.size(), mResult.type());
    Mat dst = output_M.getMat();
    dst = mResult.clone();
#endif
    std::cout << "Debug Out OFF!\n"
              << std::endl;
}

#ifndef BUILD_LIBS
int main()
{
    // cv::Mat img = cv::imread("../images/bird.png", IMREAD_GRAYSCALE);
    //  cv::resize(img, img, cv::Size(512, 512));
    cv::Mat test_mat = cv::imread("../images/bird_512.png", IMREAD_GRAYSCALE);
    test_mat.convertTo(test_mat, CV_32FC1);
    cv::Mat phase_feat_M;
    cv::Mat phase_feat_m;
    cv::Mat phase_feat_or;

    // std::cout << test_mat << std::endl;
    cout << test_mat.size() << endl;
    PhaseCongFeature feat_maker(test_mat.cols, test_mat.rows);

    // feat_maker.getFeature(img, fft_img, cv::noArray(), cv::noArray());
    double t1 = what_time_is_it_now();
    feat_maker.getFeature(test_mat, phase_feat_M, phase_feat_m, phase_feat_or);
    double t2 = what_time_is_it_now();
    cout << "phasecong3 time = " << t2 - t1 << "ms" << endl;

    // cout << phase_feat_M << endl;
    // cout << phase_feat_m << endl;
    // cout << phase_feat_or << endl;

    // cv::imwrite("1_output_M.tiff", phase_feat_M);
    // cv::imwrite("2_output_m.tiff", phase_feat_m);
    // cv::imwrite("3_output_or.tiff", phase_feat_or);

    // feat_maker.DebugOut(test_mat, phase_feat_M, noArray(), noArray());

    // cout << fft_img.channels() << endl;
    // cout << "w_out = " << fft_img.cols << endl;
    // cout << "h_out = " << fft_img.rows << endl;

    // cv::imshow("fft", fft_img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // PhaseCongFeature feat_maker(10, 7);

    return 0;
}
#endif
#ifndef __PHASECONG_3__
#define __PHASECONG_3__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <thread>
#include <vector>
#include <valarray>

#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/time.h>

#define APPLY_SIN 0
#define APPLY_COS 1
#define APPLY_ATAN2 2
#define R2D 180 / M_PI
#define D2R M_PI / 180

using namespace std;
using namespace cv;

class PhaseCongFeature
{
public:
    PhaseCongFeature(int proc_w, int proc_h, int nscale = 4, int norient = 6, int minwave = 3, float mult = 2.1, float sigmaf = 0.55, float k = 2.0, float cutOff = 0.5, int noiseMothod = -1);

    void getFeature(cv::InputArray input, cv::OutputArray output_M, cv::OutputArray output_m, cv::OutputArray output_or);

    void DebugOut(cv::InputArray input, cv::OutputArray output_M, cv::OutputArray output_m, cv::OutputArray output_or);

protected:
    void applyATAN2ToMat(cv::Mat &src1, cv::Mat &src2, cv::Mat &dst);
    void meshgrid(const cv::Range &xRange, const cv::Range &yRange, cv::Mat &x_grid, cv::Mat &y_grid, float norm_cols = 1.0, float norm_rows = 1.0);
    // void fft2d(cv::Mat &input, cv::Mat &output);

private:
    void fft2d(const Mat &src, Mat &Fourier);
    void ifft2d(const Mat &src, Mat &Fourier, Mat &Fourier_Real, Mat &Fourier_Imag);
    void circshift(cv::Mat &out, const Point &delta);
    void lowpassfilter(cv::Mat &lp, int rows, int cols, float cutOff = 0.45, int n = 15);
    void fftshift(Mat &out);
    void ifftshift(Mat &out);
    float rayleighmode(cv::Mat &src);

    int _nscale = 4;        // Number of wavelet scales.
    int _norient = 6;       // Number of filter orientations.
    int _minWaveLength = 3; // Wavelength of smallest scale filter.
    float _mult = 2.1;      // Scaling factor between successive filters.
    float _sigmaOnf = 0.55; // Ratio of the standard deviation of the
    // Gaussian describing the log Gabor filter's
    // transfer function in the frequency domain
    // to the filter center frequency.
    float _k = 2.0; // No of standard deviations of the noise
    // energy beyond the mean at which we set the
    // noise threshold point.
    float _cutOff = 0.5; // The fractional measure of frequency spread
    // below which phase congruency values get penalized.
    int _g = 10; // Controls the sharpness of the transition in
    // the sigmoid function used to weight phase
    // congruency for frequency spread.
    int _noiseMethod = -1; // Choice of noise compensation method.

    int _rayleigh_nbins = 50;
    float _epsilon = 1e-4;

    int _row = 0, _col = 0;

    double t1 = 0, t2 = 0;

    cv::Mat zero;
    cv::Mat covx2;
    cv::Mat covy2;
    cv::Mat covxy;
    cv::Mat EnergyV_1;
    cv::Mat EnergyV_2;
    cv::Mat EnergyV_3;
    cv::Mat XEnergy;
    cv::Mat MeanE, MeanO;

    cv::Mat _radius;
    cv::Mat sintheta;
    cv::Mat costheta;

    cv::Mat lp;
    std::vector<cv::Mat> logGabor;
    std::vector<cv::Mat> ds;
    std::vector<cv::Mat> dc;
    // std::vector<cv::Mat> dtheta;
    std::vector<cv::Mat> spread;
    cv::Mat edge_base;
};

#endif // __PHASECONG_3__
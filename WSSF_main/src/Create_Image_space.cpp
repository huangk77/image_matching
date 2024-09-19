#include "Create_Image_space.hpp"


void Create_Image_space(const cv::Mat &im,
                        std::vector<cv::Mat> &Nonelinear_Scalespace,
                        std::vector<cv::Mat> &E_Scalespace,
                        std::vector<cv::Mat> &Max_Scalespace,
                        std::vector<cv::Mat> &Min_Scalespace,
                        std::vector<cv::Mat> &Phase_Scalespace,
                        bool Scale_Invariance,
                        int nOctaves,
                        float ScaleValue,
                        float ratio,
                        float sigma_1,
                        float filter)
{
    cv::Mat I_enhanced;
    cv::Mat Max_Scalespace_1, Min_Scalespace_1, Phase_Scalespace_1, E_Scalespace_1, Nonelinear_Scalespace_1, im_fil;

    int base_W = im.cols;
    int base_H = im.rows;

    Nonelinear_Scalespace_1.create(base_H, base_W, CV_32FC1);

    PhaseCongFeature phasecong3_feat_base(base_W, base_H);
    int Layers = 0;
    

    if (Scale_Invariance)
    {
        Layers = nOctaves;
    }
    else
    {
        Layers = 1;
    }

    // cout << "Layers = " << Layers << endl;

    phasecong3_feat_base.getFeature(im, Max_Scalespace_1, Min_Scalespace_1, Phase_Scalespace_1);
    Max_Scalespace.push_back(Max_Scalespace_1.clone());
    Min_Scalespace.push_back(Min_Scalespace_1.clone());
    Phase_Scalespace.push_back(Phase_Scalespace_1.clone());

    // matlab E_Scalespace{1} = imgradient(image,'prewitt');
    cv::Mat prewitt_x, prewitt_y;
    cv::filter2D(im, prewitt_x, im.depth(), kernel_prewitt_x);
    cv::filter2D(im, prewitt_y, im.depth(), kernel_prewitt_y);
    cv::add(cv::abs(prewitt_x), cv::abs(prewitt_y), E_Scalespace_1);
    E_Scalespace.push_back(E_Scalespace_1.clone());

    cv::filter2D(im, im_fil, im.depth(), gaussian_W_5);
    EPSIF(im_fil, Nonelinear_Scalespace_1);
    Nonelinear_Scalespace.push_back(Nonelinear_Scalespace_1.clone());

    std::vector<float> sigma_ls;
    for (int i = 0; i < Layers; i++)
    {
        sigma_ls.push_back(sigma_1 * std::pow(ratio, i));
    }

    for (int i = 1; i < Layers; i++)
    {
        cv::Mat prev_image = Nonelinear_Scalespace.back().clone();
        int scale_w = (int)(ceil((float)prev_image.cols / ScaleValue));
        int scale_h = (int)(ceil((float)prev_image.rows / ScaleValue));

        cv::Mat Max_Scalespace_loop, Min_Scalespace_loop, Phase_Scalespace_loop, E_Scalespace_loop;
        cv::Mat Nonelinear_Scalespace_loop = cv::Mat::zeros(scale_h, scale_w, CV_32FC1);

        cv::Mat prev_image2;
        cv::resize(prev_image, prev_image2, cv::Size(scale_w, scale_h));
        PhaseCongFeature phasecong3_feat_loop(scale_w, scale_h);
        phasecong3_feat_loop.getFeature(prev_image2, Max_Scalespace_loop, Min_Scalespace_loop, Phase_Scalespace_loop);
        Max_Scalespace.push_back(Max_Scalespace_loop.clone());
        Min_Scalespace.push_back(Min_Scalespace_loop.clone());
        Phase_Scalespace.push_back(Phase_Scalespace_loop.clone());

        EPSIF(prev_image2, Nonelinear_Scalespace_loop);
        Nonelinear_Scalespace.push_back(Nonelinear_Scalespace_loop.clone());

        cv::Mat prewitt_x_loop, prewitt_y_loop;
        cv::filter2D(prev_image2, prewitt_x_loop, prev_image2.depth(), kernel_prewitt_x);
        cv::filter2D(prev_image2, prewitt_y_loop, prev_image2.depth(), kernel_prewitt_y);
        cv::add(cv::abs(prewitt_x_loop), cv::abs(prewitt_y_loop), E_Scalespace_loop);

        E_Scalespace.push_back(E_Scalespace_loop.clone());
    }

    // for (int i = 0; i < Layers; i++)
    // {
    //     std::cout << "Nonelinear_Scalespace_" << i + 1 << " size = " << Nonelinear_Scalespace.at(i).size() << endl;
    //     std::cout << "E_Scalespace_" << i + 1 << " size = " << E_Scalespace.at(i).size() << endl;
    //     std::cout << "Max_Scalespace_" << i + 1 << " size = " << Max_Scalespace.at(i).size() << endl;
    //     std::cout << "Min_Scalespace_" << i + 1 << " size = " << Min_Scalespace.at(i).size() << endl;
    //     std::cout << "Phase_Scalespace_" << i + 1 << " size = " << Phase_Scalespace.at(i).size() << endl;
    //     std::cout << "\n"
    //               << std::endl;
    // }

    
}

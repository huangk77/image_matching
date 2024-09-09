/* ep2d_guided.cpp */
#include "mex.h"
#include "matrix.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

void filter_gray(const int j, const int M, const int w_size, const double *temporary_img_yoko, const double *temporary_img_tate, const double threshold, double *output,const double *input_img);
void filter_color(const int j, const int M, const int N, const int w_size, const double *temporary_img_yoko, const double *temporary_img_tate, const double threshold, double *output,const double *input_img);
void makedim(double *input, int M, int N, int D, double *output_tate, double *output_yoko);
void padding0(double *input, double *output, int M, int N, int D, int ws);

void ep2d_simple_gray(double *output, double *input, const int M, const int N, const int w_size, double threshold, const int iter_max, double *guide, int guide_size){
    
    int i,j,k,l;
    const int ws = (w_size-1)/2;
    int index = 0;
    int iter;
    double *temporary_img_yoko,*temporary_img_tate,*input_img;
    double computed_sum=0.0,filter_sum=0.0;
    const int TempM = M+2*ws;
    using namespace std;
    
    temporary_img_yoko = (double *)calloc((M+2*ws)*(N+2*ws),sizeof(double));
    temporary_img_tate = (double *)calloc((M+2*ws)*(N+2*ws),sizeof(double));
    input_img = (double *)calloc((M+2*ws)*(N+2*ws),sizeof(double));
    memcpy(output, input, sizeof(double) * M*N);
    
    for (iter=0;iter<iter_max;iter++){
        
        // temporaryの作成
        padding0(guide, input_img, M, N, guide_size, ws);
        makedim(input_img, M+2*ws, N+2*ws, guide_size, temporary_img_tate, temporary_img_yoko);
        
        
        // initialization
        padding0(input, input_img, M, N, 1, ws);
        
        
        // filtering
        int MaxThread = max(thread::hardware_concurrency(), 1u);
        vector<thread> worker;
        for (int ThreadNum = 0; ThreadNum < MaxThread; ThreadNum++) {
            worker.emplace_back([&](int id) {
                int r0 = N/MaxThread * id + min(N%MaxThread, id);
                int r1 = N/MaxThread * (id+1) + min(N%MaxThread, id+1);
                for (int j = r0; j < r1; j++) {   
                    filter_gray(j,M,w_size,temporary_img_yoko,temporary_img_tate,threshold,output,input_img);
                }
            }, ThreadNum);}
        for (auto& t : worker) t.join();
        
        if (iter<iter_max){
            threshold = threshold*0.5;
        }
        
    }
    
    
    // end process
    free(temporary_img_yoko);
    free(temporary_img_tate);
    free(input_img);
    
}


void filter_gray(const int j, const int M, const int w_size, const double *temporary_img_yoko, const double *temporary_img_tate, const double threshold, double *output,const double *input_img){
    double *filter_rcon_yoko,*filter_rcon_tate;
    double computed_sum,filter_sum;
    int index;
    const int ws = (w_size-1)/2;
    const int TempM = M+2*ws;
    filter_rcon_yoko = (double *)calloc(w_size*w_size,sizeof(double));
    filter_rcon_tate = (double *)calloc(w_size*w_size,sizeof(double));
    
    
    for (int i=0; i<M; i++){
        // reconstructed kernel
        for (int l=0;l<w_size;l++){
            for (int k=0;k<w_size;k++){
                index = (i+k)+(j+l)*TempM;
                filter_rcon_yoko[k+l*w_size] = fabs(temporary_img_yoko[index]-temporary_img_yoko[(i+k)+(j+ws)*TempM]);
                filter_rcon_tate[k+l*w_size] = fabs(temporary_img_tate[index]-temporary_img_tate[index-k+ws]);
            }
        }
        
        // compute&apply filter
        computed_sum = 0.0;
        filter_sum = 0.0;
        for (int l=0;l<w_size;l++){
            for (int k=0;k<w_size;k++){
                index = k+l*w_size;
                if (fmin(filter_rcon_yoko[index] + filter_rcon_tate[k+ws*w_size],filter_rcon_tate[index] + filter_rcon_yoko[index-k+ws]) < threshold){
                    computed_sum += input_img[(i+k)+(j+l)*TempM];
                    filter_sum++;
                }
            }
        }
        output[i+j*M] = computed_sum/filter_sum;
    }
    
    free(filter_rcon_yoko);
    free(filter_rcon_tate);
}




void ep2d_simple_color(double *output, double *input, const int M, const int N, const int w_size, double threshold, const int iter_max, double *guide, int guide_size){
    
    int i,j,k,l;
    const int ws = (w_size-1)/2;
    int index = 0;
    const int TempLayerSize = (M+2*ws)*(N+2*ws);
    const int TempM = (M+2*ws);
    int iter = 0;
    double *temporary_img_yoko,*temporary_img_tate,*input_img;
    double *filter_rcon_yoko,*filter_rcon_tate;
    double filter_sum[4]={0.0,0.0,0.0,0.0};
    using namespace std;
    
    temporary_img_yoko = (double *)calloc(TempLayerSize,sizeof(double));
    temporary_img_tate = (double *)calloc(TempLayerSize,sizeof(double));
    input_img = (double *)calloc(TempLayerSize*3,sizeof(double));
    memcpy(output, input, sizeof(double) * M*N*3);
    
    for (iter=0;iter<iter_max;iter++){
        
        // temporaryの作成
        padding0(guide, input_img, M, N, guide_size, ws);
        makedim(input_img, M+2*ws, N+2*ws, guide_size, temporary_img_tate, temporary_img_yoko);
        
        
        // initialization
        padding0(input, input_img, M, N, 3, ws);
        
        
        // filtering
        int MaxThread = max(thread::hardware_concurrency(), 1u);
        vector<thread> worker;
        for (int ThreadNum = 0; ThreadNum < MaxThread; ThreadNum++) {
            worker.emplace_back([&](int id) {
                int r0 = N/MaxThread * id + min(N%MaxThread, id);
                int r1 = N/MaxThread * (id+1) + min(N%MaxThread, id+1);
                for (int j = r0; j < r1; j++) {   
                    filter_color(j,M,N,w_size,temporary_img_yoko,temporary_img_tate,threshold,output,input_img);
                }
            }, ThreadNum);}
        for (auto& t : worker) t.join();
        
        
        if (iter<iter_max){
            threshold = threshold*0.5;
        }
        
    }
    
    
    
    // end process
    free(temporary_img_yoko);
    free(temporary_img_tate);
    free(input_img);
    
}


void filter_color(const int j, const int M, const int N, const int w_size, const double *temporary_img_yoko, const double *temporary_img_tate, const double threshold, double *output,const double *input_img){
    double *filter_rcon_yoko,*filter_rcon_tate;
    double filter_sum[4];
    int index;
    const int ws = (w_size-1)/2;
    const int TempLayerSize = (M+2*ws)*(N+2*ws);
    const int TempM = (M+2*ws);
    filter_rcon_yoko = (double *)calloc(w_size*w_size,sizeof(double));
    filter_rcon_tate = (double *)calloc(w_size*w_size,sizeof(double));
    
    
    for (int i=0; i<M; i++){
                
                // reconstructed kernel
                for (int l=0;l<w_size;l++){
                    for (int k=0;k<w_size;k++){
                        index = (i+k)+(j+l)*TempM;
                        filter_rcon_yoko[k+l*w_size] = fabs(temporary_img_yoko[index]-temporary_img_yoko[(i+k)+(j+ws)*TempM]);
                        filter_rcon_tate[k+l*w_size] = fabs(temporary_img_tate[index]-temporary_img_tate[index-k+ws]);
                    }
                }
                
                //compute&apply filter
                filter_sum[0] = 0.0; // filter sum
                filter_sum[1] = 0.0; // R sum
                filter_sum[2] = 0.0; // G sum
                filter_sum[3] = 0.0; // B sum
                for (int l=0;l<w_size;l++){
                    for (int k=0;k<w_size;k++){
                        index = k+l*w_size;
                        if (fmin(filter_rcon_tate[index] + filter_rcon_yoko[index-k+ws],filter_rcon_yoko[index] + filter_rcon_tate[k+ws*w_size]) < threshold){
                            index = (i+k)+(j+l)*TempM;
                            filter_sum[0] ++;
                            filter_sum[1] += input_img[index];
                            filter_sum[2] += input_img[index + TempLayerSize];
                            filter_sum[3] += input_img[index + 2*TempLayerSize];
                        }
                    }
                }
                filter_sum[0]=1.0/filter_sum[0];
                output[i+j*M] = filter_sum[1]*filter_sum[0];
                output[i+j*M+M*N] = filter_sum[2]*filter_sum[0];
                output[i+j*M+2*M*N] = filter_sum[3]*filter_sum[0];
                
            }
    
    free(filter_rcon_yoko);
    free(filter_rcon_tate);
}


void makedim(double *input, int M, int N, int D, double *output_tate, double *output_yoko){
    int layersize = M*N;   
    int i,j;
    
    if (D==1){
        for (j=1; j<N; j++){
            for (i=1; i<M; i++){
                output_yoko[i+j*M] = fabs(input[i+j*M]-input[i+(j-1)*M])+ output_yoko[i+(j-1)*M];
                output_tate[i+j*M] = fabs(input[i+j*M]-input[i-1+j*M])+ output_tate[i-1+j*M];
            }
        }
    }else{
        for (j=1; j<N; j++){
            for (i=1; i<M; i++){
                output_yoko[i+j*M] = fabs(input[i+j*M]-input[i+(j-1)*M])
                                                + fabs(input[i+j*M+layersize]-input[i+(j-1)*M+layersize])
                                                + fabs(input[i+j*M+2*layersize]-input[i+(j-1)*M+2*layersize])
                                                + output_yoko[i+(j-1)*M];
                output_tate[i+j*M] = fabs(input[i+j*M]-input[i-1+j*M])
                                                + fabs(input[i+j*M+layersize]-input[i-1+j*M+layersize])
                                                + fabs(input[i+j*M+2*layersize]-input[i-1+j*M+2*layersize])
                                                + output_tate[i-1+j*M];
            }
        }
        
    }
    
}



void padding0(double *input, double *output, int M, int N, int D, int ws){
    int i,j,color;
    
    for (color=0;color<D;color++){
        for (j=ws; j<N+ws; j++){
            for (i=ws; i<M+ws; i++){
                output[i+j*(M+2*ws)+color*(M+2*ws)*(N+2*ws)] = input[(i-ws)+(j-ws)*M + color*M*N];
            }
        }
    }
}


void mexFunction( int Nreturned, mxArray *returned[], int Noperand, const mxArray *operand[] ){
    double *input, *output;
    const mwSize *input_size_temp;
    mwSize input_size[3];
    mwSize dim_count;
    int guide_size;
    
    /* 画像の次元数の判定 */
    dim_count = mxGetNumberOfDimensions(operand[0]);
    input_size_temp = mxGetDimensions(operand[0]);
    if (dim_count < 3){
        input_size[0] = input_size_temp[0];
        input_size[1] = input_size_temp[1];
        input_size[2] = 1;
    }else{
        input_size[0] = input_size_temp[0];
        input_size[1] = input_size_temp[1];
        input_size[2] = input_size_temp[2];
    }
    if (mxGetNumberOfDimensions(operand[4]) < 3){
        guide_size = 1;
    }else{
        guide_size = 3;
    }
    
    // 返り値用の領域を確保
    returned[0] = mxCreateNumericArray(dim_count,input_size, mxDOUBLE_CLASS, mxREAL);
    
    // Matlab側の変数のアドレスをC側の変数にコピーする
    input = mxGetPr(operand[0]);
    output = mxGetPr(returned[0]);
    
    //multithread
    //printf("Processor:%d\n",std::thread::hardware_concurrency());
    
    // 処理を実行
    if (input_size[2] == 1){
        ep2d_simple_gray(output,input,input_size[0],input_size[1],(int) *mxGetPr(operand[1]),*mxGetPr(operand[2]),(int) *mxGetPr(operand[3]),mxGetPr(operand[4]),guide_size);
    }else{
        ep2d_simple_color(output,input,input_size[0],input_size[1],(int) *mxGetPr(operand[1]),*mxGetPr(operand[2]),(int) *mxGetPr(operand[3]),mxGetPr(operand[4]),guide_size);
    }
}

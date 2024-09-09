function [Nonelinear_Scalespace,E_Scalespace,Max_Scalespace,Min_Scalespace,Phase_Scalespace]=Create_Image_space(im,nOctaves,Scale_Invariance,ScaleValue,...
                                                        ratio,sigma_1,...
                                                        filter)
% addpath('./edges-master/');
% model = load('./edges-master/models/forest/modelBsds.mat'); 

if (size(im, 3)==3)
    dst=rgb2gray(im);
else
    dst = im;
end
image=double(dst(:,:,1));

[M,N]=size(image);

if (strcmp(Scale_Invariance  ,'YES'))
    Layers=nOctaves;
else
    Layers=1;
end

Nonelinear_Scalespace=cell(1,Layers);
Image_Scalespace=cell(1,Layers);
E_Scalespace=cell(1,Layers);
Max_Scalespace=cell(1,Layers);
Min_Scalespace=cell(1,Layers);
Phase_Scalespace=cell(1,Layers);

for i=1:1:Layers
    Nonelinear_Scalespace{i}=zeros(M,N);
    Image_Scalespace{i}=zeros(M,N);
    E_Scalespace{i}=zeros(M,N);
end
% tic;
[Max_Scalespace{1}, Min_Scalespace{1},Phase_Scalespace{1}, ~] = phasecong3(image,4,6);
% disp(['phasecong：',num2str(toc),' 秒']);
% E = gpuArray(edgesDetect(gather(cat(3,image,image,image)), model.model));
% E_Scalespace{1} = gather(E);  
E_Scalespace{1} = imgradient(image,'prewitt');
        
windows_size=round(filter);
W=fspecial('gaussian',[windows_size windows_size],sigma_1);      
image=imfilter(image,W,'replicate');

% use fast guided filter
% r = 16;
% eps = 0.1^2;
% s = 4;
% tic
% q_sub = fastguidedfilter(image, image, r, eps, s);
% I_enhanced_sub = (image - q_sub) * 5 + q_sub;
% toc
I_enhanced_sub = EPSIF(image);
Nonelinear_Scalespace{1}=I_enhanced_sub; 

% Nonelinear_Scalespace{1}=image;       
sigma=zeros(1,Layers);
for i=1:Layers
    sigma(i)=sigma_1*ratio^(i-1);
end


for i=2:Layers

    prev_image = Nonelinear_Scalespace{1,i-1};
    prev_image2 = imresize(prev_image,1/ScaleValue,'bilinear');    
    if size(prev_image2,3)~=1
        phase_image = rgb2gray(prev_image2); 
    else
        prev_image2 = cat(3, prev_image2,prev_image2,prev_image2);
        phase_image = rgb2gray(prev_image2); 
    end    
    [Max_Scalespace{i}, Min_Scalespace{i},Phase_Scalespace{i}, ~] = phasecong3(phase_image,4,6);
    params.sigmaW = sigma(i);
    params.LineRadius = round(filter*sigma(i)*sigma(i)/2);
    % use fast guided filter
%     q_sub = fastguidedfilter(phase_image, phase_image, r, eps, s);
%     I_enhanced_sub = (phase_image - q_sub) * 5 + q_sub;
    % use edge guide filter
    I_enhanced_sub = EPSIF(phase_image);
    Nonelinear_Scalespace{i} = I_enhanced_sub;
%     E = gpuArray(edgesDetect(gather(cat(3,phase_image,phase_image,phase_image)), model.model));
%     E_Scalespace{i} = gather(E);  
    E_Scalespace{i} = imgradient(phase_image,'prewitt');

%     [Nonelinear_Scalespace{i},E_Scalespace{i}]=SAF(prev_image2, params);    
end

end


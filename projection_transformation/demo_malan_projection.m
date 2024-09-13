% this code is used to transform image from other pose to birf view
% step1: make sure the trans from image1 to image0

clc
clear
close all
%% read data
% image0 = imread('./real/ir_image_0.png');
image1 = imread('./real/down.jpg');
tic
%% load the camera parm, rotation and translation
focal_length = 24; % mm
sensor_size = 7.73; % mm
camera_width = 640; % px
camera_height = 512; % px
dx = sensor_size / camera_width;
dy = dx;

fx = focal_length / dx;
fy = focal_length / dy;
cx = camera_width / 2;
cy = camera_height / 2;

K_camera = [fx,0,cx;
            0,fy,cy;
            0,0,1];

%% given the euler and translation in world coordinate axis

euler_w2uav0 = [0,0,0]; %deg
translation_w2uav0 = [0,-200,400]; % m
R_w2uav0 = euler2rotation_matrix(euler_w2uav0);

euler_w2uav1 = [9,10,20]; % deg
translation_w2uav1 = [0,-180,320]; % m
R_w2uav1 = euler2rotation_matrix(euler_w2uav1);

euler_uav2cam = [180,0,0]; % deg
translation_uav2cam = [0,0,0]; % m
R_uav2cam = euler2rotation_matrix(euler_uav2cam);

% T_AB: the transform of A to B, T denotes [R,t;,0,1];
% T_AB = T_CB * T_CA^-1

T_w2uav0 = [R_w2uav0,translation_w2uav0';0,0,0,1];
T_w2uav1 = [R_w2uav1,translation_w2uav1';0,0,0,1];
T_uav2cam = [R_uav2cam,translation_uav2cam';0,0,0,1];

T_w2cam0 = T_w2uav0 * inv(T_uav2cam);
T_w2cam1 = T_w2uav1 * inv(T_uav2cam);


R_w2cam0  = T_w2cam0(1:3,1:3);
R_w2cam1  = T_w2cam1(1:3,1:3);

T_cam12cam0 = inv(T_w2cam0) * T_w2cam1;

R_cam12cam0 = T_cam12cam0(1:3,1:3);
% translation_cam12cam0 = T_cam12cam0(1:3,4);

T_cam02cam1 = inv(T_w2cam1) * T_w2cam0;

translation_cam02cam1 = T_cam02cam1(1:3,4);
%% given the plane defined in cam1, where you want to tans from
% plane normal vector satisfied n^T.p + d = 0, where p is the 3D points in
% camera1

n_world = [0,0,1]; % m
% n_cam1 = R_w2cam1 * n_world;
n_cam1 = n_world * R_w2cam1;

d_cam1 = 400; % m GPS relative height

%% compute the homography matrix from cam1 to cam0
% which stasified: q0 = H_01 * q1

I = diag([1,1,1]);
K_camera0 = K_camera; K_camera1 = K_camera;
H_cam12cam0 = K_camera0 * R_cam12cam0 * (I + translation_cam02cam1*n_cam1/d_cam1) * inv(K_camera1);
% H_cam12cam0 = K_camera0 * (R_cam12cam0 - translation_cam12cam0 * n_cam1/d_cam1) * inv(K_camera1);

% in matlab: [u0, v0, 1] = [u1, v1, 1] * H_01'
tform = projective2d(H_cam12cam0');

outref = imref2d([1080,1920]);
% use imwarp to project image1 to image0
% image_trans = imwarp(image1, tform,"OutputView",outref);
image_trans = imwarp(image1, tform);
toc

imwrite(image_trans,'./test_muti_modal/ir_image_trans_3.png')

%% show project results

figure;
imshow(image_trans);

% figure;
% imshow(image0);
% 
% figure;
% imshow(image1);

function rotation = euler2rotation_matrix(euler)

% [alpha,beta,gamma] = euler;
alpha = euler(1);
beta = euler(2);
gamma = euler(3);

RX = [1, 0, 0;
      0, cosd(alpha), -sind(alpha);
      0, sind(alpha), cosd(alpha)];

RY = [cosd(beta), 0, sind(beta);
     0, 1, 0;
     -sind(beta), 0, cosd(beta)];

RZ = [cosd(gamma), -sind(gamma), 0;
     sind(gamma), cosd(gamma), 0;
     0, 0, 1];

% if use 'XYZ' to rotation, then R = RZ*RY*RX;

R = RZ*RY*RX;

rotation = R;
end


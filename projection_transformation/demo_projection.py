import cv2
import numpy as np
import time
def camera_parm():
    focal_length = 24  # mm
    sensor_size = 7.73  # mm
    camera_width = 640  # px
    camera_height = 512  # px
    dx = sensor_size / camera_width
    dy = dx

    fx = focal_length / dx
    fy = focal_length / dy
    cx = camera_width / 2
    cy = camera_height / 2

    k_camera = np.array(
                [[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
    return k_camera

def euler2rotation_matrix(euler):
    alpha = euler[0] / 180 * np.pi
    beta = euler[1] / 180 * np.pi
    gamma = euler[2] / 180 * np.pi

    r_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])

    r_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])

    r_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])

    rotation_matrix = np.dot(r_z, np.dot(r_y, r_x))
    return rotation_matrix

def calculate_homography_matrix(k_camera, euler_w2uav, euler_uav2cam):
    R_w2uav = euler2rotation_matrix(euler_w2uav)
    R_uav2cam = euler2rotation_matrix(euler_uav2cam)

    R_w2uav_bird = np.eye(3)
    R_w2cam_bird = np.dot(R_w2uav_bird, R_uav2cam.T)  # R_w2cam = R_w2uav * R_uav2cam
    R_w2cam = np.dot(R_w2uav, R_uav2cam.T)  # R_w2cam = R_uav2cam * inv(R_w2uav)

    R_cam2cam_bird = np.dot(R_w2cam_bird.T, R_w2cam)  # R_cam2cam0 = R_w2uav * inv(R_w2cam)

    H_cam2cam_bird = np.dot(np.dot(k_camera, R_cam2cam_bird), np.linalg.inv(k_camera))

    return H_cam2cam_bird

if __name__ == '__main__':
    img = cv2.imread('./image0.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cam_parm = camera_parm()

    euler_w2uav = np.array([9, 10, 20])
    euler_uav2cam = np.array([180, 0, 0])
    t1 = time.time()
    homography_matrix = calculate_homography_matrix(cam_parm, euler_w2uav, euler_uav2cam)

    result = cv2.warpPerspective(img, homography_matrix, (img.shape[1], img.shape[0]))
    t2 = time.time()
    print('time:', t2 - t1)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
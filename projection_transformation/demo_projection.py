import cv2
import numpy as np
import time
def camera_parm():
    """ Camera intrinsic matrix

    Returns:
        k_camera (_type_): camera intrinsic matrix, (ndarray, (3, 3))
    """
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

def euler2rotation_matrix(euler, is_internal_rotation=False):
    """ Convert euler angle to rotation matrix

    Args:
        euler (_type_): euler angle, radian (ndarray, (3,))
        is_internal_rotation (_type_): internal rotation or external rotation (bool)

    Returns:
        rotation_matrix (_type_): rotation matrix (ndarray, (3, 3))
    """
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

    if is_internal_rotation:
        # case: internal rotation use Z-X-Y order, which is yaw-pitch-roll
        rotation_matrix = np.dot(r_y, np.dot(r_x, r_z))
    else:
        # case: external rotation use Z-Y-X order
        rotation_matrix = np.dot(r_z, np.dot(r_y, r_x))
    return rotation_matrix

def calculate_homography_matrix(k_camera, euler_w2uav, euler_uav2cam):
    """ Calculate the homography matrix between two images

    Args:
        k_camera (_type_): camera intrinsic matrix, (ndarray, (3, 3))
        euler_w2uav (_type_): euler angle from world to uav, radian (ndarray, (3,))
        euler_uav2cam (_type_): euler angle from uav to camera, radian (ndarray, (3,))

    Returns:
        H_cam2cam_bird (_type_): homography matrix between two images, (ndarray, (3, 3))
    """
    R_w2uav = euler2rotation_matrix(euler_w2uav, is_internal_rotation=True)  # in the ENU coordinate system (pitch, roll, yaw)
    R_uav2cam = euler2rotation_matrix(euler_uav2cam, is_internal_rotation=True)  # in the UAV body coordinate system (pitch, roll, yaw)
    # pthch, roll, yaw, counterclockwise positive

    R_w2uav_bird = np.eye(3)
    R_w2cam_bird = np.dot(R_w2uav_bird, R_uav2cam.T)  # R_w2cam = R_w2uav * R_uav2cam
    R_w2cam = np.dot(R_w2uav, R_uav2cam.T)  # R_w2cam = R_uav2cam * inv(R_w2uav)

    R_cam2cam_bird = np.dot(R_w2cam_bird.T, R_w2cam)  # R_cam2cam0 = R_w2uav * inv(R_w2cam)

    H_cam2cam_bird = np.dot(np.dot(k_camera, R_cam2cam_bird), np.linalg.inv(k_camera))

    return H_cam2cam_bird

def correct_H(H, target_location, w, h):
    """ Correct the homography matrix so that the projected image is fully displayed

    Args:
        H (_type_): homography matrix between two images, (ndarray, (3, 3))
        w (_type_): image width
        h (_type_): image height

    Returns:
        H (_type_): correct homography matrix, (ndarray, (3, 3))
        correct_w: projection width
        correct_h: projection height
    """
    corner_pts = np.array([[[0, 0], [w, 0], [0, h], [w, h], [w/2, h/2]]], dtype=np.float32)
    corner_pts = np.concatenate((corner_pts, target_location), axis=1)
    min_out_w, min_out_h = cv2.perspectiveTransform(corner_pts, H)[0].min(axis=0)
    projected_center_points = np.array([[w / 2, h / 2]])
    H[0, :] -= H[2, :] * min_out_w.astype(np.int)
    H[1, :] -= H[2, :] * min_out_h.astype(np.int)
    projected_location_points = projected_center_points - np.array([min_out_w, min_out_h])
    projected_points = cv2.perspectiveTransform(corner_pts, H)
    correct_w, correct_h = projected_points[0].max(axis=0).astype(np.int)
    projected_center_points = projected_points[:, 4, :2]
    projected_target_points = projected_points[:, 5, :2]

    # new_projected_center_points = cv2.perspectiveTransform(projected_center_points, H)[0]

    return H, correct_w, correct_h, projected_location_points, projected_center_points, projected_target_points


if __name__ == '__main__':
    img = cv2.imread('./image1.png')
    cam_parm = camera_parm()
    use_correct = True
    euler_w2uav = np.array([9, 10, 20])
    euler_uav2cam = np.array([180, 0, 0])
    t1 = time.time()
    homography_matrix = calculate_homography_matrix(cam_parm, euler_w2uav, euler_uav2cam)
    target_location = np.array([[[256, 512]]])
    if use_correct:
        homography_matrix_correct, correct_w, correct_h, projected_location_points, projected_center_points, projected_target_points = correct_H(homography_matrix, target_location, img.shape[1], img.shape[0])
        result = cv2.warpPerspective(img, homography_matrix_correct, (correct_w, correct_h))
    else:
        result = cv2.warpPerspective(img, homography_matrix, (img.shape[1], img.shape[0]))
    t2 = time.time()
    print('time:', t2 - t1)

    cv2.putText(result, 'location', (int(projected_location_points[0][0]), int(projected_location_points[0][1])),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
    cv2.circle(result, (int(projected_location_points[0][0]), int(projected_location_points[0][1])), 5, (0, 0, 255), -1)

    cv2.putText(result, 'center', (int(projected_center_points[0][0]), int(projected_center_points[0][1])),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
    cv2.circle(result, (int(projected_center_points[0][0]), int(projected_center_points[0][1])), 5, (0, 0, 255), -1)

    cv2.putText(result, 'target', (int(projected_target_points[0][0]), int(projected_target_points[0][1])),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
    cv2.circle(result, (int(projected_target_points[0][0]), int(projected_target_points[0][1])), 5, (0, 0, 255), -1)

    cv2.arrowedLine(result, (int(projected_location_points[0][0]), int(projected_location_points[0][1])),
                    (int(projected_center_points[0][0]), int(projected_center_points[0][1])), (0, 0, 255), 2, tipLength=0.05)

    cv2.arrowedLine(result, (int(projected_location_points[0][0]), int(projected_location_points[0][1])),
                    (int(projected_target_points[0][0]), int(projected_target_points[0][1])), (0, 0, 255), 2,
                    tipLength=0.05)

    vec_center_target_original = projected_target_points - projected_location_points
    vec_center_target = vec_center_target_original / np.linalg.norm(vec_center_target_original)

    vec_center_location_original = projected_center_points - projected_location_points
    vec_center_location = vec_center_location_original / np.linalg.norm(vec_center_location_original)

    yaw_angle = np.arccos(np.dot(vec_center_target[0], vec_center_location[0]))
    yaw_angle = yaw_angle / np.pi * 180

    # init value to solve the pitch angle
    height = 400  # m
    pitch = 9  # degree

    # calculate the pitch angle by vector projection
    projected_target2center_location = np.dot(vec_center_target_original[0], vec_center_location_original[0]) / np.linalg.norm(vec_center_location_original)
    real_projected_target2center_location = projected_target2center_location * height / (np.linalg.norm(vec_center_location_original) * np.tan(pitch / 180 * np.pi))
    pitch_angle = pitch - np.arctan(height / real_projected_target2center_location) / np.pi * 180

    print('miss angle yaw:', yaw_angle)  # 逆时针为正，顺时针为负
    print('miss angle pitch:', pitch_angle)  # 逆时针为正，顺时针为负

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
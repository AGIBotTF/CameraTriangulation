import cv2
import yaml
import numpy as np

def load_calibration_yaml(path):
    with open(path, 'r') as file:
        calib_data = yaml.safe_load(file)
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['dist_coeff'])
    return camera_matrix, dist_coeffs

camera_matrix1, dist_coeffs1 = load_calibration_yaml('calibration_matrix1.yaml')
camera_matrix2, dist_coeffs2 = load_calibration_yaml('calibration_matrix2.yaml')

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1:
        frame1_undistorted = cv2.undistort(frame1, camera_matrix1, dist_coeffs1)
        cv2.imshow('Camera 1', frame1_undistorted)

    if ret2:
        frame2_undistorted = cv2.undistort(frame2, camera_matrix2, dist_coeffs2)
        cv2.imshow('Camera 2', frame2_undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

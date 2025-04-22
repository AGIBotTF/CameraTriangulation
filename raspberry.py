import cv2
import yaml
import numpy as np
import requests
import time
import base64

SERVER_URL = 'http://192.168.35.27:5000/detect'
FPS_LIMIT = 10


def load_calibration_yaml(path):
    with open(path, 'r') as file:
        calib_data = yaml.safe_load(file)
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['dist_coeff'])
    return camera_matrix, dist_coeffs


class CameraSystem:
    def __init__(self):
        self.camera_matrix1, self.dist_coeffs1 = load_calibration_yaml('calibration_matrix1.yaml')
        self.camera_matrix2, self.dist_coeffs2 = load_calibration_yaml('calibration_matrix2.yaml')

        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(2)

        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            raise ValueError("Could not open one or both cameras")

        self.last_sent_time = 0
        self.min_interval = 1.0 / FPS_LIMIT

    def capture_synchronized_frames(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if not ret1 or not ret2:
            return None, None

        frame1_undistorted = cv2.undistort(frame1, self.camera_matrix1, self.dist_coeffs1)
        frame2_undistorted = cv2.undistort(frame2, self.camera_matrix2, self.dist_coeffs2)
        frame2_flipped = cv2.flip(frame2_undistorted, -1)

        return frame1_undistorted, frame2_flipped

    def encode_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def send_frames_to_server(self, frame1, frame2):
        current_time = time.time()

        if current_time - self.last_sent_time < self.min_interval:
            return None

        encoded_frame1 = self.encode_frame(frame1)
        encoded_frame2 = self.encode_frame(frame2)

        payload = {
            'frame1': encoded_frame1,
            'frame2': encoded_frame2,
            'timestamp': current_time
        }

        try:
            response = requests.post(SERVER_URL, json=payload, timeout=2.0)
            self.last_sent_time = current_time

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error from server: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def run(self):
        try:
            while True:
                frame1, frame2 = self.capture_synchronized_frames()

                if frame1 is None or frame2 is None:
                    print("Failed to capture frames")
                    time.sleep(0.1)
                    continue

                result = self.send_frames_to_server(frame1, frame2)
                print(result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(max(0.0, self.min_interval - (time.time() - self.last_sent_time)))

        finally:
            self.cap1.release()
            self.cap2.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_system = CameraSystem()
    camera_system.run()
import numpy as np
import cv2
import yaml
import time
from ultralytics import YOLO
import math


def load_calibration(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coeff'])
    return mtx, dist


def triangulate_point(P1, P2, point1, point2):
    point1 = np.array([point1[0], point1[1], 1.0]).reshape(3, 1)
    point2 = np.array([point2[0], point2[1], 1.0]).reshape(3, 1)

    A = np.zeros((4, 4))
    A[0] = point1[0] * P1[2] - P1[0]
    A[1] = point1[1] * P1[2] - P1[1]
    A[2] = point2[0] * P2[2] - P2[0]
    A[3] = point2[1] * P2[2] - P2[1]

    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    X = X / X[3]
    return X[:3]


def setup_cameras():
    camera_matrix1, dist_coeffs1 = load_calibration('../calibration_matrix1.yaml')
    camera_matrix2, dist_coeffs2 = load_calibration('../calibration_matrix2.yaml')

    R = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]], dtype=np.float32)

    t = np.array([[0.16], [0.09], [0]], dtype=np.float32)

    P1 = camera_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = camera_matrix2 @ np.hstack((R, t))

    midpoint = np.array([t[0, 0] / 2, t[1, 0] / 2, t[2, 0] / 2])

    return camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, P1, P2, R, t, midpoint


def correct_upside_down(image):
    return cv2.rotate(image, cv2.ROTATE_180)


def process_detections(detections1, detections2, P1, P2, frame1, frame2, midpoint):
    results = []
    bottle_found = False

    objects1 = {}
    objects2 = {}

    for det in detections1:
        boxes = det.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            class_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = model.names[class_id]

            if class_name not in objects1:
                objects1[class_name] = []

            objects1[class_name].append({
                'center': (center_x, center_y),
                'conf': conf,
                'box': (x1, y1, x2, y2)
            })

    for det in detections2:
        boxes = det.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            class_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = model.names[class_id]

            if class_name not in objects2:
                objects2[class_name] = []

            objects2[class_name].append({
                'center': (center_x, center_y),
                'conf': conf,
                'box': (x1, y1, x2, y2)
            })

    if 'bottle' in objects1 and 'bottle' in objects2:
        bottle1 = objects1['bottle'][0]
        bottle2 = objects2['bottle'][0]

        point1 = bottle1['center']
        point2 = bottle2['center']
        confidence = min(bottle1['conf'], bottle2['conf'])

        point3d = triangulate_point(P1, P2, point1, point2)

        relative_position = point3d - midpoint

        results.append({
            'class_name': 'bottle',
            'position_3d': point3d,
            'relative_position': relative_position,
            'confidence': confidence
        })
        bottle_found = True

        box1 = bottle1['box']
        box2 = bottle2['box']

        cv2.rectangle(frame1, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 255, 0), 2)
        cv2.rectangle(frame2, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 255, 0), 2)

        position_text = f"Bottle: X={relative_position[0]:.2f}m Y={relative_position[1]:.2f}m Z={relative_position[2]:.2f}m"
        cv2.putText(frame1, position_text, (int(box1[0]), int(box1[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return results, frame1, frame2, bottle_found


def calculate_distance(point):
    return math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)


def main():
    global model
    print("Loading YOLOv11 model...")
    model = YOLO('../yolo11n.pt')

    print("Setting up cameras and loading calibration...")
    camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, P1, P2, R, t, midpoint = setup_cameras()

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both cameras.")
        return

    print("System ready. Looking for bottles...")
    print("Press 'q' to quit")

    last_print_time = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Could not read frames from one or both cameras.")
            break

        frame2 = correct_upside_down(frame2)

        undistorted1 = cv2.undistort(frame1, camera_matrix1, dist_coeffs1)
        undistorted2 = cv2.undistort(frame2, camera_matrix2, dist_coeffs2)

        detections1 = model(undistorted1)
        detections2 = model(undistorted2)

        results, processed_frame1, processed_frame2, bottle_found = process_detections(
            detections1, detections2, P1, P2, undistorted1.copy(), undistorted2.copy(), midpoint
        )

        current_time = time.time()
        if bottle_found and (current_time - last_print_time) > 1.0:
            last_print_time = current_time
            for obj in results:
                if obj['class_name'] == 'bottle':
                    rel_pos = obj['relative_position']
                    print(f"\nBottle detected!")
                    print(
                        f"Position relative to midpoint: X={rel_pos[0]:.3f}m, Y={rel_pos[1]:.3f}m, Z={rel_pos[2]:.3f}m")
                    print(f"Confidence: {obj['confidence']:.2f}")

        cv2.imshow('Camera 1', processed_frame1)
        cv2.imshow('Camera 2', processed_frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import threading
import time
from ultralytics import YOLO
import math

app = Flask(__name__)

last_frame1 = None
last_frame2 = None
last_results1 = []
last_results2 = []
last_matched_objects = []

frame_lock = threading.Lock()

model = YOLO('yolo11n.pt')

CAMERA_HORIZONTAL_OFFSET = 15.5  # cm
CAMERA_VERTICAL_OFFSET = 9.3  # cm
FOCAL_LENGTH = 3.6  # mm
LENS_ANGLE = 96  # degrees


def decode_image(encoded_image):
    img_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def draw_detections(image, results):
    annotated_img = image.copy()

    if results and len(results) > 0:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                label = f"{result.names[int(cls)]}: {conf:.2f}"

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)

                cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated_img


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def get_edges(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1, y1, x1, y2), (x2, y1, x2, y2), (x1, y1, x2, y1), (x1, y2, x2, y2)]


def calculate_position_from_edges(bbox1, bbox2, img_width, img_height):
    edges1 = get_edges(bbox1)
    edges2 = get_edges(bbox2)

    positions = []

    center1 = get_center(bbox1)
    center2 = get_center(bbox2)
    center_position = calculate_position_point(center1, center2, img_width, img_height)
    if center_position[2] > 0:
        positions.append(center_position)

    corners1 = [(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]),
                (bbox1[0], bbox1[3]), (bbox1[2], bbox1[3])]
    corners2 = [(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]),
                (bbox2[0], bbox2[3]), (bbox2[2], bbox2[3])]

    for i in range(4):
        corner_position = calculate_position_point(corners1[i], corners2[i], img_width, img_height)
        if corner_position[2] > 0:
            positions.append(corner_position)

    if positions:
        avg_position = [
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions),
            sum(pos[2] for pos in positions) / len(positions)
        ]
        return avg_position
    else:
        return calculate_position_point(center1, center2, img_width, img_height)


def calculate_position_point(point1, point2, img_width, img_height):
    x1_norm = (point1[0] - img_width / 2) / (img_width / 2)
    y1_norm = (point1[1] - img_height / 2) / (img_height / 2)

    x2_norm = (point2[0] - img_width / 2) / (img_width / 2)
    y2_norm = (point2[1] - img_height / 2) / (img_height / 2)

    angle_h_rad = math.radians(LENS_ANGLE / 2)
    angle_v_rad = math.radians((LENS_ANGLE * img_height / img_width) / 2)

    theta1_h = x1_norm * angle_h_rad
    theta1_v = y1_norm * angle_v_rad

    theta2_h = x2_norm * angle_h_rad
    theta2_v = y2_norm * angle_v_rad

    disparity_h = theta1_h - theta2_h

    if abs(disparity_h) < 0.001:
        return [0, 0, 1000] # mnogo nadaleko

    z = CAMERA_HORIZONTAL_OFFSET / (2 * math.tan(disparity_h / 2))

    x = z * math.tan((theta1_h + theta2_h) / 2)
    y = z * math.tan((theta1_v + theta2_v) / 2) - CAMERA_VERTICAL_OFFSET / 2

    return [x, y, z]


def filter_outliers(positions):
    if len(positions) < 4:
        return positions

    x_values = [pos[0] for pos in positions]
    y_values = [pos[1] for pos in positions]
    z_values = [pos[2] for pos in positions]

    def get_quartiles(values):
        values = sorted(values)
        n = len(values)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        return values[q1_idx], values[q3_idx]

    q1_x, q3_x = get_quartiles(x_values)
    q1_y, q3_y = get_quartiles(y_values)
    q1_z, q3_z = get_quartiles(z_values)

    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y
    iqr_z = q3_z - q1_z

    lower_x = q1_x - 1.5 * iqr_x
    upper_x = q3_x + 1.5 * iqr_x

    lower_y = q1_y - 1.5 * iqr_y
    upper_y = q3_y + 1.5 * iqr_y

    lower_z = q1_z - 1.5 * iqr_z
    upper_z = q3_z + 1.5 * iqr_z

    filtered_positions = []
    for pos in positions:
        x, y, z = pos
        if (lower_x <= x <= upper_x and
                lower_y <= y <= upper_y and
                lower_z <= z <= upper_z):
            filtered_positions.append(pos)

    return filtered_positions if filtered_positions else positions


def position_in_words(position):
    x, y, z = position

    x_threshold = max(z / 5, 10)
    y_threshold = max(z / 5, 10)

    if x < -x_threshold:
        h_pos = "left"
    elif x > x_threshold:
        h_pos = "right"
    else:
        h_pos = "center"

    if y < -y_threshold:
        v_pos = "bottom"
    elif y > y_threshold:
        v_pos = "top"
    else:
        v_pos = "center"

    if h_pos == "center" and v_pos == "center":
        return "center"
    elif h_pos == "center":
        return v_pos
    elif v_pos == "center":
        return h_pos
    else:
        return f"{v_pos} {h_pos}"


def calc_intersection_over_union(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def calc_size_similarity(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area1 == 0 or area2 == 0:
        return 0

    return min(area1, area2) / max(area1, area2)


def match_objects(objects1, objects2, img_width, img_height):
    if len(objects1) == 0 or len(objects2) == 0:
        return []

    if len(objects1) == 1 and len(objects2) == 1:
        return [(objects1[0], objects2[0])]

    matches = []
    matched_indices2 = set()

    for i, obj1 in enumerate(objects1):
        best_match = None
        best_score = -1
        best_j = -1

        for j, obj2 in enumerate(objects2):
            if j in matched_indices2:
                continue

            center1 = get_center(obj1['bbox'])
            center2 = get_center(obj2['bbox'])

            x1_norm = (center1[0] - img_width / 2) / (img_width / 2)
            y1_norm = (center1[1] - img_height / 2) / (img_height / 2)

            x2_norm = (center2[0] - img_width / 2) / (img_width / 2)
            y2_norm = (center2[1] - img_height / 2) / (img_height / 2)

            y_score = 1 - min(abs(y1_norm - y2_norm), 1.0)

            expected_x_offset = 0.2
            x_diff = x1_norm - x2_norm
            x_score = 1 - min(abs(x_diff - expected_x_offset), 1.0)

            size_score = calc_size_similarity(obj1['bbox'], obj2['bbox'])

            combined_score = 0.5 * y_score + 0.3 * x_score + 0.2 * size_score

            if combined_score > best_score:
                best_score = combined_score
                best_match = obj2
                best_j = j

        if best_score > 0.5 and best_j >= 0:
            matches.append((obj1, best_match))
            matched_indices2.add(best_j)

    return matches


def find_matching_objects(detections1, detections2, img_width, img_height):
    matched_objects = []
    all_matches = []

    detections1_by_class = {}
    for obj in detections1:
        class_name = obj['class']
        if class_name not in detections1_by_class:
            detections1_by_class[class_name] = []
        detections1_by_class[class_name].append(obj)

    detections2_by_class = {}
    for obj in detections2:
        class_name = obj['class']
        if class_name not in detections2_by_class:
            detections2_by_class[class_name] = []
        detections2_by_class[class_name].append(obj)

    for class_name, objects1 in detections1_by_class.items():
        if class_name in detections2_by_class:
            objects2 = detections2_by_class[class_name]

            matches = match_objects(objects1, objects2, img_width, img_height)

            for obj1, obj2 in matches:
                position = calculate_position_from_edges(obj1['bbox'], obj2['bbox'], img_width, img_height)
                position_text = position_in_words(position)

                matched_obj = {
                    'name': class_name,
                    'position': position,
                    'position_in_words': position_text,
                    'confidence': (obj1['confidence'] + obj2['confidence']) / 2,
                    'bbox1': obj1['bbox'],
                    'bbox2': obj2['bbox']
                }

                matched_objects.append(matched_obj)
                all_matches.append((obj1, obj2, position, position_text))

    return matched_objects, all_matches


def draw_triangulation_visualization(frame1, frame2, matches):
    vis_frame1 = frame1.copy()
    vis_frame2 = frame2.copy()

    h1, w1 = vis_frame1.shape[:2]
    h2, w2 = vis_frame2.shape[:2]

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 165, 0),
        (128, 0, 128)
    ]

    for i, (obj1, obj2, position, position_text) in enumerate(matches):
        color = colors[i % len(colors)]

        bbox1 = obj1['bbox']
        bbox2 = obj2['bbox']

        cv2.rectangle(vis_frame1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), color, 2)
        cv2.rectangle(vis_frame2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), color, 2)

        center1 = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
        center2 = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))

        cv2.circle(vis_frame1, center1, 5, color, -1)
        cv2.circle(vis_frame2, center2, 5, color, -1)

        cv2.line(vis_frame1, (w1 // 2, h1 // 2), center1, color, 1)
        cv2.line(vis_frame2, (w2 // 2, h2 // 2), center2, color, 1)

        cv2.line(vis_frame1, (0, center1[1]), (w1, center1[1]), color, 1, cv2.LINE_AA)
        cv2.line(vis_frame2, (0, center2[1]), (w2, center2[1]), color, 1, cv2.LINE_AA)

        label = f"{obj1['class']} ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}cm)"

        text_size1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis_frame1,
                      (bbox1[0], bbox1[1] - 30),
                      (bbox1[0] + text_size1[0], bbox1[1]),
                      (0, 0, 0), -1)
        cv2.putText(vis_frame1, label, (int(bbox1[0]), int(bbox1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        text_size2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis_frame2,
                      (bbox2[0], bbox2[1] - 30),
                      (bbox2[0] + text_size2[0], bbox2[1]),
                      (0, 0, 0), -1)
        cv2.putText(vis_frame2, label, (int(bbox2[0]), int(bbox2[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_frame1, vis_frame2


@app.route('/detect', methods=['POST'])
def detect_objects():
    global last_frame1, last_frame2, last_results1, last_results2, last_matched_objects

    try:
        data = request.json

        frame1 = decode_image(data['frame1'])
        frame2 = decode_image(data['frame2'])

        img_height, img_width = frame1.shape[:2]

        results1 = model(frame1)
        results2 = model(frame2)

        detections1 = []
        detections2 = []

        for det in results1:
            for *box, conf, cls in det.boxes.data.cpu().numpy():
                x1, y1, x2, y2 = [int(x) for x in box]
                detections1.append({
                    'class': det.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2]
                })

        for det in results2:
            for *box, conf, cls in det.boxes.data.cpu().numpy():
                x1, y1, x2, y2 = [int(x) for x in box]
                detections2.append({
                    'class': det.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2]
                })

        matched_objects, raw_matches = find_matching_objects(detections1, detections2, img_width, img_height)

        with frame_lock:
            last_frame1 = frame1
            last_frame2 = frame2
            last_results1 = results1
            last_results2 = results2
            last_matched_objects = raw_matches

        return jsonify({
            'status': 'success',
            'matched_objects': matched_objects
        })

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


def display_frames():
    global last_frame1, last_frame2, last_results1, last_results2, last_matched_objects

    while True:
        with frame_lock:
            if last_frame1 is not None and last_frame2 is not None:
                if last_matched_objects:
                    vis_frame1, vis_frame2 = draw_triangulation_visualization(
                        last_frame1, last_frame2, last_matched_objects
                    )

                    cv2.imshow('Camera 1', vis_frame1)
                    cv2.imshow('Camera 2', vis_frame2)
                else:
                    annotated_frame1 = draw_detections(last_frame1, last_results1)
                    annotated_frame2 = draw_detections(last_frame2, last_results2)

                    cv2.imshow('Camera 1', annotated_frame1)
                    cv2.imshow('Camera 2', annotated_frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)


if __name__ == '__main__':
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
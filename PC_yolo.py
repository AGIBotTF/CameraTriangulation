from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import threading
import time
from ultralytics import YOLO

app = Flask(__name__)

last_frame1 = None
last_frame2 = None
last_results1 = []
last_results2 = []

frame_lock = threading.Lock()

model = YOLO('yolo11n.pt')


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


@app.route('/detect', methods=['POST'])
def detect_objects():
    global last_frame1, last_frame2, last_results1, last_results2

    try:
        data = request.json

        frame1 = decode_image(data['frame1'])
        frame2 = decode_image(data['frame2'])

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

        with frame_lock:
            last_frame1 = frame1
            last_frame2 = frame2
            last_results1 = results1
            last_results2 = results2

        return jsonify({
            'status': 'success',
            'detections_cam1': detections1,
            'detections_cam2': detections2
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def display_frames():
    global last_frame1, last_frame2, last_results1, last_results2

    while True:
        with frame_lock:
            if last_frame1 is not None and last_frame2 is not None:
                annotated_frame1 = draw_detections(last_frame1, last_results1)
                annotated_frame2 = draw_detections(last_frame2, last_results2)

                cv2.imshow('Camera 1 with Detections', annotated_frame1)
                cv2.imshow('Camera 2 with Detections', annotated_frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)


if __name__ == '__main__':
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
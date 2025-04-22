from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import threading
import time

app = Flask(__name__)

last_frame1 = None
last_frame2 = None
frame_lock = threading.Lock()

def decode_image(encoded_image):
    img_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/detect', methods=['POST'])
def detect_objects():
    global last_frame1, last_frame2

    try:
        data = request.json

        frame1 = decode_image(data['frame1'])
        frame2 = decode_image(data['frame2'])

        with frame_lock:
            last_frame1 = frame1
            last_frame2 = frame2

        return jsonify({
            'status': 'success',
            'message': 'Frames received'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def display_frames():
    global last_frame1, last_frame2

    while True:
        with frame_lock:
            if last_frame1 is not None and last_frame2 is not None:
                cv2.imshow('Camera 1', last_frame1)
                cv2.imshow('Camera 2', last_frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

if __name__ == '__main__':
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
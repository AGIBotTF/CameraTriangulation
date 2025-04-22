import cv2
import os

if not os.path.exists('images1'):
    os.makedirs('images1')
if not os.path.exists('images2'):
    os.makedirs('images2')

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

img_counter = 0

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Failed to grab frames from one or both cameras")
        break

    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    key = cv2.waitKey(1)
    if key % 256 == 27:
        print("Escape hit, closing...")
        break
    elif key % 256 == 32:
        img_name1 = f"images1/frame_{img_counter}.png"
        img_name2 = f"images2/frame_{img_counter}.png"
        cv2.imwrite(img_name1, frame1)
        cv2.imwrite(img_name2, frame2)
        print(f"Saved {img_name1} and {img_name2}")
        img_counter += 1

cap1.release()
cap2.release()
cv2.destroyAllWindows()

import cv2
import os
from ultralytics import YOLO

# initialize the detector with weights
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# path to the image folder
img_folder = "./data/input"

for img_name in os.listdir(img_folder):

    # sadece resimleri al
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(img_folder, img_name)

    # read image
    image = cv2.imread(img_path)
    if image is None:
        continue

    # predict
    detections = detector.predict(image)

    # draw detections
    for detection in detections:
        class_ids = detection.boxes.cls

        for i, bbox in enumerate(detection.boxes):
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            class_id = int(class_ids[i])
            class_name = detection.names[class_id]

            cv2.putText(
                image,
                class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    # show each image
    cv2.imshow("Detected", image)

    # 1.5 saniye göster, q ile çık
    if cv2.waitKey(1500) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

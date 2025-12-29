# import libraries
from ultralytics import YOLO
import cv2

# initilaize the detector model
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# video path
video_path = "./data/input/traffic_signs.mp4"

# read the video file
cap = cv2.VideoCapture(video_path)

# check for errors
if not cap.isOpened():
    print("Unable to open the input file")
    exit()

# processing
ret = True

while ret:
    ret, frame = cap.read()
    
    detections = detector(frame)
    
    for detection in detections:
            for bbox in detection.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]  # Get bounding box coordinates
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
            cv2.imshow("Traffic sign detector", frame)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()

# uncomment the following line to detect and save the annotated image without visualizing it
# results = detector(video_path, save=True)
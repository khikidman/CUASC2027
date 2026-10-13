# Make sure to intsall YOLO and OpenCV
from ultralytics import YOLO
import cv2 as cv

model = YOLO("Vision/best.pt").to("mps") # Set to CPU if you have Windows
cap = cv.VideoCapture(0) # Should be the main camera, if you have continuinty cam, it will be on your phone
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    cv.imshow('frame', annotated_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
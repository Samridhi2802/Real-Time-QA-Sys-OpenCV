import cv2
import numpy as np
import imutils
import torch  # Import the PyTorch library
from scipy.spatial import distance as dist
from imutils import perspective
import math

# Load YOLOv5 model (ensure you have the pretrained weights and PyTorch installed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# ShapeDetector and ColorLabeler classes
class ShapeDetector:
    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        
        return shape

class ColorLabeler:
    def label(self, lab, c):
        mask = np.zeros(lab.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean = cv2.mean(lab, mask=mask)[:3]

        if mean[0] < 100:
            return "red"
        elif mean[0] > 100 and mean[0] < 130:
            return "blue"
        elif mean[0] > 50 and mean[0] < 100:
            return "yellow"
        else:
            return "unknown"

# Utility functions
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def angle(matrix):
    z1, z2, z3, z4 = matrix
    x1, y1 = z1
    x2, y2 = z2
    slope = (y1 - y2) / (x1 - x2)
    angle = int(math.degrees(math.atan(slope)))
    tilted = "CW" if y2 > y1 else "CCW"
    return angle, tilted

# Motion detection initialization
firstFrame = None
width = 14  # Scale factor for dimension calculations
sd = ShapeDetector()
cl = ColorLabeler()

# Open video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam")
    exit()

cv2.namedWindow("Combined Object Detection", cv2.WINDOW_NORMAL)

# Function to apply the first detection algorithm (Connected Components)
def get_indexed_image(im):
    _, img_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing_im = cv2.morphologyEx(img_im, cv2.MORPH_CLOSE, kernel)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_im)
    return retval, labels, stats, centroids

# Main loop to process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Use YOLOv5 for object detection
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Get detections in numpy format

    # Draw YOLOv5 detections on the frame
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f'{model.names[int(cls)]}: {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Process for Connected Components Detection (Left Side of Output)
    gray_left = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, labels, stats, centroids = get_indexed_image(gray_left)
    im_contours_belt = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    belt = ((labels >= 1) * 255).astype('uint8')
    belt_contours, _ = cv2.findContours(belt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im_contours_belt, belt_contours, -1, (0, 255, 0), 2)
    num_objects = len(belt_contours)
    cv2.putText(im_contours_belt, f"Objects Detected: {num_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize frames for display
    height = im_contours_belt.shape[0]
    frame_resized = cv2.resize(frame, (im_contours_belt.shape[1], height))

    # Combine both results side by side
    combined_display = np.hstack((im_contours_belt, frame_resized))

    cv2.imshow("Combined Object Detection", combined_display)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

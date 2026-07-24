from ultralytics import YOLO
import cv2
from collections import Counter

# Load a more accurate YOLO model
# Options:
# yolov8n.pt (Fastest)
# yolov8s.pt
# yolov8m.pt (Recommended)
# yolov8l.pt
model = YOLO("yolov8m.pt")

print("YOLO Model Loaded Successfully")

image_path = "test2.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Better detection settings
results = model.predict(
    source=image,
    conf=0.50,      # Ignore detections below 50%
    iou=0.45,       # Helps reduce duplicate boxes
    imgsz=1280,     # Larger input for better small-object detection
    max_det=100,
    verbose=False
)

object_list = []

for result in results:

    boxes = result.boxes

    for box in boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        object_list.append(class_name)

        print(f"{class_name:15} Confidence: {confidence:.2f}")

counts = Counter(object_list)

print("\n==========================")
print("OBJECT COUNT")
print("==========================")

for obj, count in counts.items():
    print(f"{obj:15}: {count}")

annotated = results[0].plot()

cv2.imwrite("detected_output.jpg", annotated)

print("\nOutput image saved as detected_output.jpg")

cv2.imshow("YOLO Object Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
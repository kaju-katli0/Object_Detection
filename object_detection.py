from ultralytics import YOLO
import cv2
import torch
from collections import Counter


# ==========================================
# 1. CHECK GPU
# ==========================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device = 0
else:
    print("GPU not available")
    device = "cpu"


# ==========================================
# 2. LOAD YOLOv8m
# ==========================================

model = YOLO("yolov8m.pt")


# ==========================================
# 3. IMAGE
# ==========================================

image_path = "test2.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        "test.jpg not found. Put test.jpg in the project folder."
    )


# ==========================================
# 4. DETECTION
# ==========================================

results = model.predict(
    source=image,
    imgsz=1280,
    conf=0.10,
    iou=0.50,
    augment=True,
    device=device,
    verbose=True
)


# ==========================================
# 5. PROCESS RESULTS
# ==========================================

result = results[0]

detected_objects = []


print("\n======================================")
print("DETECTED OBJECTS")
print("======================================")


for box in result.boxes:

    class_id = int(box.cls.item())

    confidence = float(box.conf.item())

    object_name = model.names[class_id]

    detected_objects.append(object_name)

    print(
        f"{object_name:<20}"
        f"{confidence * 100:.2f}%"
    )


# ==========================================
# 6. COUNT OBJECTS
# ==========================================

counts = Counter(detected_objects)


print("\n======================================")
print("OBJECT COUNT")
print("======================================")


if counts:

    for object_name, count in counts.items():

        print(
            f"{object_name:<20}: {count}"
        )

else:

    print("No objects detected.")


# ==========================================
# 7. SAVE RESULT
# ==========================================

output_image = result.plot()

cv2.imwrite(
    "detected_output.jpg",
    output_image
)


print("\n======================================")
print("DONE")
print("======================================")

print("Output saved as: detected_output.jpg")


# ==========================================
# 8. DISPLAY RESULT
# ==========================================

print("\nDetection completed successfully.")
print("Open detected_output.jpg from the VS Code Explorer to view the result.")
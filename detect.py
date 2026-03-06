import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

print("✅ Model loaded successfully")
print("Classes:", model.names)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = model.names[cls].lower()

            # Helmet → GREEN
            if class_name == "helmet":
                color = (0, 255, 0)
                label = f"Helmet {conf:.2f}"

            # NoHelmet → RED
            elif class_name == "nohelmet":
                color = (0, 0, 255)
                label = f"NoHelmet {conf:.2f}"

            else:
                continue

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Show frame
    cv2.imshow("Helmet Detection Test", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
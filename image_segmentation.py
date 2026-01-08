from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)[0]

    person_count = 0

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

            for i, cls in enumerate(classes):
                if int(cls) == 0:
                    person_count += 1

                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if result.masks is not None:
                        mask = cv2.resize(
                            masks[i],
                            (frame.shape[1], frame.shape[0])
                        )
                        colored_mask = np.zeros_like(frame)
                        colored_mask[:, :, 2] = (mask * 255).astype(np.uint8)
                        frame = cv2.addWeighted(frame, 1, colored_mask, 0.4, 0)

    cv2.putText(
        frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Room Occupancy - YOLO Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

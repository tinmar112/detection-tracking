import cv2
import torch

VIDEO_PATH = "cars.mp4"
OUTPUT_PATH = "output.mp4"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="yolov5l.pt",
    device=device,
)
model.conf = 0.5
model.iou = 0.45

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results.render()[0]

    writer.write(annotated_frame)
    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved result to {OUTPUT_PATH}")

import cv2
from ultralytics import YOLO

def main():
    print("Loading YOLOv8 Model...")
    model = YOLO("yolov8n.pt")

    print("Starting Live Webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO track
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Live Object Detection & Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

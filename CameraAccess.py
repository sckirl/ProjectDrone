import cv2
from ultralytics import YOLO

class CameraAccess:
    def __init__(self, model, cameraIdx) -> None:
        self.model = model
        self.cameraIdx = cameraIdx

        self.cap = cv2.VideoCapture(self.cameraIdx) 
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def getCameraView(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return ""

        # Run YOLO inference with no verbose logging
        results = self.model.predict(frame, verbose=False)

        # Process detections
        for result in results:
            boxes = result.boxes  # Boxes object
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                print(f"Detected: {label} ({conf:.2f})")
                yield label

        # Optional: show video feed with detections drawn
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            return ""

        self.cap.release()
        cv2.destroyAllWindows()
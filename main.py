import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os

class OptimizedCoinCounter:
    def __init__(self, modelPath='./coinFall150.pt', confThreshold=0.3, counterAreaHeight=100):
        """
        Initializes the optimized coin counter with a top-side counting area.
        
        Args:
            modelPath (str): Path to the YOLO model file.
            confThreshold (float): Confidence threshold for detections.
            counterAreaHeight (int): The height of the counting area from the top of the screen.
        """
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Model file '{modelPath}' not found.")
        
        self.model = YOLO(modelPath)
        self.confThreshold = confThreshold
        
        # Initialize state variables
        self.totalCoinsCounted = 0
        self.countedTrackIds = set()  # Stores track_ids of coins that have been counted
        
        # Counting area (will be adjusted to frame size in the first process call)
        self.counterArea = None
        self.counterAreaYEnd = counterAreaHeight # The line where the top counting area ends
        
        # Class names and colors
        self.classNames = {0: 'COIN', 1: 'FALLING_COIN'}
        self.colors = {
            'COIN': (0, 255, 0),       # Green
            'FALLING_COIN': (0, 0, 255) # Red
        }
        
        print(f"YOLO model loaded from {modelPath}")
        print(f"Counting area is the top {self.counterAreaYEnd} pixels of the screen.")

    def _initializeCounterArea(self, frameWidth):
        """Initializes the counter area coordinates based on the frame size."""
        self.counterArea = {
            "x1": 0,
            "y1": 0,  # Start from the very top
            "x2": frameWidth,
            "y2": self.counterAreaYEnd # End at the specified height
        }
        print(f"Counter area initialized: {self.counterArea}")

    def processFrame(self, frame):
        """
        Detects, tracks, and counts coins within a single frame.
        
        Args:
            frame: The image frame from the camera/video.
            
        Returns:
            The processed frame with visualizations.
        """
        # Initialize the counter area on the first frame
        if self.counterArea is None:
            h, w, _ = frame.shape
            self._initializeCounterArea(w)
        
        # Perform inference with tracking
        results = self.model.track(frame, conf=self.confThreshold, persist=True, verbose=False, tracker="botsort.yaml")
        
        processedFrame = frame.copy()
        
        # Draw the counting area at the top
        self.drawCounterArea(processedFrame)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            trackIds = results[0].boxes.id.cpu().numpy().astype(int)
            classIds = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, trackId, classId in zip(boxes, trackIds, classIds):
                x1, y1, x2, y2 = box
                
                # Using top-center as the reference point is more intuitive for a top-down count
                refPointX = (x1 + x2) // 2
                refPointY = y1
                
                # Counting logic
                if trackId not in self.countedTrackIds:
                    # Check if the coin's reference point is inside the new top counting area
                    if (self.counterArea["x1"] < refPointX < self.counterArea["x2"] and
                        self.counterArea["y1"] < refPointY < self.counterArea["y2"]):
                        
                        self.totalCoinsCounted += 1
                        self.countedTrackIds.add(trackId)
                        print(f"Coin ID:{trackId} counted! Total: {self.totalCoinsCounted}")
                
                # Draw visualizations
                self.drawVisuals(processedFrame, box, trackId, classId)
                
        # Add text information to the frame
        self.addInfoToFrame(processedFrame, len(results[0].boxes) if results[0].boxes.id is not None else 0)
        
        return processedFrame

    def drawVisuals(self, frame, box, trackId, classId):
        """Draws the bounding box and labels on the frame."""
        x1, y1, x2, y2 = box
        className = self.classNames.get(classId, "UNKNOWN")
        color = self.colors.get(className, (255, 255, 255))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{trackId} {className}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def drawCounterArea(self, frame):
        """Draws the counting area on the frame."""
        x1, y1, x2, y2 = self.counterArea.values()
        
        subImage = frame[y1:y2, x1:x2]
        whiteRect = np.ones(subImage.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(subImage, 0.8, whiteRect, 0.2, 1.0)
        
        frame[y1:y2, x1:x2] = res
        cv2.putText(frame, "COUNTING AREA", (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def addInfoToFrame(self, frame, currentCoinCount):
        """Adds counting information to the frame."""
        # Place info box at the bottom to avoid collision with counter area
        info_y = frame.shape[0] - 70
        cv2.rectangle(frame, (5, info_y), (350, info_y + 65), (0, 0, 0), -1)
        
        cv2.putText(frame, f"COINS DETECTED: {currentCoinCount}", (10, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"TOTAL COUNTED: {self.totalCoinsCounted}", (10, info_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    def resetCounter(self):
        """Resets the counter to its initial state."""
        self.totalCoinsCounted = 0
        self.countedTrackIds.clear()
        print("--- COUNTER RESET ---")

def main():
    parser = argparse.ArgumentParser(description='Optimized Coin Counter with YOLO')
    parser.add_argument('--model', type=str, default='coinFall150.pt', help='Path to the .pt model file')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--area_height', type=int, default=150, help='Height of the counting area from the top')
    args = parser.parse_args()
    
    try:
        coinCounter = OptimizedCoinCounter(args.model, args.conf, args.area_height)
    except FileNotFoundError as e:
        print(e)
        return
        
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 50)
    print("AUTOMATIC COIN COUNTER (TOP COUNTING AREA)")
    print("=" * 50)
    print("Logic: A coin is counted +1 when first detected inside")
    print("       the 'COUNTING AREA' at the top.")
    print("-" * 50)
    print("Controls: [q] Quit | [r] Reset | [s] Screenshot")
    print("=" * 50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or frame could not be read.")
            break
        
        if args.source == '0':
            frame = cv2.flip(frame, 1)
            
        processedFrame = coinCounter.processFrame(frame)
        cv2.imshow('Optimized Coin Counter', processedFrame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            coinCounter.resetCounter()
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, processedFrame)
            print(f"Screenshot saved: {filename}")
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished. Total coins counted: {coinCounter.totalCoinsCounted}")

if __name__ == "__main__":
    main()

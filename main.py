from CameraAccess import CameraAccess
import WirelessAccess
import time
import cv2
from ultralytics import YOLO
from YOLOdetection import *
from ultralytics.utils.plotting import Annotator, colors
from FrameDifferencing import *
import DroneAccess

# ---- Setup ----
    
NOTIFY_COUNT = 5
LINE_Y = 600

model = YOLO("MODELS/CFDFalling-120.pt")
drone = DroneAccess.Drone("/dev/tty.usbmodem0x80000001")
object_history = {} 
seenID = set()
sent = True
totalCount = 0
lastCounted = -1 

def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # --- Calculate target dimensions once ---
    ret, firstFrame = cap.read()
    if not ret:
        print("Error: could not read the first frame.")
        cap.release()
        return
        
    originalH, originalW = firstFrame.shape[:2]
    targetW = 640
    targetH = int(targetW * (originalH / originalW))
    targetDims = (targetW, targetH)
    
    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- Initialize with resized frame ---
    ret, frame = cap.read()
    frame = cv2.resize(frame, targetDims, interpolation=cv2.INTER_AREA)
    prevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prevGray = cv2.GaussianBlur(prevGray, (21, 21), 0)
    cfdImage = np.zeros_like(prevGray)

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame = cv2.resize(frame, targetDims, interpolation=cv2.INTER_AREA)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayFrame = cv2.GaussianBlur(grayFrame, (21, 21), 0)

        # --- FIX: Update cfdImage state in each loop for accumulation ---
        cfdImage = getCFD(prevGray, grayFrame, cfdImage)
        
        opticalFlowMask = getOpticalFlow(prevGray, grayFrame)

        # --- Call the Updated Overlay Function ---
        overlaidResult = overlayMotionOnGrayscale(grayFrame, 
                                                      redMask=cfdImage, 
                                                      blueMask=opticalFlowMask)
        
        result = model.predict(overlaidResult, conf=0.6, verbose=False)
        annonated = drawAnnotator(overlaidResult, result[0])

        """if result[0]:
            drone.readGPS()"""

        # --- Display the Results ---
        cv2.imshow('RED = CFD, BLUE = Optical Flow', annonated)

        # Update the previous frame for the next iteration
        prevGray = grayFrame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
from CameraAccess import CameraAccess
import WirelessAccess
import time
import cv2
from ultralytics import YOLO
from YOLOdetection import *
from OpticalFlow import opticalFlowOverlay
from ultralytics.utils.plotting import Annotator, colors

# ---- Setup ----
NOTIFY_COUNT = 5
LINE_Y = 600

# ba = WirelessAccess.Wireless("/dev/cu.usbserial-110", 9600)
model = YOLO("ProjectDrone/coinFall150.pt")
object_history = {} 
seenID = set()
sent = True
totalCount = 0
lastCounted = -1 
prevFrame = None

if __name__ == "__main__":
    # ---- Start continuous tracking (persist keeps IDs) ----
    for result in model.track(source=0, 
                                tracker="botsort.yaml", 
                                persist=True, 
                                stream=True,
                                verbose=False,
                                imgsz=640,
                                conf=0.6):
        
        frame = cv2.resize(result.orig_img, (640,640))
        flowFrame = opticalFlowOverlay(frame, prevFrame)
        prevFrame = frame.copy()
        annonate = drawAnnotator(flowFrame, result)

        # --- Call the counting function ---
        totalCount, object_history, seenID = countLineCrossing(
            annonate, result, LINE_Y, object_history, seenID, totalCount
        )

        # --- Sending Logic ---
        if totalCount > 0 and \
            totalCount % NOTIFY_COUNT == 0 and \
            totalCount != lastCounted:
            print(f"Count is {totalCount}. Sending message")
            time.sleep(0.1)
            lastCounted = totalCount

        cv2.imshow("Tracking", annonate)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    # ba.close()
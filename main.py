from CameraAccess import CameraAccess
import WirelessAccess
import time
import cv2
from ultralytics import YOLO
from YOLOdetection import *
from ultralytics.utils.plotting import Annotator, colors


if __name__ == "__main__":
    # ---- Setup ----
    # ba = WirelessAccess.Wireless("/dev/cu.usbserial-110", 9600)
    LINE_Y = 600
    NOTIFY_COUNT = 5

    model = YOLO("ProjectDrone/coinModel.pt")
    object_history = {} 
    seenID = set()
    sent = True
    totalCount = 0
    lastCounted = -1 

    # ---- Start continuous tracking (persist keeps IDs) ----
    for result in model.track(source=0, 
                                tracker="botsort.yaml", 
                                persist=True, 
                                stream=True, 
                                classes=[1,2,3],
                                verbose=False,
                                conf=0.4,
                                imgsz=640):
        frame = result.orig_img
        frame = drawAnnotator(frame, result)

        # --- Call the counting function ---
        totalCount, object_history, seenID = countLineCrossing(
            frame, result, LINE_Y, object_history, seenID, totalCount
        )

        # --- Sending Logic ---
        if totalCount > 0 and totalCount % NOTIFY_COUNT == 0 and totalCount != lastCounted:
            print(f"Count is {totalCount}. Sending message")
            time.sleep(0.1)
            lastCounted = totalCount

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    # ba.close()
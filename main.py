from CameraAccess import CameraAccess
import WirelessAccess
import time
import cv2
from ultralytics import YOLO

if __name__ == "__main__":
    # Setup
    ba = WirelessAccess.Wireless("/dev/cu.usbserial-110", 9600)
    model = YOLO("coinModel.pt")

    seenID = set()
    sent = True

    # Start continuous tracking (persist keeps IDs)
    for result in model.track(source=0, 
                                tracker="botsort.yaml", 
                                persist=True, 
                                stream=True, 
                                classes=[1,2,3],
                                verbose=False,
                                conf=0.4):
        frame = result.plot()

        if hasattr(result, 'boxes') and result.boxes.id is not None:
            for objID in result.boxes.id:
                objID = int(objID.item())
                if objID not in seenID:
                    seenID.add(objID)

                    print(f"New object detected: ID {objID}, Total count: {len(seenID)}")
                    sent = False
                
            for leaveID in seenID.difference(set(result.boxes.id.cpu().numpy())):
                seenID.remove(leaveID)

        # when the count reaches something, returns
        if len(seenID) % 5 == 0 & sent is not True:
            ba.sendMessage("scissor")
            print(f"should be sent {len(seenID)}")
            time.sleep(0.1)
            ba.sendMessage("test")
            sent = True

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    ba.close()
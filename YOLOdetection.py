

class YOLOdetection():
    def __init__(self) -> None: 
        self.seenID = set()

    def YOLOCounter(self, detection):
        if hasattr(detection, 'boxes') and detection.boxes.id is not None:
            for objID in detection.boxes.id:
                objID = int(objID.item())
                if objID not in self.seenID:
                    self.seenID.add(objID)

                    sent = False
                    print(f"New object detected: ID {objID}, Total count: {len(self.seenID)}")
                
            for leaveID in self.seenID.difference(set(detection.boxes.id.cpu().numpy())):
                self.seenID.remove(leaveID)
            
        yield self.seenID
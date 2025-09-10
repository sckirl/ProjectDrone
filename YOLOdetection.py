import cv2
from ultralytics.utils.plotting import Annotator, colors

def drawAnnotator(frame, results):
    annotator = Annotator(frame, line_width=2)

    # If results is a list, iterate
    if isinstance(results, list):
        results = results[0]  # take first batch

    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes.xyxy.cpu()
        clss = results.boxes.cls.cpu().tolist()
        ids = results.boxes.id.cpu().tolist() if results.boxes.id is not None else [None] * len(boxes)

        for box, cls, obj_id in zip(boxes, clss, ids):
            label = f"ID:{int(obj_id)}" if obj_id is not None else f"cls:{int(cls)}"
            annotator.box_label(box, label, color=colors(int(cls), True))

    return annotator.result()

def countLineCrossing(frame, result, line_y, object_history, seenID, total_count):
    # Draw the horizontal counting line
    height, width, _ = frame.shape
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

    if hasattr(result, 'boxes') and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            # Calculate the center point of the bounding box
            center_y = int((box[1] + box[3]) / 2)
            
            # Check if we have a previous position for this object
            if obj_id in object_history:
                prev_y = object_history[obj_id]
                
                # Check for a downward cross (was above, now is below)
                if prev_y < line_y and center_y >= line_y and obj_id not in seenID:
                    total_count += 1
                    seenID.add(obj_id)
                    print(f"Object ID {int(obj_id)} crossed! Total count: {total_count}")
            
            # Update the object's last known position
            object_history[obj_id] = center_y

    # Display the total count on the frame
    cv2.putText(frame, f"Count: {total_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return total_count, object_history, seenID


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
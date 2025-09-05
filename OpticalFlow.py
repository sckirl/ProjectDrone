import cv2
import numpy as np
import os

def opticalFlow(current_frame, prev_frame, resize_width=640):
    # If there is no previous frame, we can't compute flow.
    if prev_frame is None:
        return current_frame
    
    # ---- Downscale to 640x640 to fit the image into YOLO dataset ----
    aspect_ratio = current_frame.shape[0] / current_frame.shape[1]
    resize_height = int(resize_width * aspect_ratio)

    prev_small = cv2.resize(prev_frame, (resize_width, resize_height))
    current_small = cv2.resize(current_frame, (resize_width, resize_height))

    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)

    # --- Optimization: Tuned Farneback parameters for accuracy ---
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=current_gray,
        flow=None,
        pyr_scale=0.5,
        levels=5,      
        winsize=25,
        iterations=3,  
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # Convert flow to polar coordinates to get magnitude (speed) and angle (direction)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # IMPROTANT: FALLING (vertical flow > 1) and FAST (speed > 5)
    mask = (flow[..., 1] > 4) & (magnitude > 10)

    # Clean up the mask using morphological opening to remove small noise specks
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Apply the CLEANED mask to the small grayscale frame
    isolated_object_gray = cv2.bitwise_and(current_gray, current_gray, mask=mask_cleaned)

    return isolated_object_gray

def extractFrames(frame, outputDir, count):
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # Construct filename for the frame
        frame_filename = os.path.join(outputDir, f"frame_{count:05d}.jpg")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)


frameCount = 0
# --- Main execution block for the demo ---
if __name__ == "__main__":
    cap = cv2.VideoCapture("ProjectDrone/IMG_6334.mov")
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = opticalFlow(frame, prev_frame)
        
        # Update the previous frame for the next iteration
        prev_frame = frame.copy()

        # Display the single, combined output frame
        cv2.imshow("Optical Flow Overlay", output_frame)
        # extractFrames(output_frame, "fallingDataset", frameCount)

        frameCount += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
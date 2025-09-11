import cv2
import numpy as np
import os

def overlayMotionOnGrayscale(grayFrame, redMask=None, blueMask=None, weight=0.6):
    """
    Overlays motion masks as semi-transparent heatmaps on a grayscale frame.
    Red channel is used for the first mask (e.g., CFD).
    Blue channel is used for the second mask (e.g., Optical Flow).

    Args:
        grayFrame (np.ndarray): The single-channel grayscale video frame.
        redMask (np.ndarray, optional): Motion mask for the red channel.
        blueMask (np.ndarray, optional): Motion mask for the blue channel.
        weight (float): The transparency of the motion overlay (0.0 to 1.0).

    Returns:
        np.ndarray: The BGR image with the motion data overlaid.
    """
    # Create a color heatmap from the motion masks
    heatmap = np.zeros((grayFrame.shape[0], grayFrame.shape[1], 3), dtype=np.uint8)
    if redMask is not None:
        heatmap[:, :, 2] = redMask  # Assign redMask to the Red channel
    if blueMask is not None:
        heatmap[:, :, 0] = blueMask # Assign blueMask to the Blue channel

    # Convert the single-channel grayscale frame to a 3-channel BGR image
    bgrFrame = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)

    # Return the blended image
    return cv2.addWeighted(bgrFrame, 1, heatmap, weight, 0)

def getCFD(prevGray, grayFrame, cfdImage):
    """Calculates and accumulates the frame difference."""
    frameDelta = cv2.absdiff(prevGray, grayFrame)
    thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
    # Add the new motion and apply decay
    newCfdImage = cv2.add(cfdImage, thresh)
    newCfdImage = (newCfdImage * 0.85).astype("uint8")
    return newCfdImage
    
def getOpticalFlow(prevGray, grayFrame, speed_threshold=5.0):
    flow = cv2.calcOpticalFlowFarneback(
        prev=prevGray,
        next=grayFrame,
        flow=None,
        pyr_scale=0.5,
        levels=5,
        winsize=25,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # --- FAST filter ---
    speed_threshold = 4.0  # tune this experimentally
    fast_mask = (magnitude > speed_threshold).astype(np.uint8) * 255

    # --- FALLING filter (downward motion ≈ 90° ± 30°) ---
    down_angle_min = np.pi/3   # 60°
    down_angle_max = 2*np.pi/3 # 120°
    falling_mask = ((angle > down_angle_min) & (angle < down_angle_max)).astype(np.uint8) * 255

    # --- Combine ---
    falling_fast_mask = cv2.bitwise_and(fast_mask, falling_mask)

    return falling_fast_mask

def extractFrames(datasetDirectory, frame, count):
    os.makedirs(datasetDirectory, exist_ok=True)

    frameDir = os.path.join(datasetDirectory, f"{count}_Frame.png")

    cv2.imwrite(frameDir, frame)
    print(f"Made: {frameDir}")

def main():
    """
    Main function to demonstrate CFD (Red) and Optical Flow (Blue)
    overlay on a video file, resized for YOLO.
    """
    cap = cv2.VideoCapture("Videos/Dataset60_4.mov")
    DATASET_DIRECTORY = "DATASET_CFD"
    COUNT = 120

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

        # --- Display the Results ---
        cv2.imshow('RED = CFD, BLUE = Optical Flow', overlaidResult)
        extractFrames(DATASET_DIRECTORY, overlaidResult, COUNT)
        COUNT += 1

        # Update the previous frame for the next iteration
        prevGray = grayFrame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

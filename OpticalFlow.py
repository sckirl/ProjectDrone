import cv2
import numpy as np
import os

def opticalFlow(current_frame, prev_frame, resize_width=640):
    # If there is no previous frame, we can't compute flow.
    if prev_frame is None:
        return current_frame

    # --- Optimization 1: Downscale frames for faster processing ---
    # Calculate aspect ratio to resize height properly
    aspect_ratio = current_frame.shape[0] / current_frame.shape[1]
    resize_height = int(resize_width * aspect_ratio)
    
    # Create smaller versions of the frames
    prev_small = cv2.resize(prev_frame, (resize_width, resize_height))
    current_small = cv2.resize(current_frame, (resize_width, resize_height))

    # Convert small frames to grayscale
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)

    # --- Optimization 2: Tuned Farneback parameters for speed ---
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
    # You can adjust these threshold values of 1 and 2.
    mask = (flow[..., 1] > 1) & (magnitude > 2)

    # Create a black HSV image to draw on
    hsv_mask = np.zeros_like(prev_small, dtype=np.uint8)

    # Where the mask is True, apply the color based on flow
    hsv_mask[mask, 1] = 255  # Full Saturation
    hsv_mask[mask, 0] = angle[mask] * 180 / np.pi / 2  # Hue from direction
    norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv_mask[mask, 2] = norm_magnitude[mask] # Value from speed

    # Convert the HSV visualization to BGR
    bgr_flow = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # --- Upscale the flow visualization to match the original frame size ---
    original_dims = (current_frame.shape[1], current_frame.shape[0])
    bgr_flow_upscaled = cv2.resize(bgr_flow, original_dims)

    return bgr_flow_upscaled 

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
    cap = cv2.VideoCapture("/Users/alvin/Library/CloudStorage/GoogleDrive-alvin.setiawan1010@gmail.com/My Drive/pythonProjects-mac/Dronee/ProjectDrone/IMG_6335.mov")
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
        extractFrames(output_frame, "fallingDataset", frameCount)

        frameCount += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
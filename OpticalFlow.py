import cv2
import numpy as np

def opticalFlowOverlay(current_frame, prev_frame, resize_width=480):
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
        levels=3,      # Reduced pyramid levels
        winsize=15,
        iterations=2,  # Reduced iterations
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )

    # Convert flow to polar coordinates to get magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image for visualization
    hsv_mask = np.zeros_like(prev_small, dtype=np.uint8)
    hsv_mask[..., 1] = 255 # Max saturation
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the HSV visualization to BGR
    bgr_flow = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # --- Upscale the flow visualization to match the original frame size ---
    original_dims = (current_frame.shape[1], current_frame.shape[0])
    bgr_flow_upscaled = cv2.resize(bgr_flow, original_dims)

    # --- Overlay: Blend the original frame with the flow visualization ---
    # cv2.addWeighted calculates: dst = src1*alpha + src2*beta + gamma
    overlay_frame = cv2.addWeighted(
        src1=current_frame, 
        alpha=0.7, # Weight of the original frame (mostly opaque)
        src2=bgr_flow_upscaled, 
        beta=0.8,  # Weight of the flow visualization (semi-transparent)
        gamma=0
    )

    return overlay_frame

# --- Main execution block for the demo ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = opticalFlowOverlay(frame, prev_frame)
        
        # Update the previous frame for the next iteration
        prev_frame = frame.copy()

        # Display the single, combined output frame
        cv2.imshow("Optical Flow Overlay", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
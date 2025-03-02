import cv2
import os
import numpy as np

# Input and output directories
input_dir = "results_avi_2/"
output_dir = "results_avi_2/morph"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

# Define the morphological kernel (adjust size if needed)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Loop through all AVI files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".avi") and not filename.endswith("_morph.avi"):  # Avoid reprocessing output files
        video_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".avi", "_morph_2.avi"))

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unreadable

        # Debugging output
        print(f"\nProcessing: {filename}")
        print(f"Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}")

        # Define video writer (ensure grayscale format compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale if not already
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Apply morphological opening to remove noise
            cleaned_mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

            # Convert back to 3-channel grayscale for better compatibility
            cleaned_mask = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

            # Write the processed frame
            out.write(cleaned_mask)

            """# Show output (optional)
            cv2.imshow("Cleaned Mask", cleaned_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break"""

        # Release resources
        cap.release()
        out.release()
        print(f"Saved: {output_path}")

# Cleanup
cv2.destroyAllWindows()
print("\nAll videos processed successfully!")

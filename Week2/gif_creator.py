import cv2
import imageio

def create_gif_from_video(video_path, output_gif_path, start_frame, end_frame, scale_factor=0.5, fps=10):
    """
    Extracts frames from an AVI video, resizes them, and saves them as a GIF.
    
    Parameters:
        video_path (str): Path to the input video file.
        output_gif_path (str): Path to save the output GIF.
        start_frame (int): The first frame to include in the GIF.
        end_frame (int): The last frame to include in the GIF.
        scale_factor (float): Factor to resize the frames (e.g., 0.5 for half size).
        fps (int): Frames per second for the GIF.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if start_frame <= frame_count <= end_frame:
            # Resize the frame to reduce resolution
            if scale_factor != 1.0:
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert BGR (OpenCV format) to RGB (for GIF)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        if frame_count > end_frame:
            break  # Stop when reaching the end frame
        
        frame_count += 1

    cap.release()
    
    if frames:
        imageio.mimsave(output_gif_path, frames, fps=fps)
        print(f"GIF saved successfully: {output_gif_path}")
    else:
        print("No frames were extracted for the GIF.")

# Example Usage
video_path = "../det.avi"
output_gif_path = "../det_gif.gif"
start_frame = 428   # Change this to your desired start frame
end_frame = 570     # Change this to your desired end frame
scale_factor = 0.5

create_gif_from_video(video_path, output_gif_path, start_frame, end_frame, scale_factor)

import cv2
import os
import re
import pytesseract
import numpy as np
from tqdm import tqdm

# Define the Region of Interest (ROI) for the text overlay
# [0, 0, 1, 1] == top left pxel of frame
# Replace these values with the actual coordinates of your text overlay

ROI = (0, 0, 426, 240) # Format: (x, y, width, height)

# Define Paths and File Names:
video_path  = "relative/path/to/my_video.mp4"
output_path = "relative/path/to/extracted_data/"
output_file = "video_extracted_text.txt"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory exists: {directory}")

def analyze_text(extracted_text):
    """
    Post Process the extracted text here.
    """
    return extracted_text

def extract_text_from_video(video_path, output_path, output_file):
    # Ensure the output directory exists
    ensure_dir(output_path)
    
    # Construct the full path for the output file
    full_output_path = os.path.join(output_path, output_file)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    order_frame_num = len(str(total_frames))
    
    # Try to read and save the first frame
    ret, frame = video.read()
    frame_number = 1
    
    # If first frame fails, try the second frame
    if not ret:
        ret, frame = video.read()
        frame_number = 2
    
    if ret:
        # Save the full frame
        frame_filename = os.path.join(output_path, f'frame-{frame_number:0{order_frame_num}d}.jpeg')
        cv2.imwrite(frame_filename, frame)
        print(f"Frame {frame_number} saved as '{frame_filename}'")
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Define ROI as bottom 200 pixels
        ROI = (0, height - 200, width, 200)
        print(f"ROI set to: {ROI}")
        
        # Extract and save the ROI portion
        x, y, w, h = ROI
        roi = frame[y:y+h, x:x+w]
        roi_filename = os.path.join(output_path, f'frame-{frame_number:0{order_frame_num}d}-roi.jpeg')
        cv2.imwrite(roi_filename, roi)
        print(f"ROI portion of frame {frame_number} saved as '{roi_filename}'")
    else:
        print("Failed to capture both first and second frames")
        return
    
    # Reset video capture to the beginning
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    extracted_text = []
   
    # Iterate through each frame
    for frame_num in tqdm(range(total_frames), desc="Processing frames"):
    # for frame_num in tqdm(range(min(100, total_frames)), desc="Processing frames"):
        # Read the frame
        ret, frame = video.read()
        if not ret:
            break
        
        # Extract the ROI from the frame
        x, y, w, h = ROI
        roi = frame[y:y+h, x:x+w]
        
        # Convert ROI to black and white
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Tesseract to do OCR on the binary image
        text = pytesseract.image_to_string(binary).strip()
        
        # Only add to the list if text was found
        if text:
            extracted_text.append((frame_num, text))
    
    # Release the video capture object
    video.release()
    
    print(f"Text extraction complete. {len(extracted_text)} frames contained text.")
    
    # Analyze the extracted text
    analysis_results = analyze_text(extracted_text)
    
    # Save analysis results
    analysis_file = os.path.join(output_path, "text_analysis_results.txt")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        for result in analysis_results:
            f.write(f"{result}\n")
    
    print(f"Text analysis complete. Results saved to {analysis_file}")
    
    return extracted_text, analysis_results
    
    print(f"Text extraction complete. Results saved to {full_output_path}")

extract_text_from_video(video_path, output_path, output_file)


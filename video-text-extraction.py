import cv2
import pytesseract
from tqdm import tqdm

# Define the Region of Interest (ROI) for the text overlay
# [0, 0, 1, 1] == top left pxel of frame
# Format: (x, y, width, height)
# Replace these values with the actual coordinates of your text overlay

ROI = (100, 50, 500, 100)
# Example usage
video_path = "path/to/your/video.mp4"
output_file = "extracted_text.txt"
extract_text_from_video(video_path, output_file)

def extract_text_from_video(video_path, output_file):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Open the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate through each frame
        for frame_num in tqdm(range(total_frames), desc="Processing frames"):
            # Read the frame
            ret, frame = video.read()
            if not ret:
                break
            
            # Extract the ROI from the frame
            x, y, w, h = ROI
            roi = frame[y:y+h, x:x+w]
            
            # Convert ROI to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Use Tesseract to do OCR on the ROI
            text = pytesseract.image_to_string(gray)
            
            # Write the extracted text to the file, along with the frame number
            f.write(f"Frame {frame_num} (Time: {frame_num/fps:.2f}s):\n{text}\n\n")
    
    # Release the video capture object
    video.release()
    
    print(f"Text extraction complete. Results saved to {output_file}")



"""
Real-time Motion Detection using OpenCV with Optimized Frame Difference
This program captures video from your webcam and detects moving objects.
Uses frame difference method - faster and more responsive.
"""

import cv2
import numpy as np

# Initialize webcam capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize previous frame as None (used for frame difference)
previous_frame = None

# Minimum area threshold to filter out noise (in pixels)
# Only contours larger than this will be considered as motion
MIN_AREA = 500

print("Motion Detection started. Press 'q' to quit...")
print("Green rectangles show detected motion.")

# Main loop
while True:
    # Read current frame from webcam
    ret, frame = cap.read()
    
    # Check if frame was read successfully
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Flip the frame horizontally to fix mirror effect (1 = horizontal flip)
    frame = cv2.flip(frame, 1)
    
    # Step 1: Convert frame to grayscale (easier to process than color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur to reduce noise (optimized kernel size)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Step 3: Calculate frame difference if we have a previous frame
    if previous_frame is None:
        previous_frame = blurred
        continue
    
    # Find the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(previous_frame, blurred)
    
    # Step 4: Apply thresholding with lower value for better sensitivity
    # Optimized threshold: 20 (lower = more sensitive to motion)
    _, threshold = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
    
    # Step 4: Dilate the threshold image to fill small holes and connect nearby regions
    # Optimized with more aggressive dilation for better detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(threshold, kernel, iterations=3)
    
    # Apply erosion to remove small noise
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Step 5: Find contours (outlines of moving objects)
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Draw rectangles around detected motion
    motion_detected = False
    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)
        
        # Only process contours larger than MIN_AREA to filter noise
        if area > MIN_AREA:
            motion_detected = True
            # Get bounding rectangle for this contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw green rectangle around the motion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display status text
    status_text = "MOTION DETECTED!" if motion_detected else "No motion"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)  # Red if motion, green if not
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2)
    
    # Display the frame with motion detection results
    cv2.imshow('Motion Detection', frame)
    
    # Update previous frame for next iteration
    previous_frame = blurred
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting motion detection...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Motion detection stopped.")

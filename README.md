# Real-Time Motion Detection

A simple real-time motion detection project built with Python and OpenCV.  
The program captures video from a webcam and detects moving objects by comparing consecutive frames.

## Overview

This project implements motion detection using the frame differencing method.  
It highlights moving areas in the video by drawing bounding boxes around detected regions.

The main goal is to understand the fundamentals of computer vision and real-time video processing.

## How It Works

The system processes each frame as follows:

- Captures video from the webcam  
- Converts frames to grayscale  
- Applies Gaussian blur to reduce noise  
- Computes the difference between consecutive frames  
- Applies thresholding to isolate motion  
- Uses dilation and erosion to clean the result  
- Detects contours and filters out small movements  
- Draws rectangles around moving objects  

## Features

- Real-time motion detection  
- Noise reduction using blur and morphological operations  
- Adjustable sensitivity using minimum area threshold  
- Simple and efficient implementation  

## Technologies Used

- Python  
- OpenCV  

## How to Run

1. Install dependencies:

python3 -m pip install opencv-python

2. Run the program:

python3 main.py

## Usage

- The webcam will open automatically  
- Moving objects will be highlighted with green rectangles  
- A status message shows whether motion is detected  
- Press 'q' to exit the program  

## Limitations

- Sensitive to sudden lighting changes  
- Camera movement can cause false detections  
- Designed for simple environments  

## Future Improvements

- Implement background subtraction (MOG2)  
- Add object tracking  
- Improve robustness against lighting changes  

---

This project was developed as a beginner-level computer vision exercise.

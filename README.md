# POC_Goatee
This project is a computer vision tool that automatically draws a beard-like contour on human faces in images.

# Adaptive Chin Contour Generator

**Author:** SARATH

## Project Description

This project is a computer vision tool that automatically draws a beard-like contour on human faces in images. The key feature of this tool is its **adaptive algorithm**: the contour is not a fixed shape but is procedurally generated based on the specific facial proportions of each individual, specifically the distance between the mouth and the chin.

The script processes a directory of images, detects faces using Google's MediaPipe library, and saves the modified images with the drawn contour to an output directory.

## Features

-   **Batch Processing**: Processes all images within a specified input folder.
-   **Robust Face Detection**: Uses MediaPipe Face Mesh to detect 468 facial landmarks with high accuracy.
-   **Adaptive Contour Generation**: The chin contour's size and shape are dynamically calculated to fit each unique face.
-   **Tunable Parameters**: The contour shape can be fine-tuned by adjusting scale factors within the code.
-   **Error Handling**: Gracefully skips images where no face can be detected.

## Requirements

-   Python 3.10.0
-   OpenCV (`opencv-python`)
-   MediaPipe (`mediapipe`)
-   NumPy (`numpy`)

You can install the required libraries using pip:
```bash
pip install opencv-python mediapipe numpy

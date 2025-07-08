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

-   Python 3.x
-   OpenCV (`opencv-python`)
-   MediaPipe (`mediapipe`)
-   NumPy (`numpy`)

You can install the required libraries using pip:
```bash
pip install opencv-python mediapipe numpy
```

## How to Use

1.  **Prepare Folders**: Place all the images you want to process into an input folder (e.g., `C:\Your\Path\To\imgs`).
2.  **Configure Paths**: Open the Python script and update the `input_folder` and `output_folder` variables in the `main()` function to match your system's paths.
3.  **Run the Script**: Execute the script from your terminal.
    ```bash
    python your_script_name.py
    ```
4.  **Check Results**: The processed images with the drawn contours will be saved in the specified output folder.

## Pipeline Flowchart

This flowchart illustrates the step-by-step process the script follows for each image.

```mermaid
graph TD
    A[Start] --> B[Initialize: Set Paths & Create Contour Object];
    B --> C{Loop through images in Input Folder};
    C --> D[Read Image File];
    D --> E[Detect Facial Landmarks with MediaPipe];
    E --> F{Face Detected?};
    F -- Yes --> G[Calculate Adaptive Contour Parameters <br/>(Width, Height, Center)];
    G --> H[Generate Elliptical Chin Curve Points];
    H --> I[Draw Contour on Image Copy];
    I --> J[Save Processed Image to Output Folder];
    J --> C;
    F -- No --> K[Print Warning & Skip Image];
    K --> C;
    C -- No more images --> L[End];
```

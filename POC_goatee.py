""" Author: SARATH
 Description: This script detects facial landmarks in images from a specified folder
 and draws an adaptive chin contour based on facial proportions (mouth to chin).
 The final images are saved to a results folder.
"""


import cv2
import mediapipe as mp
import numpy as np
import os

def detect_facial_landmarks(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
        
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        refine_landmarks=True
    )

    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    face_mesh.close()

    if not results.multi_face_landmarks:
        print("Warning: No face detected in the image.")
        return image, None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

    return image, landmarks


class ChinContour:
    def __init__(self, width_scale=1.0, height_scale=1.0, y_offset_scale=0.9):
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.y_offset_scale = y_offset_scale

    def draw(self, image, landmarks):
        if not landmarks:
            return image

        mouth_left_corner = np.array(landmarks[61])
        mouth_right_corner = np.array(landmarks[291])
        bottom_lip_center = np.array(landmarks[17])
        chin_tip = np.array(landmarks[152])

        mouth_width = np.linalg.norm(mouth_right_corner - mouth_left_corner)
        mouth_chin_distance = np.linalg.norm(chin_tip - bottom_lip_center)

        ellipse_center_x = int((mouth_left_corner[0] + mouth_right_corner[0]) / 2)
        ellipse_center_y = int(bottom_lip_center[1] + (mouth_chin_distance / 2) * self.y_offset_scale)

        ellipse_axis_x = int((mouth_width / 2) * self.width_scale)
        ellipse_axis_y = int((mouth_chin_distance / 2) * self.height_scale)

        chin_curve_points = cv2.ellipse2Poly(
            (ellipse_center_x, ellipse_center_y),
            (ellipse_axis_x, ellipse_axis_y),
            0, 0, 180, 15
        )

        chin_curve_points = np.flip(chin_curve_points, axis=0)
        full_contour = np.concatenate(([mouth_left_corner], chin_curve_points, [mouth_right_corner]))
        full_contour = np.array(full_contour, dtype=np.int32)

        cv2.polylines(image, [full_contour], isClosed=False, color=(0, 0, 225), thickness=5, lineType=cv2.LINE_AA)

        return image


def main():
    # Set input and output directories
    input_folder = r"C:\Users\sarat\Computer_vision\Gen_AI\imgs"
    output_folder = r"C:\Users\sarat\Computer_vision\Gen_AI\results"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Instantiate contour class with your desired tuning
    chin_style = ChinContour(width_scale=1.05, height_scale=1.1, y_offset_scale=0.95)

    # Loop through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image: {image_path}")

            original_image, landmarks = detect_facial_landmarks(image_path)

            if original_image is None or landmarks is None:
                print(f"Skipping {filename} due to missing image or landmarks.\n")
                continue

            image_with_contour = original_image.copy()
            chin_style.draw(image_with_contour, landmarks)

            output_path = os.path.join(output_folder, f"contour_{filename}")
            cv2.imwrite(output_path, image_with_contour)
            print(f" - Saved to: {output_path}\n")

    print("<----- All images processed. Check the 'results' folder for output------->")


if __name__ == "__main__":
    main()

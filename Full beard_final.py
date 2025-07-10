"""This is only for Full Beard Style"""


import cv2
import mediapipe as mp
import numpy as np
import os
import time
from glob import glob

class FaceOutlineDrawer:
    """A class to draw the outer contour of the face (jawline and cheeks)."""
    
    """This the points for the Jawline"""
    OUTLINE_INDICES = [
        # Left Jaw
        93,132,58, 172, 136, 150, 149, 176, 148,
        # Chin
        152,
        # Right Jaw
        377, 400, 378, 379, 365, 397, 288,401,323
    ]

    def __init__(self, color=(0, 0, 255), thickness=2):
        self.color = color
        self.thickness = thickness

    def draw(self, image, face_landmarks):
        """Draws the face outline on the image."""
        ih, iw, _ = image.shape
        points = []
        for idx in self.OUTLINE_INDICES:
            point = face_landmarks.landmark[idx]
            x, y = int(point.x * iw), int(point.y * ih)
            points.append((x, y))
        points_np = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points_np], isClosed=False, color=self.color, thickness=self.thickness)


class CheekContourDrawer:
    """A class to draw specific curved and straight lines on the cheeks."""
    
    # Left cheek line now starts at landmark 323 (the jaw intersection point).
    LEFT_CHEEK_INDICES = [323,376, 411, 427, 436, 432] 
    
    # Right cheek line now starts at landmark 93 (the jaw intersection point).
    RIGHT_CHEEK_INDICES = [93, 147, 187, 207, 216, 212]

    def __init__(self, color=(0, 0, 255), thickness=2):
        self.color = color
        self.thickness = thickness

    def draw(self, image, face_landmarks):
        """Draws the cheek contours on the image."""
        ih, iw, _ = image.shape
        
        # --- Draw Left Cheek Line ---
        left_points = []
        for idx in self.LEFT_CHEEK_INDICES:
            point = face_landmarks.landmark[idx]
            x, y = int(point.x * iw), int(point.y * ih)
            left_points.append((x, y))
        left_points_np = np.array(left_points, dtype=np.int32)
        cv2.polylines(image, [left_points_np], isClosed=False, color=self.color, thickness=self.thickness)

        # --- Draw Right Cheek Line ---
        right_points = []
        for idx in self.RIGHT_CHEEK_INDICES:
            point = face_landmarks.landmark[idx]
            x, y = int(point.x * iw), int(point.y * ih)
            right_points.append((x, y))
        right_points_np = np.array(right_points, dtype=np.int32)
        cv2.polylines(image, [right_points_np], isClosed=False, color=self.color, thickness=self.thickness)


def draw_custom_features(image, face_landmarks, outline_drawer, cheek_drawer):
    """A helper function to orchestrate drawing, usable by both processors."""
    annotated = image.copy()
    outline_drawer.draw(annotated, face_landmarks)
    cheek_drawer.draw(annotated, face_landmarks)
    return annotated

class ImageFileProcessor:
    """Processes all images in a folder and saves the output."""
    
    def __init__(self, image_folder='images', output_folder='results'):
        self.image_folder = image_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, # Optimized for single images
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

        self.outline_drawer = FaceOutlineDrawer(color=(0, 255, 0), thickness=2)
        self.cheek_drawer = CheekContourDrawer(color=(0, 255, 0), thickness=2)

    def process_images(self):
        """Finds all images in the folder and processes them."""
        image_paths = glob(os.path.join(self.image_folder, '*.*'))

        for file_path in image_paths:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not read: {file_path}")
                continue
            
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print(f"No face found in {file_path}")
                continue

            annotated_image = draw_custom_features(image, results.multi_face_landmarks[0], self.outline_drawer, self.cheek_drawer)
            
            filename = os.path.basename(file_path)
            output_path = os.path.join(
                self.output_folder, f"{os.path.splitext(filename)[0]}_Full_Beard_preview.png")
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved to {output_path}")

    def close(self):
        """Closes the FaceMesh model."""
        self.face_mesh.close()

class WebcamProcessor:
    """Processes a live webcam feed and displays the output."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, # Optimized for video streams
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.outline_drawer = FaceOutlineDrawer(color=(0, 255, 0), thickness=2)
        self.cheek_drawer = CheekContourDrawer(color=(0, 255, 0), thickness=2)

        self.prev_time = 0  # [FPS MOD] Initialize previous time
    
    def run(self):
        """Starts the webcam and processing loop."""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # To improve performance, optionally mark the image as not writeable
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            image.flags.writeable = True

            # Draw the annotations on the image
            if results.multi_face_landmarks:
                annotated_image = draw_custom_features(image, results.multi_face_landmarks[0], self.outline_drawer, self.cheek_drawer)
            else:
                annotated_image = image # If no face, show the original frame

            # [FPS MOD] Calculate and display FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = curr_time
            cv2.putText(annotated_image, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Live Full Beard Preview', annotated_image)
            
            # Exit loop if 'ESC' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        # Release resources
        self.face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Automatically process the image folder
    image_processor = ImageFileProcessor(
        image_folder=r"C:\Users\sarat\Computer_vision\Gen_AI\imgs", 
        output_folder=r"C:\Users\sarat\Computer_vision\Gen_AI\Full_Beard_Final_Results"
    )
    image_processor.process_images()
    image_processor.close()
    print("Image processing complete.")

    # Then run the live webcam processor
    # webcam_processor = WebcamProcessor()
    # webcam_processor.run()
    # webcam_processor.close()
    # print("Webcam processing stopped.")



if __name__ == "__main__":
    main()

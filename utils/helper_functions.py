"""
EECS 6692: Deep Learning on the Edge
File contains functions to display images and annotations
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def display_image(image_path):
    # Read the image from the specified path
    print(image_path)
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()

def display_annotated_image(image):
    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image using Matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()


# source: 
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb#scrollTo=h2q27gKz1H20

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image
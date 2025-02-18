"""
EECS 6692: Deep Learning on the Edge
File contains functions to display images and annotations
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def display_image(image_path: str) -> None:
    """
    Display an image from the specified path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    print(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()


def display_annotated_image(image: np.ndarray) -> None:
    """
    Display an annotated image.

    Args:
        image (np.ndarray): The image to display.

    Returns:
        None
    """
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Display the image using Matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draw pose landmarks on the input RGB image.

    Args:
        rgb_image (np.ndarray): The input RGB image.
        detection_result: The pose detection result containing landmarks.

    Returns:
        np.ndarray: The annotated image with pose landmarks drawn.

    Source:
    https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb#scrollTo=h2q27gKz1H20
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for pose_landmarks in pose_landmarks_list:
        # Convert pose landmarks to proto format
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])

        # Draw the pose landmarks on the image
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    return annotated_image

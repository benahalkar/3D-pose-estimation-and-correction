"""
EECS 6692: Deep Learning on the Edge
Authors: Devika Gumaste, Harsh Benahalkar
File contains functions to create angles and landmarks dictionaries
"""

import os
import cv2
import numpy as np
import pandas as pd
from utils.constants import Constants


def angle(p1, p2, p3):
    """
    Calculate the angle between three points.

    Args:
        p1 (tuple): Coordinates of the first point.
        p2 (tuple): Coordinates of the second point (vertex).
        p3 (tuple): Coordinates of the third point.

    Returns:
        float: Angle in degrees.
    """
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])

    vector_1 = np.arctan2(c[1] - b[1], c[0] - b[0])
    vector_2 = np.arctan2(a[1] - b[1], a[0] - b[0])
    radians = vector_1 - vector_2
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def create_landmarks_dict(result, pose_cols):
    """
    Create a dictionary of landmarks from the pose detection result.

    Args:
        result: Pose detection result.
        pose_cols (list): List of column names for pose data.

    Returns:
        dict: Dictionary of landmark coordinates or None if no person detected.
    """
    pose_list = []
    pre_list = []

    # If no person is detected
    if len(result.pose_landmarks) == 0:
        return None

    try:
        for landmark in result.pose_landmarks[0]:
            pre_list.append((landmark.x, landmark.y, landmark.z))

        # Shoulder, elbow, wrist
        list_11_16 = np.array([
            [pre_list[m][0], pre_list[m][1], pre_list[m][2]]
            for m in range(11, 17)
        ]).flatten().tolist()

        # Hip, knee, ankle
        list_23_33 = np.array([
            [pre_list[m][0], pre_list[m][1], pre_list[m][2]]
            for m in range(23, 33)
        ]).flatten().tolist()

        list_11_16.extend(list_23_33)

        combined_list = [pre_list[0][0], pre_list[0][1], pre_list[0][2]]
        combined_list.extend(list_11_16)
        tpl = combined_list.copy()
        tpl.append(16)
        pose_list.append(tpl)

    except Exception as e:
        print(e)
        return None

    data_pose = {pose_cols[i]: pose_list[0][i] for i in range(len(pose_cols))}
    return data_pose


def calculate_angles(landmarks_list):
    """
    Calculate angles for each joint.

    Args:
        landmarks_list (dict): Dictionary of landmark coordinates.

    Returns:
        tuple: Angles for various joints.
    """
    armpit_left = angle(
        landmarks_list["left_elbow"],
        landmarks_list["left_shoulder"],
        landmarks_list["left_hip"]
    )
    armpit_right = angle(
        landmarks_list["right_elbow"],
        landmarks_list["right_shoulder"],
        landmarks_list["right_hip"]
    )
    elbow_left = angle(
        landmarks_list["left_shoulder"],
        landmarks_list["left_elbow"],
        landmarks_list["left_wrist"]
    )
    elbow_right = angle(
        landmarks_list["right_shoulder"],
        landmarks_list["right_elbow"],
        landmarks_list["right_wrist"]
    )
    hip_left = angle(
        landmarks_list["left_shoulder"],
        landmarks_list["left_hip"],
        landmarks_list["left_knee"]
    )
    hip_right = angle(
        landmarks_list["right_shoulder"],
        landmarks_list["right_hip"],
        landmarks_list["right_knee"]
    )
    knee_left = angle(
        landmarks_list["left_hip"],
        landmarks_list["left_knee"],
        landmarks_list["left_ankle"]
    )
    knee_right = angle(
        landmarks_list["right_hip"],
        landmarks_list["right_knee"],
        landmarks_list["right_ankle"]
    )
    ankle_left = angle(
        landmarks_list["left_knee"],
        landmarks_list["left_ankle"],
        landmarks_list["left_foot_index"]
    )
    ankle_right = angle(
        landmarks_list["right_knee"],
        landmarks_list["right_ankle"],
        landmarks_list["right_foot_index"]
    )
    return (armpit_left, armpit_right, elbow_left, elbow_right,
            hip_left, hip_right, knee_left, knee_right, ankle_left, ankle_right)


def create_angles_dict(result, pose_cols):
    """
    Create a dictionary of angles from the pose detection result.

    Args:
        result: Pose detection result.
        pose_cols (list): List of column names for pose data.

    Returns:
        dict: Dictionary of joint angles or None if no person detected.
    """
    pose_list = []
    angles_dict = {}

    # If no person is detected
    if len(result.pose_landmarks) == 0:
        return None

    try:
        # Fetch the landmarks
        pre_list = [(landmark.x, landmark.y, landmark.z)
                    for landmark in result.pose_landmarks[0]]

        # List for shoulder, elbow, wrist
        list_11_16 = np.array([pre_list[m] for m in range(11, 17)]).flatten().tolist()

        # Hip, knee, ankle
        list_23_33 = np.array([pre_list[m] for m in range(23, 33)]).flatten().tolist()

        list_11_16.extend(list_23_33)

        all_list = list(pre_list[0])
        all_list.extend(list_11_16)
        tpl = all_list.copy()
        tpl.append(16)
        pose_list.append(tpl)

        data_pose = pd.DataFrame(pose_list, columns=pose_cols)

        for i, row in data_pose.iterrows():
            landmarks_list = Constants.LANDMARKS_LIST.value
            for landmark in landmarks_list:
                landmarks_list[landmark] = [row[f"{landmark}_x"], row[f"{landmark}_y"]]

            angles = calculate_angles(landmarks_list)
            angles_dict[i] = dict(zip(
                ["armpit_left", "armpit_right", "elbow_left", "elbow_right",
                 "hip_left", "hip_right", "knee_left", "knee_right",
                 "ankle_left", "ankle_right"],
                angles
            ))

        return angles_dict

    except Exception as e:
        print(e)
        return None

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

# Function to calculate angle between two points
def angle(p1, p2, p3):
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])

    vector_1 = np.arctan2(c[1] - b[1], c[0] - b[0])
    vector_2 = np.arctan2(a[1] - b[1], a[0] - b[0])
    radians =  vector_1 - vector_2 
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# Function to create landmarks data structure
def create_landmarks_dict(result, pose_cols):
    pose_list = []
    pre_list = []

    # if no person is detected
    if len(result.pose_landmarks) == 0:
        return None
    
    try:
        for landmark in result.pose_landmarks[0]:
            pre_list.append((landmark.x, landmark.y, landmark.z))

        #shoulder elbow wrist
        list_11_16 = np.array([
            [
                pre_list[m][0],
                pre_list[m][1],
                pre_list[m][2],
            ] for m in range(11, 17)
        ]).flatten().tolist()

        # hip knee ankle
        list_23_33 = np.array([
            [
                pre_list[m][0],
                pre_list[m][1],
                pre_list[m][2],
            ] for m in range(23, 33)
        ]).flatten().tolist()
    
        list_11_16.extend(list_23_33)
    
        combined_list = [
            pre_list[0][0],
            pre_list[0][1],
            pre_list[0][2]
        ]
    
        combined_list.extend(list_11_16)
        tpl = combined_list.copy()
        tpl.append(16)
        pose_list.append(tpl)
    
    except Exception as e:
        print(e)
        return None
    
    data_pose = {pose_cols[i]: pose_list[0][i] for i in range(len(pose_cols))}
    return data_pose

# Function to call angles for each joint
def calculate_angles(landmarks_list):
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
    return armpit_left, armpit_right, elbow_left, elbow_right, hip_left, hip_right, knee_left, knee_right, ankle_left, ankle_right

# Function to create angles data structure
def create_angles_dict(result, pose_cols):
    pose_list = []
    angles_dict = {}

    pre_list = []

    # if no person is detected
    if len(result.pose_landmarks) == 0:
        return None
    
    try:
        # fetch the landmarks
        for landmark in result.pose_landmarks[0]:
            pre_list.append((landmark.x,
                             landmark.y,
                             landmark.z))

        # list for shoulder, elbow, wrist
        list_11_16 = np.array([
            [
                pre_list[m][0],
                pre_list[m][1],
                pre_list[m][2]
            ] for m in range(11, 17)
        ]).flatten().tolist()

        # hip knee ankle
        list_23_33 = np.array([
            [
                pre_list[m][0],
                pre_list[m][1],
                pre_list[m][2]
            ] for m in range(23, 33)
        ]).flatten().tolist()

        list_11_16.extend(list_23_33)

        all_list = [
            pre_list[0][0],
            pre_list[0][1],
            pre_list[0][2],
        ]

        all_list.extend(list_11_16)
        tpl = all_list.copy()
        tpl.append(16)
        pose_list.append(tpl)

        data_pose = pd.DataFrame(pose_list, columns=pose_cols)

        for i, row in data_pose.iterrows():
            sl = []
            landmarks_list = Constants.LANDMARKS_LIST.value
            landmarks_list["left_shoulder"] = [row["left_shoulder_x"], row["left_shoulder_y"]]
            landmarks_list["right_shoulder"] = [row["right_shoulder_x"], row["right_shoulder_y"]]
            landmarks_list["left_elbow"] = [row["left_elbow_x"], row["left_elbow_y"]]
            landmarks_list["right_elbow"] = [row["right_elbow_x"], row["right_elbow_y"]]
            landmarks_list["left_wrist"] = [row["left_wrist_x"], row["left_wrist_y"]]
            landmarks_list["right_wrist"] = [row["right_wrist_x"], row["right_wrist_y"]]
            landmarks_list["left_hip"] = [row["left_hip_x"], row["left_hip_y"]]
            landmarks_list["right_hip"] = [row["right_hip_x"], row["right_hip_y"]]
            landmarks_list["left_knee"] = [row["left_knee_x"], row["left_knee_y"]]
            landmarks_list["right_knee"] = [row["right_knee_x"], row["right_knee_y"]]
            landmarks_list["left_ankle"] = [row["left_ankle_x"], row["left_ankle_y"]]
            landmarks_list["right_ankle"] = [row["right_ankle_x"], row["right_ankle_y"]]
            landmarks_list["left_heel"] = [row["left_heel_x"], row["left_heel_y"]]
            landmarks_list["right_heel"] = [row["right_heel_x"], row["right_heel_y"]]
            landmarks_list["left_foot_index"] = [row["left_foot_index_x"], row["left_foot_index_y"]]
            landmarks_list["right_foot_index"] = [row["right_foot_index_x"], row["right_foot_index_y"]]

            armpit_left, armpit_right, elbow_left, elbow_right, hip_left, hip_right, \
            knee_left, knee_right, ankle_left, ankle_right = calculate_angles(landmarks_list)

            angles_dict[i] = {
                "armpit_left": armpit_left,
                "armpit_right": armpit_right,
                "elbow_left": elbow_left,
                "elbow_right": elbow_right,
                "hip_left": hip_left,
                "hip_right": hip_right,
                "knee_left": knee_left,
                "knee_right": knee_right,
                "ankle_left": ankle_left,
                "ankle_right": ankle_right
            }

        return angles_dict

    except Exception as e:
        print(e)
        return None

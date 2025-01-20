"""
EECS 6692: Deep Learning on the Edge
File contains constants for use through the application
"""

from enum import Enum

# source: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

class Constants(Enum):
    BODY_KP = {
    "nose": 0,
    "left_shoulder": 11, 
    "right_shoulder": 12,
    "left_elbow": 13, 
    "right_elbow": 14,
    "left_wrist": 15, 
    "right_wrist": 16,
    "left_hip": 23, 
    "right_hip": 24,
    "left_knee": 25, 
    "right_knee": 26,
    "left_ankle": 27, 
    "right_ankle": 28,
    "left_heel": 29, 
    "right_heel": 30,
    "left_foot_index": 31, 
    "right_foot_index": 32
    }

    YOGA_CLASSES = {
        1: "chair_pose", 
        2: "downward_facing_dog", 
        3: "revolved_triangle", 
        4: "half_moon", 
        5: "tree_pose"
    }

    GROUND_TRUTHS = {
        "chair_pose": {
            "armpit_left":180,
            "armpit_right":180,
            "elbow_left":180,
            "elbow_right":180,
            "hip_left":60,
            "hip_right":60,
            "knee_left":100,
            "knee_right":100
        },
        "downward_facing_dog": {
            "armpit_left":180,
            "armpit_right":180,
            "elbow_left":180,
            "elbow_right":180,
            "hip_left":90,
            "hip_right":90,
            "knee_left":180,
            "knee_right":180,
            "ankle_left":60,
            "ankle_right":60
        },
        "revolved_triangle": {
            "armpit_left":90,
            "armpit_right":90,
            "elbow_left":180,
            "elbow_right":180,
            "hip_left":60,
            "hip_right":120,
            "knee_left":180,
            "knee_right":180,
            "ankle_left":120,
            "ankle_right":60
        },
        "half_moon": {
            "armpit_left":90,
            "armpit_right":90,
            "elbow_left":180,
            "elbow_right":180,
            "hip_left":90,
            "hip_right":180,
            "knee_left":180,
            "knee_right":180,
            "ankle_left":90,
            "ankle_right":180
        },
        "tree_pose": {
            "armpit_left":180,
            "armpit_right":180,
            "elbow_left":120,
            "elbow_right":120,
            "hip_left":180,
            "hip_right":120,
            "knee_left":180,
            "knee_right":30,
            "ankle_left":180,
            "ankle_right":90
        } 
    }

    LANDMARKS_LIST = {
    "left_shoulder": [], "right_shoulder": [],
    "left_elbow": [], "right_elbow": [],
    "left_wrist": [], "right_wrist": [],
    "left_hip": [], "right_hip": [],
    "left_knee": [], "right_knee": [],
    "left_ankle": [], "right_ankle": [],
    "left_heel": [], "right_heel": [],
    "left_foot_index": [], "right_foot_index": [],
    }

    PERSONALIZED_MESSAGES={
        "chair_pose": {
            "armpit_left":"Raise your arms overhead!",
            "armpit_right":"Raise your arms overhead!",
            "elbow_left":"Keep your elbows straight!",
            "elbow_right":"Keep your elbows straight!",
            "hip_left":"Shift your weight back into your heels!",
            "hip_right":"Shift your weight back into your heels!",
            "knee_left":"Bend your knees to make your thighs are parallel to the ground!",
            "knee_right":"Bend your knees to make your thighs are parallel to the ground!",
        },
        "downward_facing_dog": {
            "armpit_left":"Stretch your arms!",
            "armpit_right":"Stretch your arms!",
            "elbow_left":"Stretch your arms!",
            "elbow_right":"Stretch your arms!",
            "hip_left":"Lengthen your spine!",
            "hip_right":"Lengthen your spine!",
            "knee_left":"Keep your legs straight!",
            "knee_right":"Keep your legs straight!",
            "ankle_left":"Take care not to strain your ankles!",
            "ankle_right":"Take care not to strain your ankles!"
        },
        "revolved_triangle": {
            "armpit_left":"Straighten your arms to make them in one line!",
            "armpit_right":"Straighten your arms to make them in one line!",
            "elbow_left":"Don't bend your elbows!",
            "elbow_right":"Don't bend your elbows!",
            "knee_left":"Keep your legs straight!",
            "knee_right":"Keep your legs straight!",
            "ankle_left":"Press your ankle firmly into the ground!",
            "ankle_right":"Press your ankle firmly into the ground!"
        },
        "half_moon": {
            "armpit_left":"Straighten your arms to make them in one line!",
            "armpit_right":"Straighten your arms to make them in one line!",
            "elbow_left":"Don't bend your elbows!",
            "elbow_right":"Don't bend your elbows!",
            "hip_left":"Press your leg firmly into the ground!",
            "hip_right":"Lift your leg higher!",
            "knee_left":"Keep your legs straight!",
            "knee_right":"Keep your legs straight!"
        },
        "tree_pose": {
            "armpit_left":"Stretch your arms towards the ceiling!",
            "armpit_right":"Stretch your arms towards the ceiling!",
            "elbow_left":"Don't bend your elbows too much",
            "elbow_right":"Don't bend your elbows too much",
            "hip_left":"Make sure your leg on the ground is straight!",
            "hip_right":"Lift your leg higher towards you!",
            "knee_left":"Make sure your leg on the ground is straight!",
            "knee_right":"Lift your leg higher towards you!"
        } 
    }




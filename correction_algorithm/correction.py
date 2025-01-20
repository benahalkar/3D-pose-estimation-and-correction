"""
EECS 6692: Deep Learning on the Edge
Authors: Devika Gumaste, Harsh Benahalkar
File contains functions for the correction algorithm
"""

from utils.constants import *

# Function to define the error margin
def error_margin(control, value):
    if int(value) in range(control - 20, control + 21):
        return True
    return False

# Function to check if the angle is in the threshold range
def check_joint(angles, joint_name, threshold, message):
    if error_margin(threshold, angles[joint_name]):
        return None
    if angles[joint_name] > threshold:
        return message
    elif angles[joint_name] < threshold:
        return message
    return None

# Function to check the angle
def check_pose_angle(angles):
    ground_truths = Constants.GROUND_TRUTHS.value
    feedback = {}
    for pose_name, pose_angles in ground_truths.items():
        pose_feedback = {}
        all_correct = True
        for joint, threshold in pose_angles.items():
            if joint in angles:
                personalized_messages = Constants.PERSONALIZED_MESSAGES.value
                message = personalized_messages.get(pose_name, {}).get(joint, {})
                error = check_joint(angles, joint, threshold, message)
            if error:
                all_correct = False
                pose_feedback[joint] = error
        if all_correct:
            feedback[pose_name] = "Correct"
        else:
            feedback[pose_name] = pose_feedback
    return feedback

# Function to format the feedback
def format_feedback(pose_feedback):
    formatted_feedback = ""
    if pose_feedback == "Correct":
        formatted_feedback += f" Correct! Keep Going!"
    else:
        for joint, error_message in pose_feedback.items():
            if error_message not in formatted_feedback:
                formatted_feedback += f"{error_message}"
                formatted_feedback += " "
    return formatted_feedback
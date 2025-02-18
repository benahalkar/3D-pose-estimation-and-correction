"""
EECS 6692: Deep Learning on the Edge
Authors: Devika Gumaste, Harsh Benahalkar
File contains functions for the correction algorithm
"""

from utils.constants import Constants


def error_margin(control: int, value: int) -> bool:
    """
    Check if the value is within the error margin of the control.

    Args:
        control (int): The control value.
        value (int): The value to check.

    Returns:
        bool: True if within margin, False otherwise.
    """
    return control - 20 <= int(value) <= control + 20


def check_joint(angles: dict, joint_name: str, threshold: int, message: str) -> str | None:
    """
    Check if the joint angle is within the threshold.

    Args:
        angles (dict): Dictionary of joint angles.
        joint_name (str): Name of the joint to check.
        threshold (int): Threshold angle.
        message (str): Feedback message.

    Returns:
        str | None: Feedback message if outside threshold, None otherwise.
    """
    if error_margin(threshold, angles[joint_name]):
        return None
    if angles[joint_name] > threshold:
        return message
    elif angles[joint_name] < threshold:
        return message
    return None


def check_pose_angle(angles: dict) -> dict:
    """
    Check the pose angles against ground truths.

    Args:
        angles (dict): Dictionary of joint angles.

    Returns:
        dict: Feedback for each pose.
    """
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
        feedback[pose_name] = "Correct" if all_correct else pose_feedback
    return feedback


def format_feedback(pose_feedback: dict | str) -> str:
    """
    Format the feedback for output.

    Args:
        pose_feedback (dict | str): Feedback for a pose.

    Returns:
        str: Formatted feedback string.
    """
    if pose_feedback == "Correct":
        return " Correct! Keep Going!"
    
    formatted_feedback = ""
    for error_message in pose_feedback.values():
        if error_message not in formatted_feedback:
            formatted_feedback += f"{error_message} "
    return formatted_feedback.strip()

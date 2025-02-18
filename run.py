"""
EECS 6692: Deep Learning on the Edge
Authors: Harsh Benahalkar, Devika Gumaste

This script implements a real-time yoga pose detection and feedback system
using MediaPipe and Tkinter.
"""

import time
import os
import tkinter as tk
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from PIL import Image, ImageTk

from utils.helper_functions import create_angles_dict
from correction_algorithm.main import check_pose_angle
from correction_algorithm.correction import format_feedback
from utils.constants import Constants

# MediaPipe pose detection setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables
CURR_PATH = os.getcwd()
ITERATIONS = 50
loop_count = 0
camera_result = None
yoga_options = [
    "Downward Facing Dog",
    "Chair Pose",
    "Revolved Triangle",
    "Half Moon",
    "Tree Pose"
]

def uglify(pose_name):
    """Convert yoga pose names to lowercase and replace spaces with underscores."""
    return pose_name.lower().replace(" ", "_")

def print_result(result, output_image, timestamp_ms):
    """Callback function to handle camera results."""
    global camera_result
    camera_result = result

# MediaPipe pose detection options
base_options = python.BaseOptions(
    model_asset_path=os.path.join(CURR_PATH, 'models/pose_landmarker.task')
)
options = vision.PoseLandmarkerOptions(
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Tkinter window setup
window = tk.Tk()
window.geometry("1000x480")
window.configure(bg="#ffffff")
window.title("DGHB EECS6692 S24 final project")

# Camera setup
cap = cv2.VideoCapture(0)
camera_width, camera_height = 640, 480

def update_camera():
    """Update the camera feed and process pose detection."""
    global photo, loop_count
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector.detect_async(image, time.time_ns() // 1_000_000)

        if camera_result:
            for pose_landmarks in camera_result.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Generate feedback
            if loop_count == ITERATIONS:
                col_names = [f"{name}_{coord}" for name in Constants.BODY_KP.value.keys()
                             for coord in ['x', 'y', 'z']]
                pose_cols = col_names + ['pose']
                angles_dict = create_angles_dict(camera_result, pose_cols)
                if angles_dict is not None:
                    feedback = check_pose_angle(angles_dict[0])
                    feedback_final = format_feedback(feedback[uglify(curr_option)])
                    text1.delete("1.0", tk.END)
                    text1.insert(tk.END, feedback_final)
                else:
                    text1.delete("1.0", tk.END)
                    text1.insert(tk.END, "No pose detected!")
                loop_count = 0
            loop_count += 1

        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        window.after(1, update_camera)

def on_option_selected(value):
    """Handle selection of yoga pose options."""
    global curr_option
    curr_option = value
    label1.config(text=f"Posture selected: {curr_option}")

    image_path = os.path.join(CURR_PATH, "images", f"{uglify(value)}.png")
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image=image)
    label2.config(image=photo)
    label2.image = photo

# Tkinter UI components
canvas = tk.Canvas(window, width=camera_width, height=camera_height, bg="white")
canvas.pack(side=tk.LEFT)

curr_option = yoga_options[0]
label1 = tk.Label(window, text=f"Posture selected: {curr_option}", font=("Arial", 12), bg="white", fg="black")
label1.place(x=650, y=50)

default_image_path = os.path.join(CURR_PATH, "images", f"{uglify(curr_option)}.png")
default_image = Image.open(default_image_path)
default_photo = ImageTk.PhotoImage(default_image)
label2 = tk.Label(window, image=default_photo, bg="white")
label2.place(x=750, y=100)

selected_option = tk.StringVar(window)
selected_option.set(yoga_options[0])
menu = tk.OptionMenu(window, selected_option, *yoga_options, command=on_option_selected)
menu.config(bg="#ffffff", fg="black", font=("Arial", 12), width=28, height=1)
menu.pack(side=tk.RIGHT)
menu.place(x=650, y=280)

text1 = tk.Text(window, wrap="word", width=36, height=5)
text1.pack(fill="both", expand=True)
text1.place(x=650, y=330)

# Start the application
update_camera()
window.mainloop()

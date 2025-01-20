"""
EECS 6692: Deep Learning on the Edge
Authors: Harsh Benahalkar, Devika Gumaste
"""

# Import statements
import time
import os
import tkinter as tk
import cv2
import sys

# Import mediapipe library for pose estimation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from utils.helper_functions import *
from correction_algorithm.main import *
from correction_algorithm.correction import *
from utils.constants import *
from PIL import Image, ImageTk

# Setting up mediapipe pose detection objects
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Getting the current working directory
curr_path = os.getcwd()

# Initializing a tkinter window
window = tk.Tk()

# Setting up dimensions for the window and camera feed
width = 2000
height = 1000
camera_width = 640
camera_height = 480

# Constants for iteration count and loop count
ITERATIONS = 50
loop_count = 0

camera_result = None
yoga_options = ["Downward Facing Dog", "Chair Pose", "Revolved Triangle", "Half Moon", "Tree Pose"]
# Function to convert yoga pose names to lowercase and replace spaces with underscores
uglify = lambda x : x.lower().replace(" ", "_")

# Callback function to handle camera results
def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global camera_result 
    camera_result = result

# Setting up options for mediapipe pose detection
base_options = python.BaseOptions(model_asset_path=os.path.join(curr_path, 'models/pose_landmarker.task'))
options = vision.PoseLandmarkerOptions(
    num_poses = 1,
    min_pose_detection_confidence = 0.5,
    min_pose_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    base_options=base_options,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback = print_result,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# Setting the current option to the first yoga pose option
curr_option = yoga_options[0]

# Default image path based on the current option
default_image_path = os.path.join(curr_path, "images", 
                                  f"{uglify(curr_option)}.png")
default_image = Image.open(default_image_path)
default_photo = ImageTk.PhotoImage(default_image)

# Initializing the camera capture object
cap = cv2.VideoCapture(0)

# Function to update the camera feed
def update_camera():
    global photo
    global loop_count
    global ITERATIONS
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector.detect_async(image, time.time_ns() // 1_000_000)

        if camera_result:
            for pose_landmarks in camera_result.pose_landmarks:
                # Drawing detected pose landmarks on the camera frame
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, 
                                                    y=landmark.y,
                                                    z=landmark.z) for landmark in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())

            # feedback
            col_names = []
            for i in range(len(Constants.BODY_KP.value.keys())):
                name = list(Constants.BODY_KP.value.keys())[i]
                col_names.append(name + "_x")
                col_names.append(name + "_y")
                col_names.append(name + "_z")
            
            pose_cols = col_names.copy()
            pose_cols.append("pose")
            
            # Generating feedback based on detected pose angles
            if loop_count == ITERATIONS:
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
        
            loop_count = loop_count + 1

        # Displaying the camera frame on the tkinter canvas
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        window.after(1, update_camera)

# Function to handle selection of yoga pose options
def on_option_selected(value):
    global curr_option
    curr_option = value
    label1.config(text=f"Posture selected: {curr_option}")

    image_path = os.path.join(curr_path, "images", f"{uglify(value)}.png")
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image=image)
    label2.config(image=photo)
    label2.image = photo

# Setting up tkinter window and components
window.geometry("%dx%d" % (1000, camera_height))
window.configure(bg="#ffffff")
window.title("DGHB EECS6692 S24 final project")

canvas = tk.Canvas(window, width=camera_width, height=camera_height, bg="white")
canvas.pack(side=tk.LEFT)

label1 = tk.Label(window, text=f"Posture selected: {curr_option}", font=("Arial", 12), bg="white", fg="black")
label1.place(x=650, y=50)

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

# Start updating the camera feed
update_camera()

window.mainloop()





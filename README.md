# Real-time 3D Pose Estimation and Correction on Edge Devices

## Overview
This project focuses on real-time 3D pose estimation and correction on edge devices, particularly targeting yoga positions. The goal is to develop an algorithm capable of accurately estimating and correcting yoga poses in real-time, enabling users to receive immediate feedback on their posture during yoga practice.

## Features
- Real-time 3D pose estimation
- Pose correction feedback
- Compatibility with edge devices
- Lightweight deployment

## Dataset
Dataset Link: https://drive.google.com/file/d/11HjuGn3jjBs2OZWsYYRNXZbFdxtsAW9A/view?usp=drive_link

## Document
Slides Link: https://docs.google.com/presentation/d/16WHeVXlc0pAqIycT12U2x3bWU_rMbUGPWf4NYe--O5k/edit?usp=drive_link

Demo Link: https://drive.google.com/file/d/1u_buEbbIa9S7vfh5Dxm3K8lPJFKcHsHN/view?usp=drive_link

## Models
We experimented with three different pose estimation models:
1. OpenPose
2. PoseNet
3. MediaPipe

After thorough testing, we concluded that MediaPipe provided the best balance between accuracy and efficiency for our application, particularly when deployed on resource-constrained edge devices.

## Deployment
We successfully deployed the application on a 2GB Jetson Nano, demonstrating its capability to run efficiently on edge devices with limited resources.

## Usage
To use the application, follow these steps:
1. Clone the repository
```
git clone https://github.com/eecse6692/e6692-2024spring-finalproject-dghb.git
```
2. Move into the repository
```
cd e6692-2024spring-finalproject-dghb
```
3. Install the necessary dependencies.
```
pip install -r requirements.txt
```
4. Run the application on your edge device using run.py file.
```
python run.py
```
5. Follow the on-screen instructions for pose estimation and correction during yoga practice.


## Organization of the Directory
```
├── README.md
├── correction_algorithm
│   ├── correction.py
│   └── main.py
├── images
│   ├── chair_pose.png
│   ├── downward_facing_dog.png
│   ├── downward_facing_dog_org.png
│   ├── half_moon.png
│   ├── revolved_triangle.png
│   └── tree_pose.png
├── mediapipe
│   └── mediapipe_inference.ipynb
├── models
│   ├── pose_landmarker.task
│   └── pose_landmarker_lite.task
├── openpose
│   ├── openpose_inference.py
│   └── temp.txt
├── posenet
│   ├── __init__.py
│   ├── benchmark.py
│   ├── constants.py
│   ├── converter
│   │   ├── tfjs2pytorch.py
│   │   └── wget.py
│   ├── decode.py
│   ├── decode_multi.py
│   ├── get_test_images.py
│   ├── ground_truth_dataloop.py
│   ├── ground_truth_roboflow.py
│   ├── image_demo.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── mobilenet_v1.py
│   │   └── model_factory.py
│   ├── posenet_inference.py
│   ├── streamlit_demo.py
│   ├── train.py
│   ├── utils.py
│   ├── visualizations.ipynb
│   ├── visualizers.py
│   └── webcam_demo.py
├── requirements.txt
├── run.py
└── utils
    ├── constants.py
    └── helper_functions.py

9 directories, 39 files
```


import cv2
import os
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import time

import posenet
from posenet.decode_multi import *

dataset_dir = "./final_dataset"
output_dir = "./posenet_inferences"

SF = 1.0

total_time = 0
num_images = 0

if __name__ == "__main__":

  model = posenet.load_model(101)
  model = model.cuda()
  output_stride = model.output_stride

  if os.path.exists(output_dir):  os.system(f"rm -r {output_dir}")
  os.makedirs(output_dir)
    
  classes = [item for item in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, item)) and item != ".ipynb_checkpoints")]
  
  for class_ in classes:
    os.makedirs(os.path.join(output_dir, class_))
    
    filenames = [f.path for f in os.scandir(os.path.join(dataset_dir, class_)) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    
    start = time.time()
    for f in filenames:
      
      inferred_image_name = f.split('/')[-1]
      input_image, draw_image, output_scale = posenet.read_imgfile(f, scale_factor=SF, output_stride=output_stride)

      with torch.no_grad():
        input_image = torch.Tensor(input_image).cuda()

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
        pose_scores, keypoint_scores, keypoint_coords, pose_offsets = posenet.decode_multi.decode_multiple_poses(
          heatmaps_result.squeeze(0),
          offsets_result.squeeze(0),
          displacement_fwd_result.squeeze(0),
          displacement_bwd_result.squeeze(0),
          output_stride=output_stride,
          max_pose_detections=10,
          min_pose_score=0.25)

      keypoint_coords *= output_scale

      draw_image = posenet.draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords,
              min_pose_score=0.25, min_part_score=0.25)

      cv2.imwrite(os.path.join(output_dir, class_, inferred_image_name), draw_image)
      
      num_images = num_images + 1
    
    total_time = total_time + time.time() - start 
    
fps = round(num_images/total_time , 3)
print(f"FPS measured by posenet is {fps}")
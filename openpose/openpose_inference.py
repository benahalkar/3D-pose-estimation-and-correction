import cv2
# import matplotlib.pyplot as plt
import copy
import numpy as np

import time
import os
from src import model
from src import util
from src.body import Body
from src.hand import Hand

dataset_dir = "./final_dataset"
output_dir = "./openpose_inferences"

total_time = 0
num_images = 0

model_path = './model/body_pose_model.pth'


if __name__ == "__main__":
  
  body_estimation = Body(model_path, "coco")
  
  if os.path.exists(output_dir):  os.system(f"rm -r {output_dir}")
  os.makedirs(output_dir)
  
  classes = [item for item in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, item)) and item != ".ipynb_checkpoints")]
  
  for class_ in classes:
    os.makedirs(os.path.join(output_dir, class_))
    
    filenames = [f.path for f in os.scandir(os.path.join(dataset_dir, class_)) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    
    start = time.time()
    for f in filenames:
      
      inferred_image_name = f.split('/')[-1]

      oriImg = cv2.resize(cv2.imread(f), (368, 368)) 
      candidate, subset = body_estimation(oriImg)
      canvas = copy.deepcopy(oriImg)
      canvas = util.draw_bodypose(canvas, candidate, subset, "coco")

      cv2.imwrite(os.path.join(output_dir, class_, inferred_image_name), canvas)
      
      num_images = num_images + 1
      
    total_time = total_time + time.time() - start 
    print(f"{class_} done")

fps = round(num_images/total_time , 3)
print(f"FPS measured by openpose is {fps}")

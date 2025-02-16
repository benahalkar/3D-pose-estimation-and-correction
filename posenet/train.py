
# create a train module that trains the model
# Path: posenet-pytorch/train.py
# inspiration: https://github.com/youngguncho/PoseNet-Pytorch/blob/master/posenet_simple.py 
# reference: https://github.com/Lornatang/MobileNetV1-PyTorch/blob/main/train.py

# install torch with pip
# Path: posenet-pytorch/train.py

# resolution (image size: 225; stride: 16) 
# // 15 = ((225 - 1) / 16) + 1
# output[0]: heatmap  [15, 17, 33, 33] 
# output[1]: offset vectors [15, 34, 33, 33] 
# output[2]: displacement forward [15, 32, 33, 33] 
# output[3]: displacement backward [15, 32, 33, 33]

import cv2
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import posenet
import time
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from ground_truth_dataloop import *
from posenet.decode_multi import *
from visualizers import *
from scipy.optimize import linear_sum_assignment
import wandb
import torch.optim as optim
import numpy as np


os.environ["WANDB_NOTEBOOK_NAME"] = "./train_model_wandb.ipynb"



CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--train_image_dir', type=str, default='./images_train')
parser.add_argument('--test_image_dir', type=str, default= "./images_train")
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--scale_factor', type=float, default=1.0)


args = parser.parse_args()

# Loss function with Hough Voting 

# class MaskedBCEWithLogitsLoss(nn.Module):
#     def __init__(self):
#         super(MaskedBCEWithLogitsLoss, self).__init__()
#         self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, input, target, mask):
#         # Compute BCEWithLogitsLoss
#         loss = self.bce_with_logits_loss(input, target)

#         # Apply the mask
#         masked_loss = loss * mask

#         # Compute the mean loss over the masked elements
#         # mean_loss = torch.sum(masked_loss) / torch.sum(mask)

#         return masked_loss

class MultiPersonHeatmapOffsetAggregationLoss(nn.Module):
    def __init__(self, radius=3, heatmap_weight=4.0, offset_weight=1.0, use_target_weight=False, max_num_poses=15):
        super(MultiPersonHeatmapOffsetAggregationLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothl1loss = nn.SmoothL1Loss(reduction='none')
        self.radius = radius
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.use_target_weight = use_target_weight
        self.max_num_poses= max_num_poses
        


    def create_mask(self, ground_truth, threshold=0.1):
        # Threshold the ground truth heatmaps to create a binary mask
        mask = (ground_truth > threshold).float()

        # Apply dilation to create a disk-like region around each keypoint
        padding = self.radius
        kernel_size = 2 * self.radius + 1
        mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

        mask = mask.cuda()
        
        return mask


    def create_binary_target_heatmap(self, target_heatmaps, target_keypoints, radius=3):
        #TODO: check if binary target heatmaps is in the right shape and if it should be zeros_like
        binary_target_heatmaps = torch.zeros_like(target_heatmaps)

#         print("target_heatmaps shape: ", target_heatmaps.shape)
#         print("target_keypoints shape: ", target_keypoints.shape)

        for k in range(target_keypoints.shape[0]):

            x, y = target_keypoints[k, 0], target_keypoints[k, 1]
            # print("x: ", x)
            # print("y: ", y)
            
            if (x != 0 and x != -1) or (y != 0 and y != -1):
                x, y = int(x.item()), int(y.item())
                y_min, y_max = max(0, y - radius), min(binary_target_heatmaps.shape[1], y + radius + 1)
                x_min, x_max = max(0, x - radius), min(binary_target_heatmaps.shape[2], x + radius + 1)

                y_indices, x_indices = np.mgrid[y_min:y_max, x_min:x_max]
                y_indices, x_indices = torch.tensor(y_indices), torch.tensor(x_indices)
                distances = torch.sqrt((y_indices - y)**2 + (x_indices - x)**2)

                binary_target_heatmaps[k, y_min:y_max, x_min:x_max] = (distances <= radius).float()

        return binary_target_heatmaps

    def forward(self, pred_heatmaps, target_heatmaps, target_keypoints, pred_offsets, target_offsets, max_num_poses = 15):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        binary_target_heatmaps = torch.zeros_like(target_heatmaps)
        # Heatmap loss
        
        
        loss = 0.0
        
        #TODO update num_people logic
        # print("--target keypoints --")
        # print("target keypoints shape: ", target_keypoints.shape)
        
        num_people = count_people(target_keypoints)
        print("**count_people num_people: *", num_people)
        
        heatmap_loss = torch.tensor(0.0).cuda()
        offset_loss = torch.tensor(0.0).cuda()

        pred_offsets = pred_offsets.view(1, 17, 2, 33, 33).permute(0, 1, 3, 4, 2)
        # print("pred_offsets_shape: ", pred_offsets.shape)
        ground_truth_offset_maps = create_ground_truth_offset_maps(target_keypoints, height=33, width=33, max_num_poses=max_num_poses)
        
        

        for pose in range(num_people):
            # Heatmap loss 
                        
            binary_target_heatmaps[pose, :, :, :] = self.create_binary_target_heatmap(target_heatmaps[pose], target_keypoints[pose], self.radius)
            
            pose_heatmap_loss = self.bceloss(pred_heatmaps, binary_target_heatmaps[pose].float())
            
            #             print("pred_heatmaps shape: ", pred_heatmaps.shape)
#             print("binary_target_heatmaps shape: ", binary_target_heatmaps.shape)
#             print("target_heatmaps shape: ", target_heatmaps.shape)
            
            heatmap_loss += pose_heatmap_loss.cuda()
            print("in pose loop: heatmap_loss value: ", heatmap_loss)
            
            # Offset Loss
            # Ground truth offsets will turn to shape [15, 17, 33, 33, 2]
            # print("target_keypoints shape: ", target_keypoints.shape)
            
            mask = self.create_mask(target_heatmaps[pose])
            
            mask = mask.unsqueeze(-1)
            
            
            
            print("&&&&loss&&&&")
            print("mask shape: ", mask.shape)
            print("ground_truth_offset_maps[pose] shape: ", ground_truth_offset_maps[pose].shape)
            print("pred_offsets shape: ", pred_offsets.shape)
            
            masked_true_offsets = ground_truth_offset_maps[pose] * mask
            masked_pred_offsets = pred_offsets * mask
            
            
            offset_loss += self.smoothl1loss(masked_pred_offsets, masked_true_offsets).mean()
            print("in pose loop: offset_loss value: ", offset_loss)
            
        
        
        heatmap_loss /= num_people
        offset_loss = offset_loss / num_people
        
        print("ground_truth_offset_maps shape: ", ground_truth_offset_maps.shape)    

        loss += (self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss) / (self.heatmap_weight + self.offset_weight)
            
        return loss, heatmap_loss, offset_loss, binary_target_heatmaps


def match_poses(preds, gts):
    """
    Match predicted poses to ground truth poses based on Euclidean distance of keypoints.

    Args:
        preds: Predicted keypoints, a numpy array of size (num_preds, num_keypoints, 2)
        gts: Ground truth keypoints, a numpy array of size (num_gts, num_keypoints, 2)

    Returns:
        matched_pairs: List of pairs of indices (pred_index, gt_index) for matched poses
    """
    preds_cpu = preds.cpu().numpy()
    gts_cpu = gts.cpu().numpy()
    
    # Compute pairwise distance between all preds and gts
    cost_matrix = np.zeros((len(preds_cpu), len(gts_cpu)))
    for i, pred in enumerate(preds_cpu):
        for j, gt in enumerate(gts_cpu):
            cost_matrix[i, j] = np.linalg.norm(pred - gt)

    # Use Hungarian Algorithm to find optimal match
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # All pairs are considered as matched pairs
    matched_pairs = list(zip(row_ind, col_ind))
    
    print("matched_pairs length: ", len(matched_pairs))
    print(matched_pairs)

    return matched_pairs

def calculate_oks(matched_pairs, preds, gts, sigmas, variances, image_size):
    """
    Calculate Object Keypoint Similarity (OKS) for matched pairs.

    Args:
        matched_pairs: List of pairs of indices (pred_index, gt_index) for matched poses
        preds: Predicted keypoints, a numpy array of size (num_preds, num_keypoints, 2)
        gts: Ground truth keypoints, a numpy array of size (num_gts, num_keypoints, 2)
        sigmas: A numpy array of standard deviations for the positions of each keypoint
        vars: A numpy array of the variances for each keypoint (typically, this is just square of sigmas)
        image_size: The size of the image (height, width)

    Returns:
        oks: The OKS score
    """
    oks = 0
    
    preds_cpu = preds.cpu().numpy()
    gts_cpu = gts.cpu().numpy()
    
    image_size = float(image_size)
    preds_cpu = preds_cpu.astype(np.float64)
    gts_cpu = gts_cpu.astype(np.float64)
    
    
    for i, j in matched_pairs:
        print("sigmas: ", sigmas)
        print("variances: ", variances)
        # Euclidean distance between predicted and ground truth keypoints
        d = np.linalg.norm(preds_cpu[i] - gts_cpu[j])
        print("d: ", d)
        
        exp = np.exp(-d**2 / (2 * variances * (sigmas**2)))
        print("exp: ", exp)
        # Compute OKS for each keypoint and sum them up
        oks += np.sum( exp / len(preds_cpu[i]))
        
    # Average OKS over all matched pairs
    oks = len(matched_pairs) if len(matched_pairs) > 0 else 0
    
    print("oks: ", oks)

    return oks


def calculate_precision(preds, gts, threshold = 2):
    
    """
    Calculate the precision of predicted keypoints compared to ground truth keypoints.
    A predicted keypoint is considered correct (true positive) if it is within 'threshold'
    distance of a ground truth keypoint. Any predicted keypoint not within 'threshold'
    distance of any ground truth keypoint is considered a false positive.

    Args:
        preds: Predicted keypoints, a numpy array of size (num_preds, num_keypoints, 2)
        gts: Ground truth keypoints, a numpy array of size (num_gts, num_keypoints, 2)
        threshold: The maximum distance for a predicted keypoint to be considered correct

    Returns:
        precision: The precision of the predictions
    """
        
    num_true_positives = 0
    num_false_positives = 0
    
    matched_pairs = match_poses(preds, gts)
    
    for pair in matched_pairs:
        pred_idx, gt_idx = pair
        pred_pose = normalize_keypoints(preds[pred_idx]).cpu().numpy()
        gt_pose = normalize_keypoints(gts[gt_idx]).cpu().numpy()

        
        for pred_keypoint, gt_keypoint in zip(pred_pose, gt_pose):
            # Skip keypoints with values (0,0) or (-1,-1) in ground truth keypoints
            if np.all(gt_keypoint == [0, 0]) or np.all(gt_keypoint == [-1, -1]):
                if np.all(pred_keypoint != [0, 0]) and np.all(pred_keypoint != [-1, -1]):
                    # This is a false positive - predicted but not in ground truth
                    num_false_positives += 1
                    print("false positive by gt no point detected")
                continue

            # For predicted keypoints, count true positives and false positives
            if np.linalg.norm(pred_keypoint - gt_keypoint) <= threshold:
                num_true_positives += 1
                print("true positive")
                print("np.linalg.norm(pred_keypoint - gt_keypoint):  ", np.linalg.norm(pred_keypoint - gt_keypoint))
            else:
                print("false positive")
                print("np.linalg.norm(pred_keypoint - gt_keypoint):  ", np.linalg.norm(pred_keypoint - gt_keypoint))
                num_false_positives += 1

    precision = num_true_positives / (num_true_positives + num_false_positives) if num_false_positives + num_false_positives > 0 else 0
    
    
    print("num_false_positives: ", num_false_positives)
    print("num_true_positives: ", num_true_positives)
    print("precision: ", precision)
    
    return precision


def calculate_recall(preds, gts, threshold=2.0):
    """
    Calculate recall for predicted keypoints against ground truth keypoints.

    Args:
        preds: Predicted keypoints, a numpy array of size (num_preds, num_keypoints, 2)
        gts: Ground truth keypoints, a numpy array of size (num_gts, num_keypoints, 2)
        threshold: The maximum Euclidean distance between a predicted keypoint and a ground truth keypoint for the prediction to be considered correct.

    Returns:
        recall: The recall of the predicted keypoints.
    """
    matched_pairs = match_poses(preds, gts)
    
    num_true_positives = 0
    num_false_negatives = 0

    for pred_index, gt_index in matched_pairs:
        pred_pose = normalize_keypoints(preds[pred_index]).cpu().numpy()
        gt_pose = normalize_keypoints(gts[gt_index]).cpu().numpy()
        

        for pred_keypoint, gt_keypoint in zip(pred_pose, gt_pose):
            
            if (gt_keypoint == np.array([-1, -1])).all() or (gt_keypoint == np.array([0, 0])).all():
                continue
            elif (pred_keypoint == np.array([-1, -1])).all() or (pred_keypoint == np.array([0, 0])).all():
                num_false_negatives += 1
                print("false negative")
            elif np.linalg.norm(pred_keypoint - gt_keypoint) <= threshold:
                num_true_positives += 1
                print("true positive")
                print("np.linalg.norm(pred_keypoint - gt_keypoint): ", np.linalg.norm(pred_keypoint - gt_keypoint))
            else:
                num_false_negatives += 1
                print("false negative")
                print("np.linalg.norm(pred_keypoint - gt_keypoint): ", np.linalg.norm(pred_keypoint - gt_keypoint))

    recall = num_true_positives / (num_true_positives + num_false_negatives)
    print("num_true_positives: ", num_true_positives)
    print("num_false_negatives: ", num_false_negatives)
    print("recall: ", recall)
    return recall



def normalize_keypoints(keypoints):
    """
    Normalize keypoints by subtracting the mean and dividing by the standard deviation.

    Args:
        keypoints: Keypoints to normalize, a tensor of size (num_keypoints, 2)

    Returns:
        normalized_keypoints: Normalized keypoints
    """
    keypoints = keypoints.float()
    mean = keypoints.mean(dim=0, keepdim=True)
    std = keypoints.std(dim=0, keepdim=True)
    normalized_keypoints = (keypoints - mean) / std

    return normalized_keypoints

def calculate_mAP(precisions, recalls):
    """
    Calculate the Mean Average Precision (mAP).

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        mAP: The Mean Average Precision
    """

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    sorted_precisions = precisions[sorted_indices]
    sorted_recalls = recalls[sorted_indices]
    
    # Append sentinel values at the end
    sorted_precisions = np.concatenate(([0], sorted_precisions, [0]))
    sorted_recalls = np.concatenate(([0], sorted_recalls, [1]))

    # Compute the precision envelope
    for i in range(sorted_precisions.size - 1, 0, -1):
        sorted_precisions[i - 1] = max(sorted_precisions[i - 1], sorted_precisions[i])

    # Compute Average Precision (AP)
    recall_change = np.diff(sorted_recalls)
    precision_change = sorted_precisions[:-1]
    AP = np.sum(recall_change * precision_change)

    return AP


class PosenetDatasetImage(Dataset):
    def __init__(self, file_path, ground_truth_keypoints_dir=None, scale_factor=1.0, output_stride=16, train=True):
        self.file_path = file_path
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.filenames = os.listdir(file_path)
        self.train = train
        self.ground_truth_keypoints_dir = ground_truth_keypoints_dir
        
        if ground_truth_keypoints_dir:
            image_file_names = [os.path.splitext(file)[0] for file in self.filenames if file.endswith((".jpg", ".png"))]
            self.keypoints, self.heatmaps, self.offset_vectors = load_ground_truth_data(image_file_names, self.ground_truth_keypoints_dir)
            # print("--inside dataset class init --")
            # print("keypoints shape: ", self.keypoints.shape)
            # print("heatmaps shape: ", self.heatmaps.shape)
            # print("offest_vectors shape: ", self.offset_vectors.shape)
            # # self.keypoints = torch.Tensor(self.keypoints).cuda()
            # self.heatmaps = torch.Tensor(self.heatmaps).cuda()
            # self.offset_vectors = torch.Tensor(self.offset_vectors).cuda()
            
            
            self.is_ground_truth = True
            print("PosenetDatasetImage filenames: ", self.filenames)
        else:
            self.is_ground_truth = False

        
        self.data = [f.path for f in os.scandir(file_path) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        self.filenames = [os.path.basename(file_path) for file_path in self.data]

        if  self.train:
            self.transforms = transforms.Compose([
                #not mandatory - at first don't apply augmentation first before applying 
                #transforms.RandomResizedCrop(256),
                #transforms.RandomHorizontalFlip(),
                
                #mandatory 
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                
                #mean and std values based on the pretrained model
                #mean value of the pixels of each channel [r, g, b]
                #std value of the pixels of each channel [r, g, b]
                
                transforms.Normalize(mean=[5.4476, 8.3573, 7.5377], std=[3.6566, 3.5510, 4.0362])
            ])
            
            if ground_truth_keypoints_dir:
                self.keypoints = torch.Tensor(self.keypoints).cuda().requires_grad_(False)
                self.heatmaps = torch.Tensor(self.heatmaps).cuda().requires_grad_(False)
                self.offset_vectors = torch.Tensor(self.offset_vectors).cuda().requires_grad_(False)
            
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[5.4476, 8.3573, 7.5377], std=[3.6566, 3.5510, 4.0362])
            ])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print("____getitem____ idx: ", idx)

        filename = self.filenames[idx]
        # print("____getitem____ filename: ", filename)
        
        # print("get_item: ", filename)
        input_image, draw_image, output_scale = posenet.read_imgfile(
            os.path.join(self.file_path, filename),
            scale_factor=self.scale_factor,
            output_stride=self.output_stride
        )
        
        # print("----input image: ----")
        # print(input_image)
        
        # print(filename)
        # print(input_image.shape)
        
        input_image_tensor = torch.Tensor(input_image).cuda()
        
        #print("Tensor shape: ", input_image_tensor.shape[-2:])
        if input_image_tensor.shape[-2:] != (513, 513):
            input_image_resized = nn.functional.interpolate(input_image_tensor, size=(513, 513), mode='bilinear', align_corners=True)
            # print(f"Resized image {filename}: ", input_image_resized.shape)
        
        if self.is_ground_truth:
            # print("print length of keypoints: ", len(self.keypoints))
            keypoints = self.keypoints[idx]
            heatmaps = self.heatmaps[idx]
            offset_vectors = self.offset_vectors[idx]
            
            return input_image_tensor, draw_image, output_scale, filename, keypoints, heatmaps, offset_vectors
                
        else:
            return input_image_tensor, draw_image, output_scale, filename

def get_dataset_mean_std(dataset):
    # Calculate the mean and standard deviation for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for i, (input_image_tensor, draw_image, _, _, _, _, _) in enumerate(dataset):
        # print("number of outputs of dataset: ", len(next(iter(dataset))))
        # print("draw_image type: ", type(draw_image))
        # print("draw_image shape: ", draw_image.shape)
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i in range(3):
            mean[i] = draw_image[..., i].mean()
            std[i] = draw_image[..., i].std()

    mean /= len(dataset)
    std /= len(dataset)
    print(f'mean: {mean}')
    print(f'std: {std}')
    
    return mean, std



def create_ground_truth_offset_maps(ground_truth_keypoints, height, width, scale_factor=8, max_num_poses=15):
    ground_truth_keypoints = ground_truth_keypoints.cuda()

    ground_truth_offset_maps = torch.zeros((max_num_poses, NUM_KEYPOINTS, height, width, 2), dtype=torch.float32).cuda()
    
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    y_coords, x_coords = (y_coords * scale_factor).cuda(), (x_coords * scale_factor).cuda()

    ground_truth_keypoints_expanded = ground_truth_keypoints.view(max_num_poses, NUM_KEYPOINTS, 1, 1, 2)

    ground_truth_offset_maps = ground_truth_keypoints_expanded - torch.stack((y_coords, x_coords), dim=-1)
    # print("--inside create ground truth offsets --")
    # print("ground_truth_offset_maps shape: ", ground_truth_offset_maps.shape)
    return ground_truth_offset_maps

# def write_keypoints_to_file(keypoints, epoch, file_name, pose_scores="", keypoint_scores=""):
#     with open(file_name, 'a') as f:
#         f.write(f"Epoch: {epoch}\n")
#         for pose_idx, pose in enumerate(keypoints):
#             for idx, keypoint in enumerate(pose): 
#                 f.write(f"{idx}: {keypoint}\n")
#                 f.write(f"keypoint score: {keypoint_scores[pose][idx]}")
#         f.write(f"pose score: {pose_scores[pose_idx]}")
#         f.write("\n")  # Separate epochs with a new line

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride, train_image_path, test_image_path, output_dir, scale_factor, is_train=True, max_num_poses=15):
    step = 0
    score_threshold = 0.25
    train_num_batches = len(train_loader)

    # Initialize the early stopping variables
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement before stopping
    no_improve_epochs = 0
    
    # TODO : find the sigmas and variances of the dataset 
    # typically it is given by COCO Dataset 
    # sigmas = np.ones(17)
    
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])
    variances = sigmas**2
    
    for epoch in range(num_epochs):
        
        epoch_start_time = time.time()
        batch_checkpoint = 2
        
        epoch_durations = []
        running_loss_value = 0
        test_loss_value = 0
        heatmap_loss_value = 0
        offset_loss_value = 0
        test_loss = torch.zeros(1)
        
        # Set model to train mode
        # print("Initial model weights:")
        # for name, param in model.named_parameters():
        #     print(name, param.data)
        
        # print("Initial Model weight norms:")
        # for name, param in model.named_parameters():
        #     print(name, param.data.norm())

        # with torch.no_grad():    
        if is_train:
            model.train()
            
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze the output layers
            for param in model.heatmap.parameters():
                param.requires_grad = True

            for param in model.offset.parameters():
                param.requires_grad = True

            for param in model.displacement_fwd.parameters():
                param.requires_grad = True

            for param in model.displacement_bwd.parameters():
                param.requires_grad = True
                
            print("---in training ---")
            
            print(train_loader)
            
        
            # print("train loader: ", next(iter(train_loader)))

            for batch_idx, (data, draw_image, output_scale, filenames, ground_truth_keypoints, ground_truth_heatmaps, ground_truth_offsets) in enumerate(train_loader):
                # print("ENUMERATE")
                
                # Backward pass
                optimizer.zero_grad()
                # print("batch size: ", train_loader.batch_size)
            
                data.cuda()
                # print("data shape: ", data.shape)
    
                data_squeezed = data.squeeze()
        
                # print("data_squeezed shape: ", data_squeezed.shape)
                output = model(data_squeezed)
                
                
                batch_loss = 0
                batch_mAP = 0
                batch_oks = 0
                
                #heatmap tensor = output[0] 
                #heatmap size is num of images x 17 keypoints x resolution x resolution 
                #eg. if image size is 225 with output stride of 16, then resolution is 15 
                #iterate through the batch size
                print("--filenames--")
                print(filenames)
                print(output[0].shape)
                
                
                for item_idx, item in enumerate(output[0]):
                    
                    
                    offsets = output[1][item_idx]
                    displacements_fwd = output[2][item_idx]
                    displacements_bwd = output[2][item_idx]

                    train_heatmaps = item
                    
                    filename = filenames[item_idx]
                    heatmaps_dir = "pred_heatmaps_training"
                    
                    save_heatmaps(train_heatmaps.detach().cpu().numpy(), filename, 0, num_keypoints=17, heatmaps_dir=heatmaps_dir, epoch=epoch)
                    
                    
                    
                    # print("item (heatmap) type: ", type(item))
                    
                    height = train_heatmaps.shape[1]
                    width = train_heatmaps.shape[2]
                    
                    # instance_keypoint_coords, instance_keypoint_scores , train_heatmaps, train_offsets = decode_pose_from_batch_item(epoch, train_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    pose_scores, keypoint_scores, keypoint_coords, decoded_offsets = decode_pose_from_batch_item(epoch, train_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    # print("---- keypoint_coords: ----")
                    # print(keypoint_coords)         
                    
                    #turn epoch to text
                    appended_text = "train_" + str(epoch) + "_"
                    print("pose_scores shape: ", pose_scores.shape)
                    print("keypoint_scores shape: ", keypoint_scores.shape)
                    output_dir_epoch = "keypoints_output_" + str(epoch)
                    
                    print("draw_image")
                    print(type(draw_image))
                    print("draw_image[idx]: ", draw_image[item_idx].shape)
                    print(draw_image.shape)
                    draw_image_with_kp = posenet.draw_skel_and_kp(np.array(draw_image[item_idx]), pose_scores, keypoint_scores, keypoint_coords,
                                                          min_pose_score=0.25, min_part_score=0.25)
                    
                    if not os.path.exists(os.path.join('./keypoint_output_training', filename, output_dir_epoch)):
                        os.makedirs(os.path.join('./keypoint_output_training', filename, output_dir_epoch))
                    
                    keypoint_output_dir = os.path.join('./keypoint_output_training', filename, output_dir_epoch, filename + '_keypoints.jpg')
                
                    cv2.imwrite(keypoint_output_dir, draw_image_with_kp)

                    # write_keypoints_to_file(keypoint_coords, epoch, keypoint_output_dir, pose_scores, keypoint_scores)

                    # draw_coordinates_to_image_file(appended_text, train_image_path, output_dir_epoch, output_stride, scale_factor, pose_scores, keypoint_scores, keypoint_coords, filenames[item_idx], include_displacements=False)

                    decoded_offsets = torch.from_numpy(decoded_offsets)
                    decoded_offsets = decoded_offsets.to('cuda')
                    
                    # print("decoded_offsets: ", decoded_offsets)
                    print("decoded_offsets shape: ", decoded_offsets.shape)

                    keypoint_coords = torch.from_numpy(keypoint_coords)
                    keypoint_coords = keypoint_coords.to('cuda')           
                    
                    print("offsets shape: ", offsets.shape)
                    

                    

                    loss, heatmap_loss, offset_loss, binary_target_heatmaps = criterion(train_heatmaps, ground_truth_heatmaps[item_idx] , ground_truth_keypoints[item_idx],  offsets, ground_truth_offsets[item_idx], max_num_poses=max_num_poses)
                    
                    print(train_heatmaps.shape)
                    print(binary_target_heatmaps.shape)
                    
                    
                    print("loss shape: ", loss.shape)

                    print("loss: ", loss)

                    print('[Train] Epoch [{}/{}], Batch [{}/{}], Item [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), item_idx+1, output[0].shape[0], loss.mean().item()))
                    
                    running_loss_value += loss.item()
                    heatmap_loss_value += heatmap_loss.item()
                    offset_loss_value += offset_loss.item()
                    
                    #todo change the loss
                    batch_loss += loss

                    # calculate accuracy 
                    matched_pairs = match_poses(keypoint_coords, ground_truth_keypoints[item_idx])
                    print("matched_pairs shape: ")
                    print(matched_pairs)
                
                    image_size = draw_image.shape[1]
                
                    oks = calculate_oks(matched_pairs, keypoint_coords, ground_truth_keypoints[item_idx], sigmas, variances, image_size)
                
                    thresholds = np.linspace(0.0, 10.0, num=50)
                    precisions = []
                    recalls = []
                    for i, threshold in enumerate(thresholds):
                        precision = calculate_precision(keypoint_coords, ground_truth_keypoints[item_idx], threshold)
                        print("precision: ", precision)
                        recall = calculate_recall(keypoint_coords, ground_truth_keypoints[item_idx], threshold)
                        print("recall: ", recall)
                        precisions.append(precision)
                        recalls.append(recall)

                    
                    # wandb.log({"epoch": epoch, f"precision_{i}": precision, f"recall_{i}": recall})
                
                    mAP = calculate_mAP(np.array(precisions), np.array(recalls))                    
                    batch_mAP += mAP
                    batch_oks += oks
            
                    
                batch_loss = batch_loss / len(train_loader)
                running_loss_value = running_loss_value / len(train_loader)
                offset_loss_value = offset_loss_value / len(train_loader)
                heatmap_loss_value = heatmap_loss_value / len(train_loader)


                if batch_idx % batch_checkpoint == batch_checkpoint-1:
                    step += 1
                    print("--in batch checkpoint--")
                    print("train_loss: ", running_loss_value / batch_checkpoint)
                    print("heatmap_loss: ", heatmap_loss_value / batch_checkpoint)
                    print("offset_loss: ", offset_loss_value / batch_checkpoint)
                    print("mAP: ", batch_mAP / batch_checkpoint)
                    print("oks: ", batch_oks / batch_checkpoint)
                    wandb.log({"train_loss": running_loss_value / batch_checkpoint , "heatmap_loss": heatmap_loss_value / batch_checkpoint, "offset_loss": offset_loss_value / batch_checkpoint, "mAP": batch_mAP / batch_checkpoint, "oks": batch_oks/batch_checkpoint, "epoch": epoch + ((batch_idx + 1)/len(train_loader))}, step=step)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss_value / batch_checkpoint))
                    running_loss_value = 0.0
                    heatmap_loss_value = 0.0
                    offset_loss_value = 0.0
                    

                print(type(precisions)) # should be <class 'list'>
                print(type(precisions[0])) # should be <class 'float'>
                print(type(recalls)) # should be <class 'list'>
                print(type(recalls[0])) # should be <class 'float'>
                
                
                

                
                batch_loss.backward()
                optimizer.step()
                
            # print("Updated Model weight norms:")
            # for name, param in model.named_parameters():
            #     print(name, param.data.norm())
        # Evaluate on test set
        model.eval()
        
        


        with torch.no_grad():
            print("--- with torch no grad ----")
            for batch_idx, (data, draw_image, output_scale, filenames, ground_truth_keypoints, ground_truth_heatmaps, ground_truth_offsets) in enumerate(test_loader):
                data.cuda()
                data_squeezed = data.squeeze()
                # data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
                output = model(data_squeezed)
                
                # print("**output[0] device: ", output[0].device)
                # print("**ground truth offsets device: ", ground_truth_offsets.device)
                
                # print("**output[0] shape: ", output[0].shape)
                
                
                #iterate through the batch size
                for item_idx, item in enumerate(output[0]):
                                        
                    offsets = output[1][item_idx]
                    displacements_fwd = output[2][item_idx]
                    displacements_bwd = output[3][item_idx]
                    
                    test_heatmaps = item
                    height = test_heatmaps.shape[1]
                    width = test_heatmaps.shape[2]
                    
                    # print("inside item_idx loop offsets shape: ", offsets.shape)
                    # print("item (heatmap) type: ", type(item))
                    
                    pose_scores, keypoint_scores, keypoint_coords, decoded_offsets = decode_pose_from_batch_item(epoch, test_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    
                    appended_text = "test_"
                    

                    draw_coordinates_to_image_file(appended_text, test_image_path, output_dir, output_stride, scale_factor, pose_scores,keypoint_scores, keypoint_coords, filenames[item_idx], include_displacements=False)

                    
                    decoded_offsets = torch.from_numpy(decoded_offsets)
                    decoded_offsets = decoded_offsets.to('cuda')
                    # print("decoded offsets device: ", decoded_offsets.device)
                    # print("ground truth offsets device: ", ground_truth_offsets[item_idx].device)
                    
                    keypoint_coords = torch.from_numpy(keypoint_coords)
                    keypoint_coords = keypoint_coords.to('cuda')
                    
                    # print("keypoint_coords device: ", keypoint_coords.device)
                    # print("ground_truth_keypoints[item_idx] device: ", ground_truth_keypoints[item_idx].device)
                    filename = filenames[item_idx]
                    loss, heatmap_loss, offset_loss, _ = criterion(test_heatmaps, ground_truth_heatmaps[item_idx], ground_truth_keypoints[item_idx], offsets, ground_truth_offsets[item_idx], max_num_poses=max_num_poses)
                    
                    save_heatmaps(test_heatmaps.detach().cpu().numpy(), filename, 0, num_keypoints=17, heatmaps_dir="pred_heatmaps_test", epoch=epoch)
                    
                    
                    
                    test_loss += loss.item()
                    print("-inside test-")
                    print("heatmap_loss: ", heatmap_loss.item())
                    print("offset_loss: ", offset_loss.item())
                    print("inside batch loss: ", loss.item())
                    
            test_loss /= len(test_loader.dataset)
            test_loss_value = test_loss.item()
            
            wandb.log({"test_loss": float(test_loss_value)}, step=step)
            print("test_loss_value: ", test_loss_value)
            print("step: ", step)
            
            # Check for improvement
            if test_loss_value < best_val_loss:
                best_val_loss = test_loss_value
                no_improve_epochs = 0
            
            # Save best model so far
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print("Stopping training due to lack of improvement in validation loss.")
                    break  # End training
            

        # Log epoch duration
        print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch+1, running_loss_value, test_loss_value))
                          
        epoch_duration = time.time() - epoch_start_time
        wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
        epoch_durations.append(epoch_duration)
        
    # Log average epoch duration
    avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
    wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
    
    print('Training Finished')

# Count number of people from ground truth keypoints)
def count_people(target_keypoints):
    """
    Given target keypoints of shape (num_people, num_keypoints, 2), returns the number of people.
    A person is considered to exist if at least one of their keypoints has a value other than -1.
    """
    num_people = 0
    for i in range(target_keypoints.shape[0]):
        if torch.any(target_keypoints[i] != -1):
            num_people += 1
    return num_people


def decode_pose_from_batch_item(epoch, image_path, filename, item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train):
    heatmaps = item
    

    # offsets_reshaped = offsets.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    
    if is_train:
        heatmaps = heatmaps.detach()
        offsets = offsets.detach()
        displacements_fwd = displacements_fwd.detach()
        displacements_bwd = displacements_bwd.detach()
        
    else:
        heatmaps = torch.tensor(heatmaps, requires_grad=is_train)
        
    # print("---- in decode pose from batch item --- ") 
    pose_scores, keypoint_scores, keypoint_coords, decoded_offsets = posenet.decode_multi.decode_multiple_poses(
                heatmaps,
                offsets,
                displacements_fwd,
                displacements_bwd,
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=score_threshold)

    
    # print("decoded offsets: ", decoded_offsets)
    
    # Find the indices of poses with scores above the threshold
    valid_indices = np.where(pose_scores >= score_threshold)[0]

    # Filter the pose_scores, keypoint_scores, and keypoint_coords using valid_indices
    pose_scores = pose_scores[valid_indices]
    keypoint_scores = keypoint_scores[valid_indices]
    keypoint_coords = keypoint_coords[valid_indices]
    decoded_offsets = decoded_offsets[valid_indices]
    
    # instance_keypoint_scores, instance_keypoint_coords, displacement_vectors = posenet.decode.decode_pose(root_score, root_id, root_image_coord, heatmaps, offsets_reshaped, output_stride, displacements_fwd_reshaped, displacements_bwd_reshaped)
    
    appended_text = "after_decode_"
    output_dir = "output_after_decode"

    draw_coordinates_to_image_file(appended_text, image_path, output_dir, output_stride, scale_factor, pose_scores, keypoint_scores, keypoint_coords, filename, displacements_fwd, displacements_bwd, include_displacements=True)
                    
    return pose_scores, keypoint_scores, keypoint_coords, decoded_offsets



def main():
    # Set up training parameters
    batch_size = 2
    learning_rate = 0.0001
    num_epochs = 100
    max_num_poses = 10

        
    config={
        "epochs": num_epochs,
         "batch_size": batch_size,
         "lr": learning_rate,
         }
    
    with wandb.init(project="posenet", config=config, name='PoseNet 101'):

        #instatiate model 
        model = posenet.load_model(args.model)
        model = model.cuda()
    
        for param in model.parameters():
            param.requires_grad = True
    
        output_stride = model.output_stride     

            
        plt = points_to_heatmap(4.5, 4.7, 21)
    
        # Define loss function and optimizer
        criterion = MultiPersonHeatmapOffsetAggregationLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    

        # Training loop
        train_image_path = args.train_image_dir
        test_image_path = args.test_image_dir
        output_dir = args.output_dir
        scale_factor = args.scale_factor
        ground_truth_keypoints_dir = "./keypoints_updated"
    
        is_train = True
    
        train_dataset = PosenetDatasetImage(train_image_path, ground_truth_keypoints_dir, scale_factor=1.0, output_stride=output_stride, train=True)
        test_dataset = PosenetDatasetImage(test_image_path, ground_truth_keypoints_dir, scale_factor=1.0, output_stride=output_stride, train=True)
        
        # when you have updated your dataset, print the mean and std and 
        # replace the Dataset normalization transforms  in class PosenetDatasetImage(Dataset) 
        # mean, std = get_dataset_mean_std(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride, train_image_path, train_image_path, output_dir, scale_factor, is_train)

        print('Setting up...')
        
        wandb.finish()


if __name__ == "__main__":
    main()

import torch
import numpy as np
import cv2
import os
from raft.raft import RAFT
from utils import load_flow_from_file, visualize_optical_flow, calculate_optical_flow_metrics
from collections import OrderedDict
import argparse


def compute_and_store_flow(image1_path, image2_path, model, device, output_path, original_size=(376, 1241)):

    frame1 = cv2.imread(image1_path)
    frame2 = cv2.imread(image2_path)

    if frame1 is None or frame2 is None:
        raise FileNotFoundError("One or both input images not found.")

    original_height, original_width = original_size
    target_size = (1280, 512)  # Resize for model inference
    scaling_factor_x = original_width / target_size[0]
    scaling_factor_y = original_height / target_size[1]

    frame1_resized = cv2.resize(frame1, target_size)
    frame2_resized = cv2.resize(frame2, target_size)

    # Convert images to tensors
    frame1_tensor = torch.from_numpy(frame1_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    frame2_tensor = torch.from_numpy(frame2_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        _, flow = model(frame1_tensor, frame2_tensor, iters=20, test_mode=True)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()

    # Resize back
    flow_resized = cv2.resize(flow, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    flow_resized[..., 0] *= scaling_factor_x
    flow_resized[..., 1] *= scaling_factor_y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, flow_resized)
    print(f"Optical flow saved to {output_path}")

    return flow_resized


# Define required arguments for RAFT
parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
args = parser.parse_args([])  # Empty list to avoid CLI arguments in code

# Initialize RAFT model
model = RAFT(args)
device = torch.device("cpu")
model = model.to(device)
model.eval()

# load model
state_dict = torch.load('models/raft/raft-kitti.pth', map_location=device)
if any(key.startswith("module.") for key in state_dict.keys()):
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)

image1_path = "KITTI_flow/image_0/000045_10.png"
image2_path = "KITTI_flow/image_0/000045_11.png"
output_flow_path = "raft_000045.npy"

flow_result = compute_and_store_flow(image1_path, image2_path, model, device, output_flow_path)

noc_path = "KITTI_flow/flow_noc/000045_10.png"
occ_path = "KITTI_flow/flow_occ/000045_10.png"
noc_gt, noc_val = load_flow_from_file(noc_path)
occ_gt, occ_val = load_flow_from_file(occ_path)

msen, pepn = calculate_optical_flow_metrics(flow_result, noc_gt, noc_val, plot=False)
print(f'NOC - MSEN: {msen}, PEPN: {pepn}')

msen, pepn = calculate_optical_flow_metrics(flow_result, occ_gt, occ_val, plot=False)
print(f'OCC - MSEN: {msen}, PEPN: {pepn}')

visualize_optical_flow(np.dstack((flow_result[..., 0], flow_result[..., 1], noc_val)))

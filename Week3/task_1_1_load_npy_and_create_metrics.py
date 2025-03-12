import numpy as np
from utils import load_flow_from_file, visualize_optical_flow, calculate_optical_flow_metrics


image1_path = "KITTI_flow/image_0/000045_10.png"
image2_path = "KITTI_flow/image_0/000045_11.png"
flow_path = "raft_000045.npy"
flow_path = "gmflow_000045.npy"

# load Ground Truth
noc_path = "KITTI_flow/flow_noc/000045_10.png"
occ_path = "KITTI_flow/flow_occ/000045_10.png"
noc_gt, noc_val = load_flow_from_file(noc_path)
occ_gt, occ_val = load_flow_from_file(occ_path)

# Load flow_result from .npy file
flow_result = np.load(flow_path)

# compute metrics
msen, pepn = calculate_optical_flow_metrics(flow_result, noc_gt, noc_val, plot=False)
print(f'NOC - MSEN: {msen}, PEPN: {pepn}')

msen, pepn = calculate_optical_flow_metrics(flow_result, occ_gt, occ_val, plot=False)
print(f'OCC - MSEN: {msen}, PEPN: {pepn}')

# Visualize flow
visualize_optical_flow(np.dstack((flow_result[..., 0], flow_result[..., 1], noc_val)))

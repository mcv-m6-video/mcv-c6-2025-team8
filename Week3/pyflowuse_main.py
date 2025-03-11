import pyflow
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_flow_from_file, visualize_optical_flow, calculate_optical_flow_metrics

img1_path = "/home/c5mcv08/C6/KITTI_flow/image_0/000045_10.png"
img2_path = "/home/c5mcv08/C6/KITTI_flow/image_0/000045_11.png"
noc_path = "/home/c5mcv08/C6/KITTI_flow/flow_noc/000045_10.png"
occ_path = "/home/c5mcv08/C6/KITTI_flow/flow_occ/000045_10.png"

class PyFlowEstimator(): # This class is taken from last years team3 code. It is based on pyflow's demo.py implementation of pyflow.
    def __init__(self, alpha = 0.012, ratio = 0.75, minWidth = 20, nOuterFPIterations = 7, nInnerFPIterations = 1, nSORIterations = 30, colType = 0):
        self.alpha = alpha
        self.ratio = ratio
        self.minWidth = minWidth
        self.nOuterFPIterations = nOuterFPIterations
        self.nInnerFPIterations = nInnerFPIterations
        self.nSORIterations = nSORIterations
        self.colType = colType # 0 equals RGB, 1 = Grey

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, 0)
        img = img[:,:,np.newaxis]

        img = img.astype(float) / 255.
        return img
        
    def estimate(self, img1_path, img2_path):
        img1 = self.preprocess_image(img1_path)
        img2 = self.preprocess_image(img2_path)
        
        u, v, im2W = pyflow.coarse2fine_flow(
                                        img1, 
                                        img2, 
                                        self.alpha,
                                        self.ratio,
                                        self.minWidth, 
                                        self.nOuterFPIterations,
                                        self.nInnerFPIterations,
                                        self.nSORIterations, 
                                        self.colType)
     
        flow = np.dstack((u, v))

        return flow

estimator = PyFlowEstimator()
flow = estimator.estimate(img1_path, img2_path)
print(flow)

noc_gt, noc_val = load_flow_from_file(noc_path)
occ_gt, occ_val = load_flow_from_file(occ_path)







msen, pepn = calculate_optical_flow_metrics(flow, noc_gt, noc_val, plot=True)

print(f'MSEN: {msen}')
print(f'PEPN: {pepn}')

msen, pepn = calculate_optical_flow_metrics(flow, occ_gt, occ_val, plot=True)

print(f'MSEN: {msen}')
print(f'PEPN: {pepn}')
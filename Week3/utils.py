import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_optical_flow(flow_map_path: str) -> np.ndarray:
    
    
    flow_image = cv2.imread(flow_map_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    
   
    validity_mask = flow_image[:, :, 0]
     
    u_component = (flow_image[:, :, 2] - 2**15) / 64  
    v_component = (flow_image[:, :, 1] - 2**15) / 64  
    
    u_component[validity_mask == 0] = 0
    v_component[validity_mask == 0] = 0
    
    # Stack components into an output array (H, W, 3)
    return np.dstack((u_component, v_component, validity_mask))


def load_flow_from_file(file_path: str):
    
    raw_flow = load_optical_flow(file_path)

    u_flow = raw_flow[:, :, 0]
    v_flow = raw_flow[:, :, 1]

    validity_mask = raw_flow[:, :, 2]

    flow_field = np.stack((u_flow, v_flow), axis=2)

    return flow_field, validity_mask


def plot_error(error_map: np.ndarray, filename: str, save_dir: str = "./results"):
   
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(9, 3))
    plt.imshow(error_map, cmap='gray')
    
    plt.axis('off')

    output_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  



def plot_validity_mask(validity_mask: np.ndarray, filename: str, save_dir: str = "./results"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 9))
    plt.title(f"Valid Pixels for {filename}")
    plt.imshow(validity_mask, cmap='gray')
    
    output_path = os.path.join(save_dir, f"{filename}_valid_pixels_GT.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() 


def plot_error_histogram(error_map: np.ndarray, filename: str, save_dir: str = "./plots"):

    os.makedirs(save_dir, exist_ok=True)

    histogram, bin_edges = np.histogram(error_map, bins=50, density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.title(f"Squared Error Histogram for {filename}")
    plt.xlabel("Squared Error")
    plt.ylabel("Pixel Percentage")
    plt.xlim([0, 50])

    plt.bar(bin_centers, histogram, width=bin_edges[1] - bin_edges[0])

    output_path = os.path.join(save_dir, f"{filename}_error_histogram.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def calculate_optical_flow_metrics(pred_flow: np.ndarray, gt_flow: np.ndarray, validity_mask: np.ndarray, plot: bool = False, output_file: str = "metrics"):
    
    u_gt, v_gt = gt_flow[:, :, 0], gt_flow[:, :, 1]
    u_pred, v_pred = pred_flow[:, :, 0], pred_flow[:, :, 1]

    motion_vectors = np.sqrt(np.square(u_pred - u_gt) + np.square(v_pred - v_gt))

    if plot:
        plot_error_histogram(motion_vectors[validity_mask == 1], output_file)

    erroneous_pixels = (motion_vectors[validity_mask == 1] > 3).sum()

    msen = np.mean(motion_vectors[validity_mask == 1])

    pepn = (erroneous_pixels / np.sum(validity_mask == 1)) * 100  

    return msen, pepn


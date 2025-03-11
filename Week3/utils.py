import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_optical_flow(flow_map_path: str) -> np.ndarray:
    """
    Load an optical flow map from a 3-channel uint16 PNG image.

    The input image is expected to have:
    - Channel 1 (Red): Validity mask (1 for valid flow, 0 otherwise).
    - Channel 2 (Green): v-component of the optical flow.
    - Channel 3 (Blue): u-component of the optical flow.

    The function decodes the flow map and normalizes the flow vectors.

    Args:
        flow_map_path (str): Path to the optical flow image.

    Returns:
        np.ndarray: Optical flow tensor of shape (H, W, 3), where the last
                    dimension contains (u-flow, v-flow, validity mask).
    """
    # Read image as uint16 and convert to float64
    flow_image = cv2.imread(flow_map_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    
    # Extract validity mask (first channel)
    validity_mask = flow_image[:, :, 0]
    
    # Decode flow components
    u_component = (flow_image[:, :, 2] - 2**15) / 64  # Blue channel
    v_component = (flow_image[:, :, 1] - 2**15) / 64  # Green channel
    
    # Zero out invalid flow values
    u_component[validity_mask == 0] = 0
    v_component[validity_mask == 0] = 0
    
    # Stack components into an output array (H, W, 3)
    return np.dstack((u_component, v_component, validity_mask))


def load_flow_from_file(file_path: str):
    """
    Load optical flow data and its validity mask from a file.

    This function reads an optical flow map, extracts the flow components
    (u, v), and returns them along with the validity mask.

    Args:
        file_path (str): Path to the optical flow file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - flow_field (np.ndarray): Optical flow array of shape (H, W, 2),
              where the last dimension contains (u-flow, v-flow).
            - validity_mask (np.ndarray): Binary mask (H, W) indicating valid pixels.
    """
    # Load raw optical flow map
    raw_flow = load_optical_flow(file_path)

    # Extract u-flow and v-flow components
    u_flow = raw_flow[:, :, 0]
    v_flow = raw_flow[:, :, 1]

    # Extract validity mask
    validity_mask = raw_flow[:, :, 2]

    # Stack flow components into a single array
    flow_field = np.stack((u_flow, v_flow), axis=2)

    return flow_field, validity_mask


def visualize_optical_flow(flow_data: np.ndarray):
    """
    Visualize optical flow components (u, v) and the validity mask.

    This function plots three grayscale images:
    - u-flow (horizontal flow component).
    - v-flow (vertical flow component).
    - Validity mask (indicating valid flow pixels).

    Args:
        flow_data (np.ndarray): Optical flow array of shape (H, W, 3), where:
            - flow_data[:, :, 0] contains the u-component.
            - flow_data[:, :, 1] contains the v-component.
            - flow_data[:, :, 2] contains the validity mask.
    """
    # Define figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    # Define image components and titles
    flow_components = [flow_data[:, :, 0], flow_data[:, :, 1], flow_data[:, :, 2]]
    titles = ['Horizontal Flow (u)', 'Vertical Flow (v)', 'Validity Mask']

    # Plot each component
    for ax, flow_component, title in zip(axes, flow_components, titles):
        ax.imshow(flow_component, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')  # Hide axis for better visualization

    plt.tight_layout()
    plt.show()


def plot_error(error_map: np.ndarray, filename: str, save_dir: str = "./results"):
    """
    Save a visualization of the squared error as a color-mapped image.

    Args:
        error_map (np.ndarray): 2D array representing the error values.
        filename (str): Name of the output file (without extension).
        save_dir (str, optional): Directory where the plot will be saved. Defaults to "./plots".

    Saves:
        A PNG image of the error map in the specified directory.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(9, 3))
    plt.title(f'Squared Error for {filename}')
    plt.imshow(error_map, cmap='viridis')  # Use a perceptually uniform colormap
    plt.colorbar()
    
    # Save the figure
    output_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory


def plot_validity_mask(validity_mask: np.ndarray, filename: str, save_dir: str = "./results"):
    """
    Save a visualization of the validity mask as a grayscale image.

    Args:
        validity_mask (np.ndarray): 2D binary mask indicating valid pixels (H, W).
        filename (str): Name of the output file (without extension).
        save_dir (str, optional): Directory where the plot will be saved. Defaults to "./plots".

    Saves:
        A PNG image of the validity mask in the specified directory.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 9))
    plt.title(f"Valid Pixels for {filename}")
    plt.imshow(validity_mask, cmap='gray')
    
    # Save the figure
    output_path = os.path.join(save_dir, f"{filename}_valid_pixels_GT.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory


def plot_error_histogram(error_map: np.ndarray, filename: str, save_dir: str = "./results"):
    """
    Save a histogram of the squared errors from an error map.

    Args:
        error_map (np.ndarray): 2D array representing squared error values.
        filename (str): Name of the output file (without extension).
        save_dir (str, optional): Directory where the histogram will be saved. Defaults to "./plots".

    Saves:
        A PNG image of the error histogram in the specified directory.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Compute histogram
    histogram, bin_edges = np.histogram(error_map, bins=50, density=True)
    
    # Calculate the bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Squared Error Histogram for {filename}")
    plt.xlabel("Squared Error")
    plt.ylabel("Pixel Percentage")
    plt.xlim([0, 50])

    # Plot the histogram as a bar chart
    plt.bar(bin_centers, histogram, width=bin_edges[1] - bin_edges[0])

    # Save the plot
    output_path = os.path.join(save_dir, f"{filename}_error_histogram.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory


def calculate_optical_flow_metrics(pred_flow: np.ndarray, gt_flow: np.ndarray, validity_mask: np.ndarray, plot: bool = False, output_file: str = "metrics"):
    """
    Calculate the Mean Squared Error (MSE) and Percentage of Erroneous Pixels (PEPN) in non-occluded areas.

    This function compares predicted optical flow with ground truth flow, using a validity mask to 
    exclude occluded or invalid pixels.

    Args:
        pred_flow (np.ndarray): Predicted optical flow (H, W, 2), where the last dimension contains (u, v) components.
        gt_flow (np.ndarray): Ground truth optical flow (H, W, 2), where the last dimension contains (u, v) components.
        validity_mask (np.ndarray): Binary mask indicating valid pixels (H, W).
        plot (bool, optional): If True, plot the error map, validity mask, and error histogram. Defaults to False.
        output_file (str, optional): Filename prefix for saving the results. Defaults to "metrics".

    Returns:
        tuple: A tuple containing:
            - msen (float): Mean Squared Error in non-occluded areas.
            - pepn (float): Percentage of Erroneous Pixels in non-occluded areas.
    """
    # Extract flow components (u, v)
    u_gt, v_gt = gt_flow[:, :, 0], gt_flow[:, :, 1]
    u_pred, v_pred = pred_flow[:, :, 0], pred_flow[:, :, 1]

    # Compute the motion vectors (Euclidean distance between predicted and ground truth flow)
    motion_vectors = np.sqrt(np.square(u_pred - u_gt) + np.square(v_pred - v_gt))

    if plot:
        # Plot error map, validity mask, and error histogram
        plot_error(motion_vectors, output_file)
        plot_validity_mask(validity_mask, output_file)
        plot_error_histogram(motion_vectors[validity_mask == 1], output_file)

    # Calculate erroneous pixels (motion vector > 3 for valid pixels)
    erroneous_pixels = (motion_vectors[validity_mask == 1] > 3).sum()

    # Calculate Mean Squared Error in non-occluded areas (valid pixels)
    msen = np.mean(motion_vectors[validity_mask == 1])

    # Calculate Percentage of Erroneous Pixels in non-occluded areas
    pepn = (erroneous_pixels / np.sum(validity_mask == 1)) * 100  # Erroneous pixels / total valid pixels

    return msen, pepn


def convert_flow_to_color(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to a color image for visualization.

    The optical flow is represented as a 2D vector field (u, v), and the function
    converts it to a color image using the HSV color space, where:
    - Hue represents the flow direction (angle).
    - Saturation is set to the maximum value (255).
    - Value (brightness) represents the flow magnitude.

    Args:
        flow (np.ndarray): Optical flow array of shape (H, W, 2), where:
            - flow[..., 0] is the horizontal flow component (u).
            - flow[..., 1] is the vertical flow component (v).

    Returns:
        np.ndarray: Color-coded optical flow image in BGR format.
    """
    # Get the height and width of the flow field
    height, width, _ = flow.shape

    # Initialize an HSV image where:
    # - Hue (hue channel) will be calculated from flow angle.
    # - Saturation (saturation channel) is set to maximum (255).
    # - Value (value channel) will be normalized magnitude of the flow.
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Set saturation to max

    # Compute the flow magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Map the angle to the hue channel (scale angle from radians to degrees and then normalize to [0, 180])
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Normalize the magnitude to [0, 255] for the value channel
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the HSV image to BGR for visualization
    bgr_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr_image
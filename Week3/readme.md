# Multi-Target Tracking with Optical Flow  

This project focuses on integrating optical flow techniques into multi-target single-camera (MTSC) tracking. The workflow includes evaluating optical flow estimation methods, improving object tracking using optical flow, and applying MTSC tracking to real-world datasets.  

## Project Structure  

### 1. Optical Flow  

#### 1.1 Optical Flow Estimation with Off-the-Shelf Methods  
- Utilize **PyFlow** for optical flow estimation.  
  - Dependency & credit: [PyFlow](https://github.com/pathak22/pyflow)  
- Compare PyFlow results with two state-of-the-art (SOTA) optical flow methods.  
- Evaluate performance using sequence 45 (image_0) from the **KITTI Flow 2012 dataset**.  
  - Dataset credit: [KITTI](http://www.cvlibs.net/datasets/kitti/)  

#### 1.2 Improving Object Trackers with Optical Flow  
- Enhance last week's object trackers using optical flow.  
- Experiment with integrating optical flow into:  
  - **IOU-based trackers**  
  - **Kalman filter-based trackers**  

### 2. Multi-Target Single-Camera (MTSC) Tracking  

#### 2.1 MTSC Tracking on AI City Challenge Dataset (Sequence S01)  
- Implement MTSC tracking on sequence **S01** of the AI City Challenge dataset.  
- Dataset credit: [AI City Challenge 2022](https://www.aicitychallenge.org/2022-data-and-evaluation/)  

#### 2.2 MTSC Tracking on Additional AI City Sequences  
- Extend MTSC tracking to sequences **S03** and **S04** of the AI City dataset.  

## Acknowledgments  

We acknowledge and credit the following datasets and tools used in this project:  
- **PyFlow**: [https://github.com/pathak22/pyflow](https://github.com/pathak22/pyflow)  
- **KITTI Flow 2012 dataset**: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)  
- **AI City Challenge Dataset**: [https://www.aicitychallenge.org/2022-data-and-evaluation/](https://www.aicitychallenge.org/2022-data-and-evaluation/)  



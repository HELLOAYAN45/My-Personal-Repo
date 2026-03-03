NEUROSCAN AI: MULTIMODAL BRAIN TUMOR SEGMENTATION & CLINICAL ASSISTANT
======================================================================

PROJECT OVERVIEW
----------------
NeuroScan AI is an end-to-end, multimodal enterprise medical dashboard. It bridges deep learning computer vision, Explainable AI (XAI), and local Large Language Models (LLMs) to detect, map, and clinically summarize intracranial anomalies from 2D axial MRI scans.

This system moves beyond basic "black-box" binary classification. It utilizes a ResNet34 + U-Net hybrid architecture to draw pixel-perfect boundary masks, generates Grad-CAM thermal heatmaps for clinical transparency, and orchestrates a local Llama 3 LLM to provide contextualized, quantitative clinical summaries.

KEY FEATURES
------------
* Semantic Segmentation: Accurately maps tumor boundaries and calculates total area/volume using a ResNet34-powered U-Net.
* Explainable AI (Grad-CAM): Generates real-time thermal heatmaps highlighting the exact neural gradients and textures the AI prioritized for its prediction.
* Multimodal LLM Integration: Automatically pipes geometric telemetry (pixel count, volume %, confidence score) into a local Llama 3 engine to stream a highly professional clinical summary.
* Dynamic Noise Filtering: Includes an interactive UI threshold slider to mathematically filter out jagged edge artifacts caused by MRI "skull-stripping" preprocessing.


SYSTEM ARCHITECTURE & ENGINEERING
---------------------------------

1. The Vision Engine (ResNet34 + U-Net)
The segmentation model relies on Transfer Learning. The standard U-Net encoder was replaced with a pre-trained ResNet34 architecture. This prevents the vanishing gradient problem and leverages ImageNet-trained convolutional filters to instantly recognize complex geometric boundaries and textures.

2. Data Pipeline & Optimization
Trained on the BraTS (Brain Tumor Segmentation) dataset, the data pipeline was engineered to handle massive 3D NIfTI (.nii) volumes without triggering GPU Out-Of-Memory (OOM) errors:
* Dynamic Slicing: A custom Python generator slices the 3D cubes along the Z-axis, dropping empty scans and saving viable cross-sections as lightweight 2D .npy arrays.
* Batch Processing: Images are fed to the GPU in strict batches to maintain a flat, highly optimized memory footprint.

3. Custom Mathematical Loss Function
To combat extreme class imbalance (where healthy black pixels severely outnumber tumor pixels), the model was trained using a custom hybrid loss function combining Binary Cross-Entropy (BCE) and the Jaccard Index (IoU):

$$\mathcal{L}_{Total} = \mathcal{L}_{BCE} + (1 - J(A,B))$$

This forces the network to heavily penalize false positives while strictly mapping the chaotic borders of the anomaly.


HOW TO RUN FROM SCRATCH (INSTALLATION GUIDE)
--------------------------------------------
Follow these steps to deploy the multimodal dashboard on your local Windows/Linux machine.

PREREQUISITES
* Python 3.9
* A dedicated GPU (Optional but recommended for faster vision inference)
* At least 8GB of System RAM (for the local LLM)

STEP 1: CLONE THE REPOSITORY
git clone https://github.com/YourUsername/NeuroScan-AI.git
cd NeuroScan-AI
# 🧠 NeuroScan AI  
### Multimodal Brain Tumor Segmentation & Clinical Assistant

NeuroScan AI is an end-to-end medical intelligence dashboard that integrates **Deep Learning Computer Vision**, **Explainable AI (XAI)**, and **Local Large Language Models (LLMs)** to detect, segment, and clinically summarize intracranial anomalies from 2D axial MRI scans.

This system moves beyond black-box classification. It performs **pixel-accurate tumor segmentation**, generates **Grad-CAM interpretability maps**, and produces **quantitative clinical summaries** using a locally deployed Llama 3 model.

---

## 🚀 Key Features

### 🔬 Semantic Segmentation  
- ResNet34-powered U-Net hybrid architecture  
- Pixel-level tumor boundary detection  
- Automated tumor area & volume estimation  

### 🔥 Explainable AI (Grad-CAM)  
- Real-time heatmaps  
- Visualizes model attention regions  
- Improves clinical transparency  

### 🤖 Multimodal LLM Integration  
- Streams geometric telemetry to local Llama 3  
- Generates structured, professional clinical summaries  
- Converts numerical metrics into contextual explanations  

### 🎛 Dynamic Noise Filtering  
- Interactive threshold slider  
- Removes skull-stripping artifacts  
- Improves segmentation clarity  

---

# 🏗 System Architecture

## 1️⃣ Vision Engine — ResNet34 + U-Net

The encoder of U-Net is replaced with a pretrained **ResNet34** backbone:

- Prevents vanishing gradients  
- Leverages ImageNet-trained filters  
- Improves boundary precision  
- Enhances texture recognition  

---

## 2️⃣ Data Pipeline & Optimization

Trained on the **BraTS (Brain Tumor Segmentation) Dataset**.

### Memory-Efficient Processing

- **Dynamic Z-Axis Slicing**  
  Converts heavy 3D `.nii` volumes into optimized 2D `.npy` slices  
  Automatically drops empty scans  

- **Strict Batch Processing**  
  Prevents GPU Out-Of-Memory (OOM) errors  
  Maintains consistent memory footprint  

---

## 3️⃣ Custom Hybrid Loss Function

To combat severe class imbalance:

\[
\mathcal{L}_{Total} = \mathcal{L}_{BCE} + (1 - J(A,B))
\]

Where:

- **BCE** → Penalizes pixel misclassification  
- **J(A,B)** → Jaccard Index (IoU)  
- Encourages strict boundary alignment  
- Reduces false positives  

---

# 🖥 Dashboard Interface

The deployed application features a custom-styled dashboard:

- Original MRI slice  
- Grad-CAM heatmap  
- Segmented tumor overlay  
- AI Confidence Score  
- Tumor Area  
- Brain Volume Occupied (%)  

---

# ⚙️ Installation Guide

## 📌 Prerequisites

- Python 3.9  
- 8GB+ RAM (for local LLM)  
- GPU recommended (for faster inference)  

---

## Step 1 — Clone Repository

```bash
git clone https://github.com/YourUsername/NeuroScan-AI.git
cd NeuroScan-AI

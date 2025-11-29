Hybrid-DeepCoro: Research-Grade Coronary Artery Segmentation & QCA Pipeline
Deep Learning for Automated Coronary Imaging Analysis
EfficientNet-B6 U-Net++ | SCSE Attention | Multi-Loss Optimization | Surgical-Path Planning

📌 Overview
Hybrid-DeepCoro is a fully reproducible, research-grade deep-learning pipeline for:
  *Coronary artery segmentation from angiographic images
  *Automated quantitative coronary analysis (QCA)
  *Detection of stenosis severity
  *Surgical planning, including graft landing zone estimation, angle analysis, and path measurement
The system integrates a highly optimized U-Net++ architecture with EfficientNet-B6 (Noisy-Student) encoder and SCSE attention, achieving precise vessel extraction and robust stenosis assessment.
This repository is designed for academic use, clinical AI research, and cardiovascular imaging labs, and is suitable for sharing with collaborators and faculty in leading medical AI programs worldwide.

⭐ Key Features
🧠 1. Deep Learning Model
    *U-Net++ architecture
    *EfficientNet-B6 backbone (Noisy-Student pre-trained)
    *SCSE (Concurrent Spatial & Channel Squeeze-Excitation) attention
    *Multiclass segmentation (background + stenosis)

📚 2. Multi-Loss Optimization
    *Combined loss for high accuracy on small vascular structures:
    *Tversky Loss (α=0.7, β=0.3)
    *Focal Loss (γ=2.0)
    *Dice Loss
    *Weighted final loss: 0.4*Tversky + 0.4*Focal + 0.2*Dice

🗂️ 3. Automatic Dataset Scanner
    *Automatically detects compatible datasets on:
    */kaggle/input
    */content (Google Colab)
    Supports both:
    *Stenosis datasets
    *Coronary syntax datasets

🎛️ 4. Augmentation Pipeline (Albumentations)
    *CLAHE
    *Elastic/Grid/Optical Distortion
    *Random Brightness & Contrast
    *Multi-flip + Rotation
    *Normalization
    *Tensor conversion

📉 5. Training Engine
    *AMP training (mixed precision)
    *AdamW optimizer
    *CosineAnnealingWarmRestarts scheduler
    *Aggregated validation metrics for stenosis (IoU, F1, Accuracy, Sensitivity)

🔬 6. QCA Module (Quantitative Coronary Analysis)
    Extracts:
    *Minimum lumen diameter
    *Reference diameter
    *Stenosis percentage
    *Vessel centerline
    *Adaptive graft landing zone
    *Angle estimation
    
🧪 7. Final Research Report Generator
    Runs inference on test images and produces:
    *CSV report
    *Matplotlib visualizations
    *Surgical path & stenosis annotations
    
🧬 Model Architecture
EfficientNet-B6 Encoder (Noisy Student)
        ↓
SCSE Attention Blocks
        ↓
U-Net++ Nested Decoder
        ↓
Segmentation Mask (2 classes)

🛠️ Installation
1. Clone the Repository
    git clone https://github.com/<your-username>/Hybrid-DeepCoro.git
    cd Hybrid-DeepCoro
2. Install Dependencies
    pip install -r requirements.txt

🚀 Training
  python train.py
    The script will:
    *Auto-detect datasets
    *Apply augmentations
    *Train Hybrid-DeepCoro
    *Save the best model
    *Generate QCA report

🧪 Inference Example
model.eval()
pred = model(image_tensor)[0]
mask = torch.argmax(pred, dim=0)
    *Graft length prediction (mm)

🖼️ Sample Visualization
Overlay of image + segmentation mask
Stenosis center marked (red ×)
Graft landing point (green ✳️)
Estimated path (spline)

📈 Performance Metrics
IoU for stenosis region
F1-score (Dice)
Accuracy
Sensitivity

👩‍⚕️ Author & Contact
For academic collaboration, questions, or manuscript-related inquiries:
Arash Amadeh Taheri, BCs
Email: amadehtaheria@gmail.com

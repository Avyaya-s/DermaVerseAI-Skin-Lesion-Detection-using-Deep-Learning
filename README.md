Hierarchical Skin Lesion Analysis Pipeline:

      This repository contains the deployment-ready implementation of a segmentation-guided hierarchical deep learning pipeline for automated skin lesion analysis, as presented in our IEEE paper.
      
      The system performs:
      
            (1)Lesion segmentation
            (2)Binary classification (Benign vs Malignant)
            (3)Fine-grained multiclass classification of malignant lesions
      
      The pipeline is designed to reflect real-world clinical diagnostic workflows rather than maximize isolated model accuracy.

Project Highlights

(1)Hierarchical (cascaded) design inspired by clinical decision-making
(2)Segmentation-guided inference using ROI extraction
(3)Binary gating to control class imbalance
(4)End-to-end pipeline evaluation (not isolated models)
(5)Deployment-ready Flask backend with API support
(5)Compatible with frontend integration and ngrok-based exposure

Repo Structure


System Requirements

    (1)Hardware:
    
        (i)  NVIDIA GPU recommended (tested on RTX 3050 / 3070 / 4050)
        (ii) Minimum 16 GB RAM
    
    (2)Software
    
        (i)  Python 3.9+
        (ii) CUDA-compatible GPU drivers (if using GPU)

Model Training and Weights

    All models used in this project were trained from scratch by the authors using the ISIC 2018 datasets,
    rather than directly deploying off-the-shelf pretrained solutions.
    
    Training Details:
    
        Segmentation Model:
        
            Architecture: U²-NetP
            
            Dataset: ISIC 2018 Task-1 (Lesion Segmentation)
            
            Purpose: Accurate lesion boundary detection and ROI localization
        
        Binary Classification Model (Benign vs Malignant):
        
            Architecture: EfficientNet-B0
            Dataset: ISIC 2018 Task-3
            Purpose: Malignancy screening and hierarchical gating
        
        Multiclass Malignant Classification Model:
        
            Architecture: EfficientNet-B0
            Dataset: ISIC 2018 Task-3 (Malignant classes only)
            Classes: Melanoma (MEL), Basal Cell Carcinoma (BCC), Actinic Keratosis (AKIEC)
        
        All training was conducted using strict train–test separation, and no test images were used during training or validation.
    
    Model Weights Availability:  Due to repository size limitations, trained model weights are not included in this repository.
    
  Trained weights can be provided upon request or shared via external storage (e.g., Google Drive / HuggingFace).

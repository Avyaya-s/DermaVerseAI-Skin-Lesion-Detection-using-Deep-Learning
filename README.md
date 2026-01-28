🩺 Hierarchical Skin Lesion Analysis Pipeline

This repository contains a deployment-ready implementation of a segmentation-guided hierarchical deep learning pipeline for automated skin lesion analysis, as presented in our IEEE paper.

The system performs:

Lesion Segmentation

Binary Classification (Benign vs Malignant)

*Fine-Grained Multiclass Classification of Malignant Lesions

The pipeline is designed to reflect real-world clinical diagnostic workflows rather than optimizing isolated model accuracy.

🚀 Project Highlights

✅ Hierarchical (cascaded) design inspired by clinical decision-making

✅ Segmentation-guided inference using ROI extraction

✅ Binary gating to control class imbalance

✅ End-to-end pipeline evaluation (not isolated models)

✅ Deployment-ready Flask backend with API support

✅ Compatible with frontend integration and ngrok-based exposure

📁 Repository Structure
Skin-Lesion-Hierarchical-Pipeline/ <br>
│<br>
├── deployment/                  # Final system used in the paper<br>
│   ├── app.py                   # Flask API entry point<br>
│   ├── pipeline_service.py      # End-to-end inference pipeline<br>
│   ├── infer/                   # Segmentation & classification services<br>
│   ├── utils/                   # ROI extraction utilities<br>
│   ├── preprocess/              # Image preprocessing<br>
│   ├── models/                  # Model definitions (weights excluded)<br>
│   ├── uploads/                 # Runtime upload directory<br>
│   ├── temp/                    # Temporary mask/ROI storage<br>
│   └── requirements.txt<br>
│
├── experiments/ (optional)      # Local experiments / notebooks <br>
└── README.md<br>

🖥️ System Requirements
Hardware:   NVIDIA GPU recommended (tested on RTX 3050 / 3070 / 4050)
            Minimum 16 GB RAM

Software:   Python 3.9+
            CUDA-compatible GPU drivers (if using GPU)

🧠 Model Training and Weights: All models were trained from scratch by the authors using the ISIC 2018 datasets. 
                                No off-the-shelf pretrained models were deployed.

🔹 Training Details
        1. Segmentation Model: Architecture: U²-NetP
                               Dataset: ISIC 2018 Task-1 (Lesion Segmentation)
                               Purpose: Accurate lesion boundary detection and ROI localization
        2. Binary Classification Model (Benign vs Malignant): Architecture: EfficientNet-B0
                                                              Dataset: ISIC 2018 Task-3
                                                              Purpose: Malignancy screening and hierarchical gating
        3. Multiclass Malignant Classification Model: Architecture: EfficientNet-B0
                                                      Dataset: ISIC 2018 Task-3 (Malignant classes only)
                                                      Classes:
                                                          Melanoma (MEL)
                                                          Basal Cell Carcinoma (BCC)
                                                          Actinic Keratosis (AKIEC)
✔️ Strict train–test separation was followed.
✔️ No test images were used during training or validation.

📦 Model Weights Availability
    Due to repository size limitations, trained weights are not included.
    Weights can be provided upon request or shared via:
                                                        Google Drive
                                                        HuggingFace
                                                        Other external storage

⚙️ Setup Instructions
      1️⃣ Clone the Repository
            git clone https://github.com/Avyaya-s/DermaVerseAI-Skin-Lesion-Detection-using-Deep-Learning.git
            cd DermaVerseAI-Skin-Lesion-Detection-using-Deep-Learning
      2️⃣ Create Virtual Environment (Recommended)
            python -m venv venv
            Activate:
                # Linux / Mac
                source venv/bin/activate
                # Windows
                venv\Scripts\activate
      3️⃣ Install Dependencies
            pip install -r deployment/requirements.txt
      ▶️ Running the Backend Server
            From the project root:
                python deployment/app.py
            If successful:
                Running on http://127.0.0.1:5000
🔌 API Endpoints
      ✅ Health Check
      GET /health
          Response
          {
            "status": "ok"
          }
      🖼️ Image Prediction
          POST /predict
          Form-data
          image : input dermoscopic image
          Response
          {
            "binary_prediction": "Malignant",
            "binary_probability": 0.82,
            "final_prediction": "MEL",
            "multiclass_probabilities": {
              "MEL": 0.63,
              "BCC": 0.21,
              "AKIEC": 0.16
            }
          }
🌐 Frontend & ngrok Integration (Optional)
      To expose the backend publicly:
      ngrok http 5000
      Use the generated:
      https://xxxx.ngrok-free.app
      URL in your frontend API calls.
📊 Dataset & Evaluation
      Dataset: ISIC 2018 Task-3
      Test Set Size: 1,512 images
      Evaluation: Performed only on unseen test data
      Test-Time Augmentation: None
      Data Leakage: None
      Reported results reflect pipeline-level performance, not isolated model metrics.

🎯 Design Philosophy
      Unlike single-stage classifiers, this work prioritizes:
          🛡️ Clinical safety (high malignant recall)
          🔍 Interpretability
          🧭 Error traceability
          🚀 Deployment realism

Accuracy trade-offs are explicitly analyzed via cascade error analysis.

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

🖥️ System Requirements<br>
Hardware:   NVIDIA GPU recommended (tested on RTX 3050 / 3070 / 4050)<br>
            Minimum 16 GB RAM<br>

Software:   Python 3.9+<br>
            CUDA-compatible GPU drivers (if using GPU)<br>

🧠 Model Training and Weights: All models were trained from scratch by the authors using the ISIC 2018 datasets. <br>
                                No off-the-shelf pretrained models were deployed.<br>

🔹 Training Details<br>
        1. Segmentation Model: Architecture: U²-NetP<br>
                               Dataset: ISIC 2018 Task-1 (Lesion Segmentation)<br>
                               Purpose: Accurate lesion boundary detection and ROI localization<br>
        2. Binary Classification Model (Benign vs Malignant): Architecture: EfficientNet-B0<br>
                                                              Dataset: ISIC 2018 Task-3<br>
                                                              Purpose: Malignancy screening and hierarchical gating<br>
        3. Multiclass Malignant Classification Model: Architecture: EfficientNet-B0<br>
                                                      Dataset: ISIC 2018 Task-3 (Malignant classes only)<br>
                                                      Classes:<br>
                                                          Melanoma (MEL)<br>
                                                          Basal Cell Carcinoma (BCC)<br>
                                                          Actinic Keratosis (AKIEC)<br>
✔️ Strict train–test separation was followed.<br>
✔️ No test images were used during training or validation.<br>

📦 Model Weights Availability<br>
    Due to repository size limitations, trained weights are not included.<br>
    Weights can be provided upon request or shared via:<br>
                                                        Google Drive<br>
                                                        HuggingFace<br>
                                                        Other external storage<br>

⚙️ Setup Instructions<br>
      1️⃣ Clone the Repository<br>
            git clone https://github.com/Avyaya-s/DermaVerseAI-Skin-Lesion-Detection-using-Deep-Learning.git<br>
            cd DermaVerseAI-Skin-Lesion-Detection-using-Deep-Learning<br>
      2️⃣ Create Virtual Environment (Recommended)<br>
            python -m venv venv<br>
            Activate:<br>
                # Linux / Mac<br>
                source venv/bin/activate<br>
                # Windows<br>
                venv\Scripts\activate<br>
      3️⃣ Install Dependencies<br>
            pip install -r deployment/requirements.txt<br>
      ▶️ Running the Backend Server<br>
            From the project root:<br>
                python deployment/app.py<br>
            If successful:<br>
                Running on http://127.0.0.1:5000<br>
🔌 API Endpoints<br>
      ✅ Health Check<br>
      GET /health<br>
          Response<br>
          {
            "status": "ok"
          }<br>
      🖼️ Image Prediction<br>
          POST /predict<br>
          Form-data<br>
          image : input dermoscopic image<br>
          Response<br>
          {
            "binary_prediction": "Malignant",<br>
            "binary_probability": 0.82,<br>
            "final_prediction": "MEL",<br>
            "multiclass_probabilities": {<br>
              "MEL": 0.63,<br>
              "BCC": 0.21,<br>
              "AKIEC": 0.16<br>
            }<br>
          }<br>
🌐 Frontend & ngrok Integration (Optional)<br>
      To expose the backend publicly:<br>
      ngrok http 5000<br>
      Use the generated:
      https://xxxx.ngrok-free.app<br>
      URL in your frontend API calls.<br>
📊 Dataset & Evaluation<br>
      Dataset: ISIC 2018 Task-3<br>
      Test Set Size: 1,512 images<br>
      Evaluation: Performed only on unseen test data<br>
      Test-Time Augmentation: None<br>
      Data Leakage: None<br>
      Reported results reflect pipeline-level performance, not isolated model metrics.<br>

🎯 Design Philosophy<br>
      Unlike single-stage classifiers, this work prioritizes:<br>
          🛡️ Clinical safety (high malignant recall)<br>
          🔍 Interpretability<br>
          🧭 Error traceability<br>
          🚀 Deployment realism<br>

Accuracy trade-offs are explicitly analyzed via cascade error analysis.

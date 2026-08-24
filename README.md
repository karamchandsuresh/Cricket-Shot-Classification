# 🏏 Cricket Shot Classification and Analytics

A deep learning-based web application for **cricket batting shot classification and pose analytics** using **MobileNetV2, GRU, MediaPipe, Flask, React, and Vite**.

The system accepts a cricket batting video, extracts spatial features from video frames using MobileNetV2, analyzes temporal information using a GRU network, predicts the cricket shot class, and performs pose analysis using MediaPipe.

---

## 🌐 Live Demo

**Frontend:**  
https://cricket-shot-classification.vercel.app

> **Deployment Note:**  
> The frontend is deployed on Vercel. The machine-learning backend was deployed on Render and works as a web service, but video inference may fail on the Render free instance because TensorFlow, MobileNetV2, MediaPipe, and video processing can exceed the available memory or request-processing limits.
>
> The complete application works locally, including video upload, shot classification, confidence scores, class probabilities, and pose analytics.

---

## 🎯 Project Objective

The objective of this project is to automatically identify cricket batting shots from short video clips using deep learning.

Unlike image classification, cricket shot recognition requires both:

- **Spatial information** — what the player and batting posture look like in individual frames.
- **Temporal information** — how the player's movement changes across consecutive frames.

The final system therefore combines **MobileNetV2 for spatial feature extraction** with a **GRU for temporal sequence classification**.

MediaPipe is additionally used to provide pose-based analytics.

---

## 🧠 Final Architecture

```text
Cricket Video
      │
      ▼
Frame Sampling
      │
      ▼
16 Video Frames
      │
      ▼
MobileNetV2
      │
      ▼
Spatial Feature Extraction
      │
      ▼
Feature Sequence
      │
      ▼
GRU
      │
      ▼
Softmax Classification
      │
      ▼
Predicted Cricket Shot
```

A parallel pose-analysis pipeline processes the video using MediaPipe:

```text
Video
  │
  ▼
MediaPipe Pose
  │
  ▼
Pose Landmarks
  │
  ▼
Pose Detection Analytics
```

---

## 🤖 Deep Learning Model

### MobileNetV2

MobileNetV2 is used as a pretrained convolutional neural network for extracting **spatial features** from individual video frames.

Transfer learning allows the system to use visual representations learned from ImageNet rather than training a CNN completely from scratch.

### GRU

A **Gated Recurrent Unit (GRU)** processes the sequence of MobileNetV2 feature vectors.

The GRU learns temporal relationships between frames, allowing the model to understand the motion involved in performing a cricket shot.

### MediaPipe

MediaPipe Pose is used separately to detect human body landmarks across video frames.

The application reports pose information such as:

- Total processed frames
- Frames where pose was detected
- Pose detection percentage
- Average detected landmarks

---

## 📊 Final Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **69.60%** |
| Number of Classes | **10** |
| Frames per Video | **16** |
| Spatial Model | **MobileNetV2** |
| Temporal Model | **GRU** |
| Pose Engine | **MediaPipe** |

The final **MobileNetV2 + GRU** architecture was selected for the deployed application.

---

## 🏏 Supported Cricket Shots

The model classifies videos into the following 10 batting-shot categories:

1. Cover
2. Defense
3. Flick
4. Hook
5. Late Cut
6. Lofted
7. Pull
8. Square Cut
9. Straight
10. Sweep

---

## 📁 Dataset

The project uses a cricket-shot video dataset containing **10 batting-shot categories**.

Videos are organized by class and divided into training, validation, and testing data.

During preprocessing, each video is converted into a fixed sequence of frames so that all samples have a consistent input structure.

The preprocessing pipeline includes:

```text
Video
 ↓
Frame Extraction
 ↓
Uniform Frame Sampling
 ↓
Resize Frames
 ↓
Normalization
 ↓
16-Frame Video Sequence
```

The dataset itself is not stored in this GitHub repository because video datasets are comparatively large.

---

## 🔄 Feature Extraction

Instead of repeatedly processing every video through MobileNetV2 during GRU training, MobileNetV2 features can be precomputed.

The feature extraction script is located at:

```text
features/extract_mobilenet_features.py
```

The generated NumPy feature arrays are excluded from Git because they are generated artifacts and can be recreated when required.

---

## 🧪 Model Experimentation

Multiple deep-learning architectures were explored during development.

The repository contains trained model files for experiments including:

```text
cricket_shot_3dcnn_epoch5.keras
cricket_shot_3dcnn_epoch10.keras
cricket_shot_3dcnn_epoch15.keras
cricket_shot_cnn_gru.keras
precomputed_mobilenet_gru_best.keras
```

The experiments helped compare approaches for spatial and temporal video classification.

The **MobileNetV2 + GRU model** was ultimately selected for the final application.

---

## 🔍 Prediction Output

For an uploaded video, the backend can return:

```json
{
  "prediction": "defense",
  "confidence_percent": 99.46,
  "confidence_level": "high",
  "probabilities": {
    "cover": "...",
    "defense": "...",
    "flick": "...",
    "hook": "...",
    "late_cut": "...",
    "lofted": "...",
    "pull": "...",
    "square_cut": "...",
    "straight": "...",
    "sweep": "..."
  },
  "pose": {
    "total_frames": 75,
    "frames_with_pose": 64,
    "pose_detection_percent": 85.33
  }
}
```

The confidence value represents the model's confidence for the **individual prediction**. It should not be confused with the overall **69.60% test-set accuracy**.

---

## 💻 Web Application

The project contains a full-stack interface for uploading and analyzing cricket videos.

### Frontend

Built using:

- React
- Vite
- JavaScript
- CSS

The interface displays:

- Video upload
- Model architecture
- Analysis pipeline
- Predicted cricket shot
- Prediction confidence
- Class probabilities
- Pose analytics
- Supported cricket-shot classes

### Backend

Built using:

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy

The Flask backend exposes an `/analyze` endpoint that accepts a cricket video and performs the complete inference pipeline.

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Spatial Feature Extraction | MobileNetV2 |
| Temporal Classification | GRU |
| Pose Estimation | MediaPipe |
| Video Processing | OpenCV |
| Numerical Processing | NumPy |
| Backend | Flask |
| Frontend | React + Vite |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |
| Version Control | Git + GitHub |

---

## 📂 Project Structure

```text
Cricket-Shot-Classification/
│
├── backend/
│   └── app.py
│
├── dataset/
│
├── features/
│   ├── extract_mobilenet_features.py
│   └── mobilenetv2/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── inference/
│   ├── analyzer.py
│   └── predictor.py
│
├── models/
│   ├── cricket_shot_3dcnn_epoch5.keras
│   ├── cricket_shot_3dcnn_epoch10.keras
│   ├── cricket_shot_3dcnn_epoch15.keras
│   ├── cricket_shot_cnn_gru.keras
│   └── precomputed_mobilenet_gru_best.keras
│
├── notebooks/
│   └── 02_model_training.ipynb
│
├── pose_estimation/
│
├── preprocessing/
│
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt
└── requirements-deploy.txt
```

---

# 🚀 Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/karamchandsuresh/Cricket-Shot-Classification.git
```

Move into the project:

```bash
cd Cricket-Shot-Classification
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start the Backend

From the project root:

```bash
python backend/app.py
```

The Flask backend runs locally at:

```text
http://127.0.0.1:5000
```

The analysis API is:

```text
POST /analyze
```

---

## 5. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local Vite URL displayed in the terminal.

---

## ☁️ Deployment

### Frontend — Vercel

The React/Vite frontend is deployed on Vercel.

**Live Application:**

https://cricket-shot-classification.vercel.app

### Backend — Render

The Flask backend has been configured for Render using Gunicorn.

The deployed service can successfully initialize the trained GRU classifier and MobileNetV2 feature extractor.

However, the Render free instance has limited resources. Full video analysis requires TensorFlow inference, MobileNetV2 feature extraction, OpenCV video processing, and MediaPipe pose estimation. This can cause the worker to exceed the available memory or execution limits during inference.

Therefore, the **local version should be used for the complete functional demonstration**.

---

## ⚠️ Limitations

- Overall test accuracy is currently **69.60%**.
- Predictions depend on video quality, camera angle, player visibility, and shot execution.
- The system currently supports only 10 predefined cricket-shot classes.
- Pose detection can be affected by occlusion and poor video quality.
- CPU-based video inference can take time.
- The free cloud backend has insufficient resources for reliable TensorFlow + MediaPipe video inference.
- The system performs shot classification and pose analytics but does not yet provide advanced biomechanical coaching recommendations.

---

## 🔮 Future Scope

The project can be extended with:

- Larger and more diverse cricket datasets
- Data augmentation
- Improved transfer learning
- Transformer-based video models
- Real-time cricket shot recognition
- Bat and ball tracking
- Player-specific performance analysis
- Batting technique evaluation
- Joint-angle analysis
- Automated coaching recommendations
- Cloud deployment using GPU-enabled infrastructure
- Mobile application support

---

## 🎓 Learning Outcomes

This project demonstrates practical implementation of:

- Deep learning
- Convolutional neural networks
- Recurrent neural networks
- GRU networks
- Transfer learning
- Video classification
- Spatial feature extraction
- Temporal sequence modeling
- Pose estimation
- Model evaluation
- REST API development
- React frontend development
- Full-stack ML integration
- Cloud deployment

---

## 👨‍💻 Author

**Karamchand Suresh**

MCA — Generative AI  
Alliance University, Bengaluru

---

## 📌 Repository

GitHub:

https://github.com/karamchandsuresh/Cricket-Shot-Classification

---

## 🌐 Live Application

https://cricket-shot-classification.vercel.app

---

## 🏁 Conclusion

This project demonstrates an end-to-end deep learning pipeline for cricket batting-shot analysis.

By combining **MobileNetV2 for spatial feature extraction**, **GRU for temporal sequence learning**, and **MediaPipe for pose analytics**, the system analyzes both the visual appearance and movement patterns contained in cricket batting videos.

The project also demonstrates the complete machine-learning development lifecycle—from video preprocessing and model experimentation to inference, API integration, frontend development, and cloud deployment.
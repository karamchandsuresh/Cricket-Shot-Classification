# Cricket Shot Classification & Analytics

A deep learning-based cricket shot classification and player pose analytics system using **MobileNetV2, GRU, MediaPipe, Flask, and React**.

The application accepts a short cricket batting video, predicts the type of cricket shot, calculates prediction confidence, performs player pose analysis, and displays the results through an interactive web dashboard.

---

## Project Overview

Cricket shot recognition is a video classification problem where both spatial and temporal information are important. Many cricket shots have similar player movements, bat positions, and camera viewpoints, making accurate classification challenging.

This project explores multiple deep learning approaches for cricket shot classification. After comparing different architectures, the final system uses **precomputed MobileNetV2 features with a GRU classifier**.

The complete application combines:

- **MobileNetV2** for pretrained spatial feature extraction
- **GRU** for temporal sequence learning
- **MediaPipe Pose** for player pose analytics
- **Flask** for the backend API
- **React + Vite** for the frontend dashboard

The final selected model achieved **69.60% test accuracy** on the 10-class test dataset.

---

## Supported Cricket Shots

The system classifies videos into 10 cricket shot categories:

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

## System Architecture

    Cricket Batting Video
            |
            v
    Video Preprocessing
            |
            v
    16 Sampled RGB Frames
       112 x 112 x 3
            |
            v
    Pretrained MobileNetV2
            |
            v
    Frame-Level Features
       16 x 1280
            |
            v
      GRU Classifier
            |
            v
    10-Class Prediction
            |
       +----+----+
       |         |
       v         v
    Confidence  MediaPipe Pose
                    |
                    v
               Pose Analytics
            |
            v
       Flask Backend
            |
            v
       React Dashboard

---

## Dataset

The project uses a cricket shot video dataset divided into training, validation, and testing sets.

| Split | Number of Videos |
|---|---:|
| Training | 1250 |
| Validation | 250 |
| Testing | 250 |
| **Total** | **1750** |

The dataset contains **10 cricket shot classes**.

The dataset itself is not included in the GitHub repository because of its size.

---

## Video Preprocessing

Every video is converted into a consistent representation before being passed to the model.

The preprocessing pipeline:

1. Opens the cricket video using OpenCV.
2. Determines the total number of video frames.
3. Samples **16 frames** across the video.
4. Resizes every frame to **112 × 112 pixels**.
5. Converts the frames to RGB.
6. Normalizes the image data.
7. Passes the processed frames to the feature extraction pipeline.

The resulting video representation has the shape:

    (16, 112, 112, 3)

Where:

- `16` = number of sampled frames
- `112 × 112` = spatial resolution
- `3` = RGB channels

---

## Deep Learning Model Experiments

Several deep learning architectures were evaluated during development.

### 1. 3D CNN

A lightweight 3D Convolutional Neural Network was trained directly on the cricket video frames.

**Test Accuracy: 44.40%**

The model learned useful spatial-temporal patterns but struggled to distinguish several visually similar cricket shots.

---

### 2. CNN + GRU

A CNN-based spatial feature extractor was combined with a GRU for temporal sequence learning.

**Test Accuracy: 37.20%**

This approach performed below the 3D CNN baseline.

---

### 3. MobileNetV2 + GRU

A pretrained MobileNetV2 model was used to extract spatial information from individual video frames, while a GRU learned the temporal relationships between the frames.

Running MobileNetV2 repeatedly during every training epoch was computationally expensive on CPU.

This led to the final optimized approach: **precomputing MobileNetV2 features before GRU training**.

---

### 4. R3D-18 Transfer Learning

A pretrained R3D-18 video recognition network was also evaluated.

The pretrained backbone was used for spatio-temporal feature extraction and the final classification layer was adapted for the 10 cricket shot classes.

**Test Accuracy: 41.60%**

R3D-18 provided useful pretrained video features but did not outperform the final MobileNetV2 + GRU approach.

---

### 5. Precomputed MobileNetV2 Features + GRU

This was selected as the **final model architecture**.

Instead of running MobileNetV2 during every training epoch, MobileNetV2 features are extracted once and saved.

Each video becomes a sequence of:

    16 frames × 1280 features

The GRU then learns the temporal relationship between these frame-level feature vectors.

### Final Test Accuracy

**69.60%**

This was the best result obtained among the evaluated approaches.

---

## Model Comparison

| Model | Test Accuracy |
|---|---:|
| CNN + GRU | 37.20% |
| R3D-18 Transfer Learning | 41.60% |
| 3D CNN | 44.40% |
| **Precomputed MobileNetV2 + GRU** | **69.60%** |

The final model improved the test accuracy by approximately **25 percentage points** compared with the 3D CNN baseline.

---

## Final Model Performance

The final MobileNetV2 + GRU model was evaluated using **250 unseen test videos**.

### Overall Test Accuracy

**69.60%**

### Per-Class Recall

| Cricket Shot | Recall |
|---|---:|
| Cover | 60% |
| Defense | 96% |
| Flick | 40% |
| Hook | 88% |
| Late Cut | 76% |
| Lofted | 92% |
| Pull | 52% |
| Square Cut | 40% |
| Straight | 80% |
| Sweep | 72% |

The results show that the model performs particularly well for **Defense, Lofted, Hook, and Straight**.

Classes such as **Flick and Square Cut** remain more difficult because of similarities between batting actions.

---

## Pose Analytics

The application also integrates **MediaPipe Pose**.

Pose estimation is used as an additional analytics component and does not directly determine the cricket shot prediction.

For an uploaded video, the application reports:

- Total frames analyzed
- Frames where a player pose was detected
- Pose detection percentage
- Average detected landmarks

MediaPipe can detect up to **33 body landmarks** for a detected person.

This provides additional information about how successfully the player's body movement can be tracked throughout the batting video.

---

## Web Application

The trained deep learning pipeline is integrated into a complete web application.

### Frontend

The frontend is developed using:

- React
- Vite
- JavaScript
- CSS

The dashboard allows the user to:

- Upload a cricket batting video
- Preview browser-supported video formats
- Analyze the uploaded cricket shot
- View the predicted shot
- View prediction confidence
- View confidence level
- View pose detection statistics
- View all 10 class probabilities
- View information about the model architecture

---

## Backend

The backend is developed using **Flask**.

The main analysis endpoint is:

    POST /analyze

When a video is submitted, the backend:

1. Receives the uploaded video.
2. Temporarily stores the video.
3. Runs video preprocessing.
4. Extracts MobileNetV2 features.
5. Passes the features through the GRU classifier.
6. Performs MediaPipe pose analysis.
7. Combines the results.
8. Returns the analysis to the React frontend as JSON.
9. Removes the temporary uploaded video.

---

## Example Analysis

One successful Defense test produced:

    Predicted Shot: Defense
    Confidence: 99.97%
    Pose Detection: 81.18%
    Frames Analyzed: 85
    Pose Frames: 69

A Hook test produced:

    Predicted Shot: Hook
    Confidence: 62.76%
    Pose Detection: 38.32%
    Frames Analyzed: 107
    Pose Frames: 41

The application also displays the probability assigned to every supported cricket shot.

---

## Project Structure

    Cricket-Shot-Classification/
    │
    ├── backend/
    │   └── app.py
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
    │   │   ├── index.css
    │   │   └── main.jsx
    │   ├── package.json
    │   └── vite.config.js
    │
    ├── inference/
    │   ├── analyzer.py
    │   └── predictor.py
    │
    ├── notebooks/
    │   └── 02_model_training.ipynb
    │
    ├── pose_estimation/
    │
    ├── preprocessing/
    │
    ├── models/
    │
    ├── dataset/
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

Large/generated project files such as the dataset, virtual environment, extracted NumPy features, frontend `node_modules`, cache files, and other generated artifacts are excluded using `.gitignore`.

---

## Installation

### 1. Clone the Repository

    git clone https://github.com/karamchandsuresh/Cricket-Shot-Classification.git

Then:

    cd Cricket-Shot-Classification

### 2. Create the Python Virtual Environment

    python -m venv venv

Activate it on Windows:

    .\venv\Scripts\Activate.ps1

### 3. Install Python Dependencies

    pip install -r requirements.txt

### 4. Install Frontend Dependencies

    cd frontend
    npm install

---

## Running the Application

The backend and frontend should be run in separate terminals.

### Backend

From the project root, activate the virtual environment and run:

    python -m backend.app

The backend runs locally on:

    http://127.0.0.1:5000

### Frontend

Open another terminal:

    cd frontend
    npm run dev

The frontend is normally available at:

    http://localhost:5173

Open the frontend in the browser, select a cricket batting video, and click the analysis button.

---

## Technologies Used

### Deep Learning

- TensorFlow
- Keras
- PyTorch
- Torchvision
- MobileNetV2
- GRU
- 3D CNN
- R3D-18

### Computer Vision

- OpenCV
- MediaPipe

### Backend

- Python
- Flask
- Flask-CORS

### Frontend

- React
- Vite
- JavaScript
- CSS

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## Key Challenges

Major challenges encountered during the project included:

- Limited cricket video training data
- Similar movements between different cricket shots
- Variations in camera viewpoints
- CPU-based deep learning training
- Combining spatial and temporal video information
- Balancing model complexity with training time
- Improving generalization to unseen videos

Precomputing pretrained MobileNetV2 features provided a major improvement in both training efficiency and classification performance.

---

## Limitations

The current system has several limitations:

- Overall test accuracy is approximately **69.60%**.
- Some visually similar shots can still be confused.
- Flick and Square Cut are particularly difficult classes.
- Prediction quality depends on video quality and camera viewpoint.
- The video should ideally contain one clear batting action.
- Pose detection can decrease when the batsman is small or partially obscured.
- The training dataset is relatively limited.
- The model may not generalize equally well to cricket footage very different from the training dataset.

---

## Future Scope

Future improvements could include:

- Training with a larger and more diverse cricket dataset
- Additional video data augmentation
- Higher-resolution input frames
- Increasing the number of temporal frames
- Fine-tuning pretrained video models
- Video Transformer architectures
- Incorporating pose landmarks directly into classification
- Bat and ball tracking
- Real-time shot classification
- Shot quality analysis
- Player technique comparison
- Mobile application support
- Cloud deployment

---

## Conclusion

This project demonstrates an end-to-end deep learning approach for cricket shot classification and analytics.

Multiple architectures were experimentally evaluated, including **3D CNN, CNN + GRU, MobileNetV2 + GRU, and R3D-18**.

The final **precomputed MobileNetV2 features + GRU** architecture achieved the best test performance with **69.60% accuracy**.

The final application combines deep learning classification, confidence analysis, MediaPipe pose estimation, a Flask backend API, and a React dashboard into a complete cricket video analytics system.

---

## Author

**Karamchand Suresh**

MCA — Generative AI  
Alliance University, Bengaluru
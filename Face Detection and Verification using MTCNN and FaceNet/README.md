***

# 📌 Face Detection and Verification using MTCNN and FaceNet

## 📖 Overview

This project implements a complete pipeline for **face detection and face verification** using state-of-the-art deep learning models:

* **MTCNN** for accurate face detection and alignment
* **FaceNet** for generating embeddings and performing face verification

The system detects faces from images, extracts meaningful feature embeddings, and compares them for identity verification.

***

## 🎯 Features

* ✅ Face detection with bounding boxes and landmarks
* ✅ Face alignment and preprocessing
* ✅ Feature extraction using FaceNet embeddings
* ✅ Face verification using cosine similarity / distance metrics
* ✅ Support for multiple faces in a single image
* ✅ Easy-to-extend pipeline for recognition tasks

***

## 🧠 Model Architecture

### 🔹 MTCNN (Multi-task Cascaded Convolutional Network)

Used for:

* Face detection
* Facial landmark localization
* Face alignment

### 🔹 FaceNet

Used for:

* Extracting 128-D (or 512-D) embeddings
* Converting faces into numerical representations for comparison

***

## 📂 Project Structure

```bash
.
├── data/                 # Input images / datasets
├── models/               # Pretrained models
├── src/
│   ├── detect.py        # Face detection (MTCNN)
│   ├── embed.py         # Face embedding (FaceNet)
│   ├── verify.py        # Face verification logic
│   └── utils.py         # Helper functions
├── notebooks/           # Experiments / demos
├── requirements.txt
└── README.md
```

***

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd face-verification

pip install -r requirements.txt
```

***

## 🚀 Usage

### 1️⃣ Face Detection

```python
from detect import detect_faces

faces = detect_faces("image.jpg")
```

***

### 2️⃣ Feature Extraction

```python
from embed import get_embedding

embedding = get_embedding(face_image)
```

***

### 3️⃣ Face Verification

```python
from verify import verify_faces

result, distance = verify_faces(img1, img2)

print("Same person:", result)
```

***

## 📊 Verification Method

Faces are compared using:

```text
Cosine similarity OR Euclidean distance
```

Typical threshold:

```text
distance < 0.8 → same person
distance >= 0.8 → different person
```

(*Tune depending on dataset*)

***

## 🧪 Example Output

```text
Image 1 vs Image 2
Distance: 0.54
Result: SAME PERSON ✅
```

***

## 📦 Dependencies

* Python 3.8+
* PyTorch / TensorFlow (depending on implementation)
* OpenCV
* NumPy
* MTCNN library
* FaceNet model

***

## 🔧 Improvements / Future Work

* 🔹 Real-time video face verification
* 🔹 Face recognition database (multi-identity classification)
* 🔹 Model optimization for edge devices
* 🔹 ONNX / TensorRT deployment
* 🔹 Implement for Andriod application

***

## ⚠️ Limitations

* Sensitive to extreme lighting and occlusion
* Performance depends on alignment quality
* Threshold tuning required for different datasets

***

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

## 👤 Author

Jalil Nourmohammadi Khiarak

***

🙏 Acknowledgment / Inspiration
This project is inspired by the work of:

Mr. Data Scientist

The following resource provided the initial idea and implementation guidance:

Citation:
[1] Face Detection & Verification with MTCNN & FaceNet | ML Project #3 | End to End Project, https://youtu.be/gLE3vRJ3FJA?si=9M5BQk14KDdZzVUe
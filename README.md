# 🧠 Deepfake Image Detection using AI

A complete AI-powered web application that detects whether an uploaded image is **REAL or DEEPFAKE** using deep learning techniques (CNN - MobileNetV2) along with statistical analysis.

---

## 🚀 Features

* Upload an image (drag & drop or file input)
* Detect whether image is **REAL or FAKE**
* Confidence score display
* Visual analysis indicators (texture, noise, symmetry, etc.)
* Clean frontend UI
* Flask backend API
* Lightweight and runs on CPU

---

## 🏗️ Project Structure

```
deepfake-detect/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
│       └── deepfake_model.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── README.md
```

---

## ⚙️ Setup & Run (VS Code)

Follow these steps carefully:

---

### 🔹 Step 1: Open Project in VS Code

* Open VS Code
* Click **File → Open Folder**
* Select the project folder: `deepfake-detect`

---

### 🔹 Step 2: Open Terminal

Press:

```
Ctrl + `
```

---

### 🔹 Step 3: Navigate to Backend

```
cd backend
```

---

### 🔹 Step 4: Create Virtual Environment

```
python -m venv venv
```

---

### 🔹 Step 5: Activate Virtual Environment

#### Windows:

```
venv\Scripts\activate
```

#### Mac/Linux:

```
source venv/bin/activate
```

You should see:

```
(venv)
```

---

### 🔹 Step 6: Install Dependencies

```
pip install flask flask-cors numpy Pillow scipy tensorflow==2.17.0
```

OR (if using requirements file):

```
pip install -r requirements.txt
```

---

### 🔹 Step 7: Run Backend Server

```
python app.py
```

Expected output:

```
Server starting on http://localhost:5000
```

---

### 🔹 Step 8: Run Frontend

* Go to:

  ```
  frontend/index.html
  ```
* Open it in browser

OR (recommended):

* Install **Live Server extension**
* Right-click `index.html`
* Click **Open with Live Server**

---

### 🔹 Step 9: Use the App

1. Upload an image
2. Click **Analyze Image**
3. View result:

   * REAL / FAKE
   * Confidence %

---

## 🧠 How It Works

This project uses a hybrid approach:

### 🔹 1. Deep Learning (CNN)

* MobileNetV2 extracts image features
* Detects patterns like:

  * texture inconsistencies
  * unnatural edges
  * facial artifacts

### 🔹 2. Statistical Analysis

* Color distribution
* Noise patterns
* Symmetry detection
* Texture sharpness (Laplacian variance)

### 🔹 3. Decision Logic

* Combines multiple indicators
* Computes a **fake probability**
* Classifies image based on threshold

---

## ⚠️ Notes

* Runs on CPU (no GPU required)
* Uses pretrained model (no heavy training needed)
* Designed for demonstration & educational purposes

---

## 🎥 Demo Purpose

This project was built as part of an academic submission to demonstrate:

* AI-based deepfake detection
* Full-stack integration (Frontend + Backend + ML)
* Real-time prediction workflow

---

## ✨ Author

Developed as part of a technical project on:

**"Artificial Intelligence Techniques for Detecting Deepfake Images and Videos"**

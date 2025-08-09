
# 🛡️ GuardiAI – Facial Security System

**AI-Powered Face Recognition for Secure Access**

GuardiAI is a deep learning–based facial authentication system designed to provide a smarter, faster, and more intuitive way to secure access. Instead of relying on traditional passwords or PINs, GuardiAI uses advanced computer vision and machine learning to identify authorized users — all in real time.

---

## 🚀 Features

* **Facial Recognition Authentication** – Unlock systems or trigger actions only for recognized faces.
* **Anchor & Positive Image Dataset Support** – Customizable training with user-specific images.
* **Negative Dataset Integration** – Robust model performance with diverse background data.
* **Real-Time Webcam Feed** – Continuous live video capture and authentication.
* **Optimized Image Preprocessing** – Cropping, resizing, and cleaning for consistent training input.

---

## 🛠️ Tech Stack

* **Python 3.11+**
* **OpenCV** – Real-time video processing.
* **Pytorch** – Deep learning model creation and training.
* **NumPy & Pandas** – Data manipulation and preprocessing.
* **UUID** – Unique image naming for dataset storage.

---

## 📂 Project Structure

```
GuardiAI/
│
├── data/
│   ├── anchor/        # Anchor images for training
│   ├── positive/      # Positive (authorized) images
│   ├── negative/      # Negative dataset images
│
├── model/             # Saved model files
├── scripts/           # Python scripts for training & testing
├── README.md
└── requirements.txt
```

---

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/yashraj01-se/GuardiAI.git
cd GuardiAI

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Usage

### 1️⃣ **Collect Dataset**

Run the dataset collection script to capture anchor & positive images from your webcam:

```bash
python collect_data.py
```

### 2️⃣ **Train the Model**

Train GuardiAI on your collected images:

```bash
python train.py
```

### 3️⃣ **Run Real-Time Authentication**

```bash
python authenticate.py
```

---

## 📊 Dataset

* **Anchor & Positive** – Captured manually via webcam.
* **Negative** – Uses the [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset).

---

## ⚠️ Disclaimer

This project is for **educational purposes** and should not be used as the sole security measure in high-risk environments.

---

## 📎 Links

🔗 **GitHub Repo**: [GuardiAI](https://github.com/yashraj01-se/GuardiAI)
📚 **Dataset Source**: [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

---


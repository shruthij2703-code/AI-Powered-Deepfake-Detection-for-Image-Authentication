# 🔍 AI-Powered Deepfake Detection For Image Authentication

> A deep learning-based web application that detects whether an image is **real or AI-generated (deepfake)** using MobileNetV2 and Streamlit.

---

## 📸 Demo

Upload any face image and the **DeepScan AI** engine will analyze it and return a verdict — **REAL** or **FAKE** — along with a confidence score and visual indicators.

---

## 🚀 Features

- ✅ Detects deepfake / AI-generated images with high confidence
- ✅ Built on **MobileNetV2** (transfer learning from ImageNet)
- ✅ Beautiful **Streamlit UI** with animated results and confidence meter
- ✅ Real-time inference on uploaded images
- ✅ Scanline animation, burst effects, and verdict cards
- ✅ Easy to retrain on your own dataset

---

## 🗂️ Project Structure

```
AI_Deepfake_Detection/
│
├── app.py               # Streamlit web app (main UI)
├── train.py             # Model training script
├── test.py              # Command-line image testing script
├── model.h5             # Trained MobileNetV2 model (binary classifier)
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python 3.10 runtime for cloud deployment
├── test.jpg             # Sample test image
└── dataset/             # Training data (not included — see below)
    ├── real/            # Real face images
    └── fake/            # AI-generated / deepfake images
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_Deepfake_Detection.git
cd AI_Deepfake_Detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 🧪 Testing via Command Line

To test a single image without the UI:

```bash
python test.py
```

Edit the last line of `test.py` to point to your image:

```python
predict_image("your_image.jpg")
```

---

## 🏋️ Training Your Own Model

### 1. Prepare your dataset

Organize your images into this folder structure:

```
dataset/
├── real/     ← real face images
└── fake/     ← deepfake / AI-generated images
```

### 2. Run training

```bash
python train.py
```

The best model is automatically saved as `model.h5`.

**Training configuration (in `train.py`):**

| Parameter   | Value |
|-------------|-------|
| Image Size  | 224×224 |
| Batch Size  | 16 |
| Epochs      | 10 (with early stopping) |
| Base Model  | MobileNetV2 (ImageNet weights) |
| Optimizer   | Adam |
| Loss        | Binary Crossentropy |

---

## 🧠 Model Architecture

```
MobileNetV2 (frozen, pretrained on ImageNet)
       ↓
GlobalAveragePooling2D
       ↓
Dense(128, relu)
       ↓
Dropout(0.5)
       ↓
Dense(1, sigmoid)   →   0 = REAL,  1 = FAKE
```

---

## 📦 Dependencies

```
streamlit
tensorflow==2.15.0
opencv-python-headless
numpy==1.24.3
pillow
scikit-learn
matplotlib
h5py
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The `dataset/` folder is **not included** in this repository due to size constraints.

You can use publicly available deepfake datasets such as:

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)

---

## 📁 Dataset Not Included?

The trained `model.h5` is included. If it's too large for GitHub, download it from:

> 🔗 *(Add your Google Drive / HuggingFace model link here)*

---

## 🛠️ Built With

- [TensorFlow / Keras](https://www.tensorflow.org/) — Deep learning framework
- [MobileNetV2](https://keras.io/api/applications/mobilenet/) — Pretrained CNN backbone
- [Streamlit](https://streamlit.io/) — Web app framework
- [OpenCV](https://opencv.org/) — Image processing
- [Pillow](https://pillow.readthedocs.io/) — Image loading

---

## 👤 Author

**JUJJUVARAPU SHRUTHI**  
📧 shruthij.2703@gmail.com  
🔗 [GitHub](https://github.com/shruthij2703-code)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

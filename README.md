# Deepfake Detector

An **end-to-end deepfake detection pipeline** that extracts faces from real and fake videos, creates a dataset, trains a ResNet18-based binary classification model, and provides a Flask web app for inference.

---

## **Project Overview**

Deepfakes are manipulated videos where faces are swapped or modified using AI techniques. Detecting deepfakes is important for content verification, security, and research. This project provides:

1. **Face Extraction & Preprocessing**  
   - Uses **MediaPipe** to extract faces from videos.
   - Saves faces into a structured dataset for training.
   - Supports multiple video formats (`.mp4`, `.avi`, `.mov`).

2. **Model Training**  
   - Fine-tunes **ResNet18** for binary classification: `REAL` vs `FAKE`.
   - Uses transfer learning from ImageNet weights.
   - Supports data augmentation for better generalization.

3. **Flask Web Application**  
   - Upload a video and receive a deepfake prediction.
   - Uses the trained model to analyze faces frame by frame.
   - Returns **average confidence** and final verdict (`REAL` or `FAKE`).

---

## **Folder Structure**

```

deepfake-detector/
│
├── app.py                 # Flask web app
├── output.py              # Model inference script
├── train.py               # Training script
├── preprocessing.py       # Face extraction script
│
├── model/
│   └── resnet\_model2.pth # Trained model weights
│
├── templates/
│   └── index.html         # HTML template for Flask app
│
├── uploads/               # Temporary folder for uploaded videos
├── processed              # Folder for extracted faces (fake/real)
├── requirements.txt       # Python dependencies
├── README.md
└── .gitignore

````

---

## **Setup Instructions**

### **1. Clone the repository**
```bash
git clone https://github.com/sai-1973/Deepfake-Detection.git
cd Deepfake-Detection
````

### **2. Create a virtual environment (recommended)**

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

*or using venv:*

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **Face Extraction & Preprocessing**

1. Place your videos in `data/fake` and `data/real`.
2. Run the extraction script:

```bash
python preprocessing.py
```

* Extracts faces using MediaPipe.
* Saves them into `processed/fake` and `processed/real`.
* You can adjust **number of frames per video** in the script (`FRAMES_PER_VIDEO`).

---

## **Training the Model**

1. Run the training script:

```bash
python train.py
```

2. Key features:

   * Uses **ResNet18** with pretrained ImageNet weights.
   * Binary classification (`REAL` vs `FAKE`).
   * Training/validation split (80/20).
   * Model saved as `resnet_model2.pth` in `model/`.

> You can increase `NUM_EPOCHS` or add **data augmentation** to improve accuracy.

---

## **Running the Flask Web App**

1. Make sure your trained model is in `model/resnet_model2.pth`.
2. Start the server:

```bash
python app.py
```

3. Open your browser:

```
http://127.0.0.1:5000/
```

4. Upload a video to get predictions:

* Returns **average confidence** per face.
* Shows **final verdict**: `REAL` or `FAKE`.

---

## **Dependencies**

* Python 3.10+
* [PyTorch](https://pytorch.org/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* OpenCV (`opencv-python`)
* Pillow (`PIL`)
* MediaPipe
* Flask
* tqdm

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## **Notes & Tips**

* If videos are **large**, MediaPipe extracts faces **fast**, but the Flask inference may take time depending on CPU/GPU.
* Ensure **model weights** match your trained model.
* For better results, train on **more videos** and use **augmentation**.
* Only upload **small demo videos** to the Flask app — avoid very large files.

---


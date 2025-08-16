import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image

MODEL_PATH = 'model/resnet_model2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not read video file."

    predictions = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = Image.fromarray(face)
                face = transform(face).unsqueeze(0).to(DEVICE)

                output = model(face)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)

    cap.release()

    if predictions:
        avg = sum(predictions) / len(predictions)
        percent = avg * 100
        verdict = "REAL" if avg > 0.5 else "FAKE"
        return f"Avg Confidence: {percent:.2f}%<br>Final Prediction: {verdict}"
    else:
        return "No faces detected."

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

app = FastAPI(
    title="Covi-Shift API",
    description="AI-Powered COVID-19 Chest X-Ray Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

MODEL_PATH = "model_checkpoints/resnet18.pth"

model = ResNetClassifier(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

CLASS_NAMES = {0: "COVID", 1: "Normal", 2: "Pneumonia"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ResNet-18", "classes": list(CLASS_NAMES.values())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return {"success": False, "error": "Only JPG/PNG images are accepted"}

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = float(probabilities[predicted_class])

    all_probs = {CLASS_NAMES[i]: f"{float(probabilities[i]) * 100:.2f}%" for i in range(3)}

    return {
        "success": True,
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": f"{confidence * 100:.2f}%",
        "all_probabilities": all_probs
    }

# MUST be the very last line
app.mount("/", StaticFiles(directory="static", html=True), name="static")
import torch
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import io

from model import CatDogCNN
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cat vs Dog Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CatDogCNN().to(device)
model.load_state_dict(torch.load("Model/model.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Cat vs Dog Inference API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(image).item()

    if prob >= 0.5:
        prediction = "Dog"
        confidence = prob
    else:
        prediction = "Cat"
        confidence = 1 - prob

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }

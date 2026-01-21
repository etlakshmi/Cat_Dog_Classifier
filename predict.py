import torch
from torchvision import transforms
from PIL import Image
from model import CatDogCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CatDogCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(image).item()

    if prob >= 0.5:
        return "Dog", prob
    else:
        return "Cat", 1 - prob

label, confidence = predict("images/8.jpg")
print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}")

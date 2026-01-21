# Cat vs Dog Classifier API

A FastAPI-based REST API for classifying images as either cats or dogs using a PyTorch CNN model.

## Features

- Fast image classification using PyTorch
- RESTful API with FastAPI
- CORS enabled for cross-origin requests
- GPU support with automatic fallback to CPU
- Confidence scores for predictions

## Prerequisites

- Python 3.8+
- PyTorch
- FastAPI
- PIL (Pillow)
- torchvision

## Installation

1. Clone the repository and navigate to the project directory

2. Install required dependencies:
```bash
pip install torch torchvision fastapi uvicorn pillow python-multipart
```

3. Ensure you have the following files:
   - `predict_api.py` - Main API file
   - `model.py` - Contains the `CatDogCNN` model definition
   - `model.pth` - Pre-trained model weights

## Project Structure

```
.
├── predict_api.py      # FastAPI application
├── model.py            # CNN model architecture
├── model.pth           # Trained model weights
└── README.md           # This file
```

## Usage

### Starting the Server

Run the API server using uvicorn:

```bash
uvicorn predict_api:app --reload
```

For production:
```bash
uvicorn predict_api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```
GET /
```

**Response:**
```json
{
  "message": "Cat vs Dog Inference API is running"
}
```

#### Predict Image
```
POST /predict
```

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (JPG, PNG, etc.)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("cat_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Example using JavaScript (fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response:**
```json
{
  "prediction": "Dog",
  "confidence": 0.9234
}
```

- `prediction`: Either "Cat" or "Dog"
- `confidence`: Confidence score between 0 and 1

## Model Details

### Input Specifications
- Image size: Resized to 224x224 pixels
- Color space: RGB
- Normalization: ImageNet mean and std
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Prediction Logic
- Output probability ≥ 0.5 → Dog
- Output probability < 0.5 → Cat
- Confidence = max(prob, 1 - prob)

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Hardware Acceleration

The API automatically detects and uses CUDA-enabled GPU if available, otherwise falls back to CPU.

To check which device is being used, check the server logs on startup.

## CORS Configuration

Currently configured to allow all origins (`*`). For production, modify the CORS settings in `predict_api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify allowed domains
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Model file not found
Ensure `model.pth` is in the same directory as `predict_api.py`

### CUDA out of memory
The API will automatically fall back to CPU if GPU memory is insufficient

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Unsupported image format
The API converts all images to RGB, but ensure your image file is a valid format (JPG, PNG, etc.)

## Performance Tips

- Use GPU for faster inference
- Consider batch prediction for multiple images
- Implement caching for frequently predicted images
- Use production ASGI server like gunicorn with uvicorn workers

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
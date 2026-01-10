import io
import os
import datetime
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --- CONFIGURATION ---
MODEL_PATH = 'model/deeplabv3_resnet101_coco.pth'

# --- 1. SQL DATABASE SETUP ---
# This automatically creates 'history.db' to store records
SQLALCHEMY_DATABASE_URL = "sqlite:///./history.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UploadRecord(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# --- 2. AI MODEL SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

try:
    print(f"🔄 Loading Model on {device}...")
    # Initialize Architecture
    model = deeplabv3_resnet101(weights=None, aux_classifier=True)
    
    # Check for local file, else fallback to Internet Download (Safety Net)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Custom Local Model Loaded!")
    else:
        print("⚠️ Local .pth not found. Downloading official weights for demo...")
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights, aux_classifier=True)
    
    model.to(device).eval()
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# --- 3. API APP SETUP ---
app = FastAPI()

# Enable CORS so your HTML file can talk to this Python file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. IMAGE PROCESSING FUNCTIONS ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(520), # Resize for speed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device), image

@app.post("/remove-bg/")
async def remove_background(file: UploadFile = File(...)):
    # 1. Log entry to SQL
    db = SessionLocal()
    record = UploadRecord(filename=file.filename, status="Processing")
    db.add(record)
    db.commit()

    try:
        # 2. Run AI
        contents = await file.read()
        input_tensor, original_image = transform_image(contents)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # 3. Create Transparent Image
        mask = (output_predictions > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask).resize(original_image.size, Image.NEAREST)
        
        result = original_image.copy()
        result.putalpha(mask_img)

        # 4. Return Result
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Update SQL status
        record.status = "Success"
        db.commit()

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        record.status = f"Failed: {str(e)}"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
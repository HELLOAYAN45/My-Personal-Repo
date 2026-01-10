from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware # <--- NEW IMPORT
from fastapi.responses import Response
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# --- 🟢 NEW: SECURITY PERMISSION SLIP (CORS) ---
# This tells your backend: "It is okay to accept requests from the Vercel website."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" means allow ANY website (Vercel, Mobile, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------------------------

# Load the AI Model
print("Loading Model...")
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

def make_transparent(image):
    # 1. Prepare image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    # 2. Run AI
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # 3. Create Mask
    mask = output.argmax(0).byte().cpu().numpy()
    
    # 4. Apply Mask (Cutout)
    original = image.convert("RGBA")
    data = original.getdata()
    new_data = []
    width, height = original.size
    
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 15: # 15 is the class ID for 'Person'
                new_data.append(data[y * width + x])
            else:
                new_data.append((0, 0, 0, 0)) # Transparent
                
    original.putdata(new_data)
    return original

@app.post("/remove-bg/")
async def remove_background(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result_image = make_transparent(image)
    
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
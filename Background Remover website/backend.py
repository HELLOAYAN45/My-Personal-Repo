from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🟢 BACK TO HIGH QUALITY MODEL (ResNet101)
print("Loading High-Quality Model (ResNet101)...")
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

def make_transparent(image):
    # 🟢 SPEED TRICK: Resize huge images to max 1024px
    # This makes ResNet run 3x faster without noticeable quality loss
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    mask = output.argmax(0).byte().cpu().numpy()
    
    original = image.convert("RGBA")
    data = original.getdata()
    new_data = []
    width, height = original.size
    
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 15: # 15 is 'Person'
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
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import io

app = FastAPI()

# Enable CORS for any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated damage detection
def detect_damage(image: Image.Image) -> List[str]:
    return [
        "Detected dent on front bumper",
        "Scuff marks on right fender"
    ]

# Simulated estimate generation
def generate_estimate(detections: List[str], vehicle_info: str) -> dict:
    return {
        "vehicle_info": vehicle_info,
        "damage_summary": detections,
        "estimated_cost": "$1,200 - $1,500"
    }

@app.post("/analyze")
async def analyze_photos(
    files: List[UploadFile] = File(...),
    vehicle_info: str = Form(...)
):
    detections = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        detections.extend(detect_damage(image))
    estimate = generate_estimate(detections, vehicle_info)
    return JSONResponse(content=estimate)
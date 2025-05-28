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
    allow_origins = [
    "http://nspxn.com",
    "https://nspxn.com",  # ✅ Add this!
    "http://localhost:3000",
    "https://*.nspxn.com"
]
,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import random

# Updated mock damage detection
def detect_damage(image: Image.Image) -> List[str]:
    damage_types = [
        "Dent on front bumper",
        "Scuff on right fender",
        "Crack in headlamp",
        "Scratch on hood",
        "Dent on driver-side door",
        "Broken mirror",
        "Rear bumper gouge",
        "Quarter panel scrape"
    ]
    # Randomly return 1-2 damages per image
    return random.sample(damage_types, k=random.randint(1, 2))

# Updated mock estimate calculation
def generate_estimate(detections: List[str], vehicle_info: str) -> dict:
    unique_damages = list(set(detections))
    base_price = 300
    cost = base_price + len(unique_damages) * 200
    return {
        "vehicle_info": vehicle_info,
        "damage_summary": unique_damages,
        "estimated_cost": f"${cost:,} - ${cost + 300:,}"
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

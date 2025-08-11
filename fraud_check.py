from PIL import Image
import re
from datetime import datetime
import io
from ultralytics import YOLO
import os

def calculate_fraud_risk(combined_text, image_files=None):
    score = 0
    flags = []
    current_year = datetime.now().year  # 2025

    # 1. Check for suspicious terms
    suspicious_terms = ["fraud", "fake", "altered", "manipulated", "forged"]
    if any(term in combined_text.lower() for term in suspicious_terms):
        flags.append("Suspicious terms detected")
        score += 15

    # 2. Claim number consistency
    claim_pattern = r"claim\s*(?:#?:\s*)?([0-9]{6}-[0-9]{6}-[A-Z]{2}-[0-9]{2})"
    claim_numbers = re.findall(claim_pattern, combined_text, re.IGNORECASE)
    if claim_numbers:
        valid_claims = [c for c in claim_numbers if re.match(r"^[0-9]{6}-[0-9]{6}-[A-Z]{2}-[0-9]{2}$", c)]
        if valid_claims:
            if len(valid_claims) > 1 and len(set(valid_claims)) > 1:
                flags.append(f"Multiple inconsistent claim numbers detected: {', '.join(valid_claims)}")
                score += 20
        else:
            flags.append("No valid claim number format detected")
            score += 10
    else:
        flags.append("No claim number found")
        score += 10

    # 3. Estimate vs. Photo Damage Comparison
    if image_files:
        # Extract repair items from estimate
        repair_items_pattern = r"(?:repl|rpr|r&I|blnd)\s+([a-z\s&]+)(?:\s+\w+)*"
        repair_items = re.findall(repair_items_pattern, combined_text, re.IGNORECASE)
        repair_items = [item.strip() for item in repair_items if item.strip()]  # e.g., ["bumper cover", "headlamp assy"]
        logger.debug(f"Extracted repair items: {repair_items}")

        # Load YOLO model for damage detection (assuming damage-detector.pt is available)
        try:
            model_path = os.path.join(os.getcwd(), "damage-detector.pt")
            if not os.path.exists(model_path):
                logger.warning(f"Damage detection model not found at {model_path}, skipping damage comparison")
            else:
                model = YOLO(model_path)
                detected_damages = []
                for img in image_files:
                    img.file.seek(0)
                    image = Image.open(io.BytesIO(img.file.read())).convert("RGB")
                    results = model(image)
                    for result in results:
                        for box in result.boxes:
                            class_name = model.names[int(box.cls[0])] if box.cls else "unknown"
                            detected_damages.append(class_name.lower())
                detected_damages = list(set(detected_damages))
                logger.debug(f"Detected damages: {detected_damages}")

                # Map repair items to damage types (basic mapping)
                damage_map = {
                    "bumper": ["bumper", "front bumper", "rear bumper"],
                    "headlamp": ["headlamp", "lamp"],
                    "hood": ["hood"],
                    # Add more mappings as needed
                }
                estimated_damages = set()
                for item in repair_items:
                    for key, values in damage_map.items():
                        if any(v in item for v in values):
                            estimated_damages.add(key)

                # Compare
                missing_damages = estimated_damages - set(detected_damages)
                extra_damages = set(detected_damages) - estimated_damages
                if missing_damages or extra_damages:
                    discrepancy = []
                    if missing_damages:
                        discrepancy.append(f"Estimated damages not in photos: {', '.join(missing_damages)}")
                    if extra_damages:
                        discrepancy.append(f"Damages in photos not in estimate: {', '.join(extra_damages)}")
                    flags.append("Discrepancy between estimate and photo damage: " + "; ".join(discrepancy))
                    score += 20  # Penalty for significant mismatch

        except Exception as e:
            logger.error(f"Damage comparison error: {str(e)}")
            flags.append("Error in damage comparison")
            score += 10

    # 4. Edited/Manipulated Image Indicators
    if image_files:
        for img in image_files:
            try:
                img.file.seek(0)
                image = Image.open(io.BytesIO(img.file.read()))
                # Check EXIF data for manipulation indicators
                exif_data = image._getexif()
                if exif_data:
                    exif_date = exif_data.get(36867) or exif_data.get(306)  # DateTimeOriginal or DateTime
                    if exif_date:
                        try:
                            exif_datetime = datetime.strptime(exif_date, "%Y:%m:%d %H:%M:%S")
                            if exif_datetime.year != current_year:
                                flags.append("EXIF date outside 2025")
                                score += 20
                        except ValueError:
                            flags.append("Invalid EXIF date format")
                            score += 15
                    else:
                        flags.append("Missing EXIF date")
                        score += 10
                else:
                    flags.append("No EXIF data (possible manipulation)")
                    score += 15
                # Basic manipulation check (e.g., high compression or metadata tampering)
                if image.format == "JPEG" and image.info.get("quality", 95) < 80:
                    flags.append("High compression detected (possible editing)")
                    score += 15
            except Exception as e:
                flags.append(f"Image processing error: {str(e)}")
                score += 10

    # Cap score at 100%
    score = min(100, score)

    # Ensure explanation is always provided
    explanation = "No fraud indicators detected." if not flags else "\n".join(flags)

    return {"score": score, "flags": flags, "explanation": explanation}d." if not flags else "\n".join(flags)

    return {"score": score, "flags": flags, "explanation": explanation}

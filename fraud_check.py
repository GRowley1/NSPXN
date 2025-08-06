from PIL import Image
import re
from datetime import datetime

def calculate_fraud_risk(combined_text, image_files=None):
    score = 0
    flags = []
    claim_number = None

    # Extract claim number from text
    claim_match = re.search(r"Claim\s*#?\s*[:=]?\s*(R226\d+.*|[A-Z0-9-]+)", combined_text, re.IGNORECASE)
    if claim_match:
        claim_number = claim_match.group(1).strip()

    # Check claim number consistency across text and images
    if claim_number:
        text_claims = re.findall(r"Claim\s*#?\s*[:=]?\s*(R226\d+.*|[A-Z0-9-]+)", combined_text, re.IGNORECASE)
        if len(set(text_claims)) > 1:
            flags.append("Inconsistent claim numbers in text")
            score += 25

    # Process images for EXIF date and manipulation indicators
    if image_files:
        for img_file in image_files:
            try:
                img_file.file.seek(0)
                img = Image.open(io.BytesIO(img_file.file.read()))
                # Check EXIF data for date
                exif_date = img._getexif().get(36867) if img._getexif() else None  # 36867 is DateTimeOriginal
                if exif_date:
                    exif_datetime = datetime.strptime(exif_date, "%Y:%m:%d %H:%M:%S")
                    if exif_datetime.year != 2025:
                        flags.append(f"EXIF date {exif_date} not in 2025")
                        score += 25
                else:
                    flags.append("Missing EXIF date")
                    score += 15

                # Basic manipulation check (e.g., high compression or metadata tampering)
                # Note: This is a simplified check; advanced detection requires libraries like exiftool or forensic analysis
                if img.format == "JPEG" and img.info.get("quality", 100) < 80:
                    flags.append("Possible image compression manipulation")
                    score += 20
            except Exception as e:
                flags.append(f"Image processing error: {str(e)}")
                score += 10

    # Check for suspicious terms
    suspicious_terms = ["fraud", "fake", "altered", "manipulated", "forged", "suspicious"]
    if any(term in combined_text.lower() for term in suspicious_terms):
        flags.append("Suspicious terms detected")
        score += 30

    # Cap score at 100%
    score = min(100, score)

    return {"score": score, "flags": flags}

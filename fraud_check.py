from PIL import Image
import re
from datetime import datetime
import io

def calculate_fraud_risk(combined_text, image_files=None):
    score = 0
    flags = []
    current_year = datetime.now().year  # 2025

    # 1. Check for suspicious terms
    suspicious_terms = ["fraud", "fake", "altered", "manipulated", "forged"]
    if any(term in combined_text.lower() for term in suspicious_terms):
        flags.append("Suspicious terms detected")
        score += 15  # Reduced from 25 to calibrate severity

    # 2. Claim number consistency
    claim_pattern = r"Claim\s*#?\s*([0-9]{6}-[0-9]{6}-[A-Z]{2}-[0-9]{2})"
    claim_numbers = re.findall(claim_pattern, combined_text, re.IGNORECASE)
    if claim_numbers:
        valid_claims = [c for c in claim_numbers if re.match(r"^[0-9]{6}-[0-9]{6}-[A-Z]{2}-[0-9]{2}$", c)]
        if len(valid_claims) > 1 and len(set(valid_claims)) > 1:
            flags.append(f"Multiple inconsistent claim numbers detected: {', '.join(valid_claims)}")
            score += 20  # Penalty for multiple different claims
        elif not valid_claims:
            flags.append("No valid claim number format detected")
            score += 10
    else:
        flags.append("No claim number found")
        score += 10

    # 3. Edited/Manipulated Image Indicators
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

    return {"score": score, "flags": flags, "explanation": explanation}

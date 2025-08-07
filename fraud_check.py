from PIL import Image
import re
from datetime import datetime
import io

def calculate_fraud_risk(combined_text, image_files=None, reference_claim=None):
    score = 0
    flags = []
    current_year = datetime.now().year  # 2025
    reference_claim = reference_claim or "010683-162546-AD-01"  # Default to provided claim number

    # 1. Check for suspicious terms
    suspicious_terms = ["fraud", "fake", "altered", "manipulated", "forged"]
    if any(term in combined_text.lower() for term in suspicious_terms):
        flags.append("Suspicious terms detected")
        score += 25

    # 2. Claim number consistency
    claim_numbers = re.findall(r"Claim\s*#?\s*([A-Z0-9-]+)", combined_text, re.IGNORECASE)
    if claim_numbers:
        if reference_claim not in claim_numbers:
            flags.append(f"Inconsistent claim number; expected {reference_claim}, found {', '.join(claim_numbers)}")
            score += 25
    else:
        flags.append(f"No claim number found; expected {reference_claim}")
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
    explanation = "No fraud indicators detected." if not flags else "\n".join(flags)

    return {"score": score, "flags": flags, "explanation": explanation}

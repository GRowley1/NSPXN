
import imagehash
from PIL import Image, ExifTags
import piexif
import io
import logging

logger = logging.getLogger(__name__)

def calculate_fraud_risk(image_files, texts) -> dict:
    issues = []
    hashes = set()
    exif_errors = 0
    duplicate_count = 0

    for file in image_files:
        try:
            file.file.seek(0)
            img_bytes = file.file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Check perceptual hash for duplication
            h = imagehash.phash(img)
            if h in hashes:
                duplicate_count += 1
                issues.append(f"Duplicate photo detected: {file.filename}")
            else:
                hashes.add(h)

            # Check EXIF metadata integrity
            try:
                exif = img._getexif()
                if not exif:
                    exif_errors += 1
                    issues.append(f"Missing EXIF data: {file.filename}")
                else:
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        if decoded == "DateTime" and "202" not in str(value):
                            issues.append(f"Suspicious timestamp in {file.filename}: {value}")
            except Exception:
                exif_errors += 1
                issues.append(f"EXIF read error: {file.filename}")

        except Exception as e:
            logger.warning(f"Fraud check failed for {file.filename}: {str(e)}")
            issues.append(f"Unreadable image: {file.filename}")

    score = 0
    if duplicate_count:
        score += 40
    if exif_errors > 1:
        score += 30
    if any("Suspicious timestamp" in i for i in issues):
        score += 30

    risk_level = (
        "❌ High" if score >= 70 else
        "⚠️ Moderate" if score >= 40 else
        "✅ Low"
    )

    return {
        "risk_level": risk_level,
        "score": score,
        "issues": issues
    }

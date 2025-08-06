def calculate_fraud_risk(text):
    # Example implementation
    flags = []
    score = 0
    if "suspicious" in text.lower():
        flags.append("Suspicious keyword detected")
        score = 75
    return {"score": score, "flags": flags}

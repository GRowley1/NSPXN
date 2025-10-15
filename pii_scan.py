#!/usr/bin/env python
import sys, pathlib, json
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

# Preserve VIN & Claim # (don’t count as PII)
VIN_PATTERN = r"\b([A-HJ-NPR-Z0-9]{17})\b"
CLAIM_PATTERN = r"\b(?:(?:Claim|CLM|Clm)\s*#?\s*[:\-]?\s*)?([A-Z0-9]{5,}[A-Z0-9\-]{0,})\b"

analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(
    PatternRecognizer("VIN", [Pattern("vin-17", VIN_PATTERN, 0.8)])
)
analyzer.registry.add_recognizer(
    PatternRecognizer("CLAIM_NUMBER", [Pattern("claim-generic", CLAIM_PATTERN, 0.6)])
)

REDACT_ENTITY_TYPES = {
    "PERSON","PHONE_NUMBER","EMAIL_ADDRESS","US_SSN","CREDIT_CARD",
    "IBAN_CODE","LOCATION","NRP","ORGANIZATION","DATE_TIME","IP_ADDRESS","CRYPTO","MEDICAL_LICENSE","URL"
}

def analyze_text(t: str):
    results = analyzer.analyze(text=t, language="en")
    return [r for r in results if r.entity_type in REDACT_ENTITY_TYPES]

violations = []
for path in sys.argv[1:]:
    text = pathlib.Path(path).read_text(errors="ignore")
    hits = analyze_text(text)
    if hits:
        violations.append({
            "file": path,
            "count": len(hits),
            "entities": sorted({h.entity_type for h in hits})
        })

if violations:
    print("[Presidio] Potential PII found:\n" + json.dumps(violations, indent=2))
    sys.exit(1)

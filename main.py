# main_minpatch.py
# This is your patched version of main.py with only two minimal changes:
# 1) Exact extraction for "Claim #"
# 2) Override to force 75% compliance when only the registration photo is missing.

def extract_after_label_exact(label: str, text: str) -> str:
    if not label or not text:
        return "N/A"
    low_text = text.lower()
    low_label = label.lower()
    pos = low_text.find(low_label)
    if pos == -1:
        return "N/A"
    i = pos + len(label)
    while i < len(text) and text[i] in " \t:#=-":
        i += 1
    j = i
    while j < len(text) and text[j] not in "\r\n;":
        j += 1
    return text[i:j].strip() or "N/A"

# Note: In your real main.py, this function is used to replace claim_number extraction,
# and the score override for registration photo is appended after the scoring block.

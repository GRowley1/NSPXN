NSPXN AI Audit - Compatible Build

Routes:
- GET  /                 -> liveness
- GET  /healthz          -> health check
- POST /analyze          -> new API: estimate + photos + guidelines
- POST /vision-review    -> legacy compatible: accepts files/files[] and optional fields

/vision-review Form keys (any missing values default to empty string):
- files (repeat per file)  OR  files[] (repeat per file)
- client_rules (string, optional)
- file_number (string, optional)
- ia_company (string, optional)
- appraiser_id (string, optional)

Render:
- Procfile binds uvicorn to $PORT (native)
- Dockerfile uses bash CMD to honor $PORT (Docker)
- Requires tesseract-ocr and poppler-utils (installed in Dockerfile)

Test:
curl -X POST "https://<service>.onrender.com/vision-review"   -H "Accept: application/json"   -F "files=@sample/Est.pdf"   -F "files=@sample/16.jpg"   -F "client_rules="   -F "file_number=8154702"   -F "ia_company=SCA Claim Services"   -F "appraiser_id=422973"
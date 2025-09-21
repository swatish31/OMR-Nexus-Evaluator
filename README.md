# **OMR-Nexus-Evaluator**

\# OMR-Nexus-Evaluator

\#\# Overview  
\*\*OMR-Nexus-Evaluator\*\* is an automated Optical Mark Recognition (OMR) evaluation system for educational and research use.    
It processes OMR sheets captured with mobile or flatbed scanners, detects marked answers using computer vision, and generates detailed score reports.

\---

\#\# System Architecture  
\#\#\# Core Components  
\- \*\*FastAPI Backend\*\* – RESTful API for processing    
\- \*\*OpenCV Engine\*\* – Image preprocessing and bubble detection    
\- \*\*Reference-Based Evaluation\*\* – Template-based position matching    
\- \*\*Result Management\*\* – Score calculation and export

\---

\#\# Installation  
\#\#\# Prerequisites  
\- Python ≥3.8    
\- OpenCV ≥4.8.1    
\- FastAPI ≥0.104.1    
\- Uvicorn ≥0.24.0  

\#\#\# Setup

git clone https://github.com/yourusername/OMR-Nexus-Evaluator.git  
cd OMR-Nexus-Evaluator

python \-m venv venv  
source venv/bin/activate      \# Linux/Mac  
venv\\Scripts\\activate         \# Windows

pip install \-r requirements.txt  
uvicorn main:app \--reload  
API Endpoints  
Method	Endpoint	Description  
POST	/upload-reference/	Upload blank OMR sheet to create template  
POST	/process-omr/	Process single student OMR sheet  
POST	/batch-process/	Process multiple sheets  
GET	/reference-status/	Check template status  
GET	/health/	Health check  
GET	/answer-key/	Retrieve answer key  
POST	/update-answer-key/	Update answer key

OMR Sheet Format  
100 questions (20 per subject)

Subjects: Python, Data Analysis, MySQL, Power BI, Advanced Statistics

Options: A/B/C/D

Layout: 20 rows × 5 questions per row

Answer Key Example

json  
Copy code  
{  
  "PYTHON": \["B", "C", "D", "A", ...\],  
  "DATA\_ANALYSIS": \["C", "D", "A", "B", ...\],  
  "MySQL": \["D", "A", "B", "C", ...\],  
  "POWER\_BI": \["A", "B", "C", "D", ...\],  
  "Adv\_STATS": \["B", "C", "D", "A", ...\]  
}  
Configuration  
Create a .env file:

HOST=0.0.0.0  
PORT=8000  
DEBUG=True  
UPLOAD\_DIR=./uploads  
RESULTS\_DIR=./results  
ANSWER\_KEY\_DIR=./answer\_keys  
Directory Layout

OMR-Nexus-Evaluator/  
├── main.py  
├── main\_core\_ff.py  
├── answer\_keys/  
│   └── version\_1.json  
├── uploads/  
├── results/  
└── requirements.txt  
Usage Example  
python  
Copy code  
import requests

\# Upload reference sheet  
with open("reference\_omr.jpg", "rb") as f:  
    requests.post("http://localhost:8000/upload-reference/", files={"file": f})

\# Process student sheet  
with open("student\_omr.jpg", "rb") as f:  
    r \= requests.post("http://localhost:8000/process-omr/", files={"file": f})  
    print(r.json())  
Sample Response

json  
Copy code  
{  
  "success": true,  
  "result": {  
    "image\_path": "student\_omr.jpg",  
    "answers": \["A", "B", "C", "X", ...\],  
    "scores": {  
      "PYTHON": 15,  
      "DATA\_ANALYSIS": 18,  
      "MySQL": 16,  
      "POWER\_BI": 17,  
      "Adv\_STATS": 14,  
      "total": 80  
    },  
    "threshold": 27.6,  
    "status": "processed"  
  },  
  "processing\_time": 2.45  
}  
Image Processing Pipeline  
Preprocessing: resize, grayscale, enhance contrast, reduce noise, adaptive threshold

Bubble Detection: contour analysis, area and radius filters, grid alignment

Answer Extraction: pixel intensity analysis, ambiguity detection

Scoring: subject-wise comparison, total score, statistics

Error Handling  
Issue	Recommendation  
No bubbles detected	Adjust preprocessing parameters  
Misalignment	Re-upload a clean reference sheet  
Low contrast	Improve lighting  
Perspective distortion	Capture image directly from above

Response codes: 200 OK · 400 Bad request · 500 Server error

Performance  
Reference sheet: 3–5 s

Student sheet: 2–3 s

Batch (10 sheets): 15–25 s

Accuracy: 99% (good quality) → 85% (low quality)

Workflow  
mathematica  
Copy code  
Capture → Upload → Preprocess → Detect Bubbles  
    ↓  
Template → Match Positions → Extract Answers  
    ↓  
Compare Key → Score → Generate Results → Export  
Contributing  
Fork the repository

Create a feature branch

Commit with clear messages

Open a pull request

Follow PEP 8, include type hints, docstrings, and unit tests.

License  
MIT License – see LICENSE.

Support  
Review existing issues

Open new issues with logs, images, and environment details

Deployment  
Uvicorn (production):

uvicorn main:app \--host 0.0.0.0 \--port 8000 \--workers 4  
Gunicorn \+ Uvicorn:

gunicorn main:app \-k uvicorn.workers.UvicornWorker \-w 4  
Docker:

dockerfile

FROM python:3.8-slim  
WORKDIR /app  
COPY requirements.txt .  
RUN pip install \-r requirements.txt  
COPY . .  
CMD \["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"\]  
javascript


# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from datetime import datetime
from typing import List

from core_omr import OMRProcessor

app = FastAPI(title="OMR-Nexus API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = OMRProcessor()

@app.post("/process-omr/")
async def process_omr(file: UploadFile = File(...)):
    """Process a single OMR sheet image"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process the image
        start_time = datetime.now()
        result = processor.process_image(tmp_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "result": result,
            "processing_time": processing_time
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
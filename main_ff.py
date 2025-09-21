# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from datetime import datetime
from typing import List, Optional
import json

from main_core_ff import OMRProcessor

app = FastAPI(title="OMR-Nexus API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = OMRProcessor()

@app.post("/upload-reference/")
async def upload_reference_sheet(file: UploadFile = File(...)):
    """Upload and process a reference OMR sheet"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process the reference sheet
        start_time = datetime.now()
        result = processor.process_reference_sheet(tmp_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "message": "Reference sheet processed successfully",
            "bubbles_detected": len(result['bubble_positions']),
            "processing_time": processing_time
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/process-omr/")
async def process_omr(file: UploadFile = File(...), use_reference: bool = True):
    """Process a single OMR sheet image"""
    try:
        # Check if reference sheet is available if requested
        if use_reference and processor.reference_data is None:
            raise HTTPException(status_code=400, detail="No reference sheet available. Please upload a reference sheet first.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process the image
        start_time = datetime.now()
        result = processor.process_image(tmp_path, use_reference=use_reference)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "result": result,
            "processing_time": processing_time,
            "used_reference": use_reference
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/batch-process/")
async def batch_process_omr(files: List[UploadFile] = File(...)):
    """Process multiple OMR sheets in batch"""
    try:
        if processor.reference_data is None:
            raise HTTPException(status_code=400, detail="No reference sheet available. Please upload a reference sheet first.")
        
        results = []
        total_time = 0
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Process the image
            start_time = datetime.now()
            result = processor.process_image(tmp_path, use_reference=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            total_time += processing_time
            
            result["filename"] = file.filename
            result["processing_time"] = processing_time
            results.append(result)
            
            # Clean up
            os.unlink(tmp_path)
        
        return {
            "success": True,
            "total_sheets_processed": len(results),
            "total_processing_time": total_time,
            "average_time_per_sheet": total_time / len(results) if results else 0,
            "results": results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/reference-status/")
async def reference_status():
    """Check if reference sheet is loaded"""
    return {
        "has_reference": processor.reference_data is not None,
        "bubbles_detected": len(processor.reference_data['bubble_positions']) if processor.reference_data else 0
    }

@app.get("/answer-key/")
async def get_answer_key(version: str = "version_1"):
    """Get the current answer key"""
    try:
        return {
            "success": True,
            "version": version,
            "answer_key": processor.answer_keys.get(version, {})
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/update-answer-key/")
async def update_answer_key(version: str = "version_1", answer_key: dict = None):
    """Update the answer key"""
    try:
        if answer_key:
            processor.answer_keys[version] = answer_key
            # Save to file
            os.makedirs("answer_keys", exist_ok=True)
            with open(f"answer_keys/{version}.json", "w") as f:
                json.dump(answer_key, f, indent=2)
            
            return {
                "success": True,
                "message": f"Answer key {version} updated successfully",
                "answer_key": answer_key
            }
        else:
            raise HTTPException(status_code=400, detail="No answer key provided")
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "reference_loaded": processor.reference_data is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
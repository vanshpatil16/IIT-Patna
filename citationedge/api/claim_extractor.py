from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import *
import tempfile
import os
import numpy as np  # Add this import
from citationedge.services.claim_extractor import main_rag_pipeline
from citationedge.utils.shortlist import *

print("📦 [claim_extractor.py] Module loaded")

router = APIRouter()
print("✅ Router initialized")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

@router.post("/extract-claims")
async def analyze_paper_endpoint(
    json_file: UploadFile = File(...)):
    """Analyze PDF paper and extract keywords"""
    pdf_file = json_file
    print("📥 Received request to /analyze-paper")
    print(f"📄 Uploaded file name: {pdf_file.filename}")

    if not pdf_file.filename.endswith('.json'):
        print("❌ File is not a json. Aborting.")
        raise HTTPException(status_code=400, detail="File must be a json")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        print("📂 Created temporary file")
        content = await pdf_file.read()
        temp_file.write(content)
        temp_path = temp_file.name
        print(f"📝 File saved to temp path: {temp_path}")

    try:
        print("🚀 Starting PDF analysis using run_paper_analysis()")
        claims = main_rag_pipeline(temp_path)
        print("✅ Analysis completed")
        print(f"📌 Extracted {len(claims)} claims")
        claims = shortlist_top_claims(claims, top_k=5, debug=True)
        # Convert numpy types before returning
        response_data = {
            "claims": convert_numpy_types(claims)  # Apply conversion here
        }
        
        return response_data

    except Exception as e:
        print(f"🔥 Exception occurred during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🧹 Temporary file deleted: {temp_path}")
        else:
            print("⚠️ Temporary file not found for deletion.")

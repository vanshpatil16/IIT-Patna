from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional, List, Dict, Any
import tempfile
import os
import numpy as np
from citationedge.services.analyze_argumentation import argumentation_analysis
import json
import re

print("📦 [analyze_argumentation.py] Module loaded")

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

@router.post("/analyze-argumentation")
async def analyze_argumentation(
    json_file: UploadFile = File(...),
    claims_file: Optional[UploadFile] = File(default=None)):
    """Analyze JSON paper and extract argumentation analysis"""
    print("📥 Received request to /analyze-argumentation")
    print(f"📄 Uploaded file name: {json_file.filename}")
    
    if claims_file:
        print(f"📄 Claims file name: {claims_file.filename}")

    if not json_file.filename.endswith('.json'):
        print("❌ File is not a json. Aborting.")
        raise HTTPException(status_code=400, detail="File must be a json")
    
    if claims_file and not claims_file.filename.endswith('.json'):
        print("❌ Claims file is not a json. Aborting.")
        raise HTTPException(status_code=400, detail="Claims file must be a json")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        print("📂 Created temporary file")
        content = await json_file.read()
        temp_file.write(content)
        temp_path = temp_file.name
        print(f"📝 File saved to temp path: {temp_path}")

    claims_temp_path = None
    if claims_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_claims_file:
            print("📂 Created temporary claims file")
            claims_content = await claims_file.read()
            temp_claims_file.write(claims_content)
            claims_temp_path = temp_claims_file.name
            print(f"📝 Claims file saved to temp path: {claims_temp_path}")

    try:
        paper_json = None
        claims = None
        
        with open(temp_path, 'r', encoding="utf-8") as f:
            paper_json = json.load(f)

        # Handle claims - either from uploaded file or default file
        if claims_temp_path:
            # Use uploaded claims file
            with open(claims_temp_path, 'r', encoding="utf-8") as f:
                claims = json.load(f)
        else:
            # Fallback to reading from default file
            with open("D:\\citationedge\\claims.json", 'r', encoding="utf-8") as f:
                claims = json.load(f)
            
        print("🚀 Starting argumentation analysis")
        categorized_gaps = argumentation_analysis(paper_json, claims["claims"])
        print("✅ Analysis completed")
        print(f"📌 Extracted {len(categorized_gaps)} categorized gaps")

        # Convert numpy types before returning
        response_data = {
            "categorized_gaps": convert_numpy_types(categorized_gaps)
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
            
        if claims_temp_path and os.path.exists(claims_temp_path):
            os.unlink(claims_temp_path)
            print(f"🧹 Claims temporary file deleted: {claims_temp_path}")
        elif claims_temp_path:
            print("⚠️ Claims temporary file not found for deletion.")

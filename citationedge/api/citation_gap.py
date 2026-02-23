from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from typing import *
import tempfile
import os
import numpy as np  # Add this import
from citationedge.services.citation_gap import analyze_citation_gaps
import json
print("📦 [citation_gap.py] Module loaded")

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

@router.post("/analyze-citation-gaps")
async def analyze_citation_gap(
    json_file: UploadFile = File(...),
    keywords: Optional[str] = Form(default=None)):
    """Analyze PDF paper and extract keywords"""
    pdf_file = json_file
    print("📥 Received request to /citation-gaps")
    print(f"📄 Uploaded file name: {pdf_file.filename}")
    print(f"📥 Raw keywords received: {keywords}")

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

        paper_json=0
        with open(temp_path, 'r', encoding="utf-8") as f:
            paper_json = json.load(f)

        processed_keywords = []
        if keywords:
            try:
                # Try to parse as JSON array first
                parsed_keywords = json.loads(keywords)
                if isinstance(parsed_keywords, list):
                    processed_keywords = [str(k).strip() for k in parsed_keywords if k and str(k).strip()]
                else:
                    processed_keywords = [str(keywords).strip()]
            except json.JSONDecodeError:
                # If not JSON, try comma-separated values
                processed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        
        keywords = processed_keywords
        
        print(f"📌 Using keywords: {keywords}")
        print("🚀 Starting PDF analysis using run_paper_analysis()")
        categorized_gaps = analyze_citation_gaps(paper_json, keywords)
        print("✅ Analysis completed")
        print(f"📌 Extracted {len(categorized_gaps)} claims")

        # Convert numpy types before returning
        response_data = {
            "categorized_gaps": convert_numpy_types(categorized_gaps)  # Apply conversion here
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

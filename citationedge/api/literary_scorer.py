from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional, List, Dict, Any
import tempfile
import os
import numpy as np
from citationedge.services.literary_scorer import analyze_literary_score
import json
import re
from citationedge.utils.date_helpers import *

print("📦 [literary_score.py] Module loaded")

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

@router.post("/analyze-literary-score")
async def analyze_literaryscore(
    json_file: UploadFile = File(...),
    citation_gaps: Optional[UploadFile] = File(default=None),
    claims_analysis: Optional[UploadFile] = File(default=None)):
    """Analyze JSON paper and extract literary score analysis"""

    if not json_file.filename.endswith('.json'):
        print("❌ File is not a json. Aborting.")
        raise HTTPException(status_code=400, detail="File must be a json")
    
    if citation_gaps and not citation_gaps.filename.endswith('.json'):
        print("❌ Citation gaps file is not a json. Aborting.")
        raise HTTPException(status_code=400, detail="Citation gaps file must be a json")

    if claims_analysis and not claims_analysis.filename.endswith('.json'):
        print("❌ Claims analysis file is not a json. Aborting.")
        raise HTTPException(status_code=400, detail="Claims analysis file must be a json")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        content = await json_file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    citation_temp_path = None
    if citation_gaps:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_gaps_file:
            print("📂 Created temporary citation gaps file")
            gaps_content = await citation_gaps.read()
            temp_gaps_file.write(gaps_content)  # ✅ Fixed: was claims_content
            citation_temp_path = temp_gaps_file.name
            print(f"📝 Citation gaps file saved to temp path: {citation_temp_path}")

    claim_analysis_temp_path = None
    if claims_analysis:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_claims_file:
            print("📂 Created temporary claims analysis file")
            claims_content = await claims_analysis.read()
            temp_claims_file.write(claims_content)
            claim_analysis_temp_path = temp_claims_file.name
            print(f"📝 Claims analysis file saved to temp path: {claim_analysis_temp_path}")

    try:
        paper_json = None
        gaps = None
        claims = None
        
        # Load main paper JSON
        with open(temp_path, 'r', encoding="utf-8") as f:
            paper_json = json.load(f)
            
        # Load citation gaps (from uploaded file or default)
        if citation_temp_path:
            with open(citation_temp_path, 'r', encoding="utf-8") as f:
                gaps = json.load(f)
        else:
            with open("D:\\citationedge\\gaps.json", 'r', encoding="utf-8") as f:
                gaps = json.load(f)

        # Load claims analysis (from uploaded file or default)
        if claim_analysis_temp_path:
            with open(claim_analysis_temp_path, 'r', encoding="utf-8") as f:
                claims = json.load(f)
        else:
            with open("D:\\citationedge\\argumentation.json", 'r', encoding="utf-8") as f:
                claims = json.load(f)
            
        print("🚀 Starting literary score analysis")
        literary_score = analyze_literary_score(paper_json, gaps["categorized_gaps"], claims["categorized_gaps"])
        print("✅ Analysis completed")
        print(f"📌 Generated literary score analysis")

        # Convert numpy types before returning
        response_data = {
            "literary_score_report": convert_numpy_types(literary_score)
        }
        
        return response_data

    except Exception as e:
        print(f"🔥 Exception occurred during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🧹 Temporary file deleted: {temp_path}")
        else:
            print("⚠️ Temporary file not found for deletion.")
            
        if citation_temp_path and os.path.exists(citation_temp_path):
            os.unlink(citation_temp_path)
            print(f"🧹 Citation gaps temporary file deleted: {citation_temp_path}")
        elif citation_temp_path:
            print("⚠️ Citation gaps temporary file not found for deletion.")
        
        if claim_analysis_temp_path and os.path.exists(claim_analysis_temp_path):
            os.unlink(claim_analysis_temp_path)
            print(f"🧹 Claims analysis temporary file deleted: {claim_analysis_temp_path}")
        elif claim_analysis_temp_path:
            print("⚠️ Claims analysis temporary file not found for deletion.")

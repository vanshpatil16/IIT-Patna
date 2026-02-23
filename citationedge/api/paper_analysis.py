from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import tempfile
import os
from citationedge.services.pdf_processor import *

print("📦 [paper_analysis.py] Module loaded")

router = APIRouter()
print("✅ Router initialized")

@router.post("/analyze-paper")
async def analyze_paper_endpoint(
    pdf_file: UploadFile = File(...),
    num_keywords: int = Form(10)):
    """Analyze PDF paper and extract keywords"""

    if not pdf_file.filename.endswith('.pdf'):
        print("❌ File is not a PDF. Aborting.")
        raise HTTPException(status_code=400, detail="File must be a PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        content = await pdf_file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        keywords, semantic_keywords, paper_data = run_paper_analysis(
            pdf_path=temp_path,
            num_keywords=num_keywords
        )

        return {
            "keywords": keywords,
            "semantic_keywords": semantic_keywords,
            "paper_data": paper_data
        }

    except Exception as e:
        print(f"🔥 Exception occurred during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🧹 Temporary file deleted: {temp_path}")
        else:
            print("⚠️ Temporary file not found for deletion.")

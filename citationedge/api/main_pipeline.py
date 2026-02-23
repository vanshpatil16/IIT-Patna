from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from typing import *
import tempfile
import os
from dotenv import load_dotenv
import numpy as np
import json
import re
import time
import functools
import threading
from datetime import datetime
from citationedge.services.pdf_processor import *
from citationedge.services.claim_extractor import extract_claims_from_paper_rag
from citationedge.services.citation_gap import analyze_citation_gaps
from citationedge.services.analyze_argumentation import argumentation_analysis
from citationedge.services.literary_scorer import analyze_literary_score
from citationedge.utils.date_helpers import *
from citationedge.utils.paper_json_processing import *
from citationedge.utils.shortlist import *
from citationedge.services.report_generator import create_citation_report

load_dotenv()

print("📦 [main_pipeline.py] Module loaded")

router = APIRouter()
print("✅ Router initialized")


class TimingLogger:
    def __init__(self):
        self.start_time = None
        self.section_times = {}
        self.total_start = None
    
    def start_total(self):
        """Start timing the entire process"""
        self.total_start = time.time()
        print(f"\n🚀 STARTING PAPER ANALYSIS at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
    
    def log_section(self, section_name, start_time=None):
        """Log timing for a major section"""
        if start_time is None:
            # Starting a section
            self.section_times[section_name] = time.time()
            print(f"\n📍 STARTING: {section_name} at {datetime.now().strftime('%H:%M:%S')}")
            return time.time()
        else:
            # Ending a section
            duration = time.time() - start_time
            print(f"✅ COMPLETED: {section_name} - Duration: {duration:.2f}s ({duration/60:.2f} min)")
            return duration
    
    def log_step(self, step_name, duration=None, start_time=None):
        """Log timing for individual steps"""
        if duration is not None:
            print(f"   ⏱️  {step_name}: {duration:.2f}s")
        elif start_time is not None:
            duration = time.time() - start_time
            print(f"   ⏱️  {step_name}: {duration:.2f}s")
            return duration
    
    def end_total(self):
        """End timing and show summary"""
        if self.total_start:
            total_duration = time.time() - self.total_start
            print("\n" + "=" * 80)
            print(f"🏁 TOTAL ANALYSIS TIME: {total_duration:.2f}s ({total_duration/60:.2f} min)")
            print(f"   Finished at {datetime.now().strftime('%H:%M:%S')}")
            return total_duration


def time_function(func_name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start = time.time()
            print(f"   🔄 Starting {name}...")
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                print(f"   ✅ {name} completed: {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                print(f"   ❌ {name} failed after {duration:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator


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


@router.post("/citation-edge-paper")
async def citation_edge_paper(
    pdf_file: UploadFile = File(...),
    is_quick: bool = True,
    generate_report: bool = True,
):
    """Analyze PDF paper and optionally generate citation report"""
    
    timer = TimingLogger()
    timer.start_total()
    
    print("📥 Received request to /citation-edge-paper")
    print(f"📄 Uploaded file name: {pdf_file.filename}")
    
    # File validation timing
    validation_start = timer.log_section("File Validation")
    if not pdf_file.filename.endswith('.pdf'):
        print("❌ File is not a PDF. Aborting.")
        raise HTTPException(status_code=400, detail="File must be a PDF")
    timer.log_section("File Validation", validation_start)

    # File processing timing
    file_proc_start = timer.log_section("File Processing")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        print("📂 Created temporary file")
        content = await pdf_file.read()
        temp_file.write(content)
        temp_path = temp_file.name
        print(f"📝 File saved to temp path: {temp_path}")
    timer.log_section("File Processing", file_proc_start)

    try:
        # PDF Analysis (Keywords & Semantic Keywords)
        pdf_analysis_start = timer.log_section("PDF Analysis & Keyword Extraction")
        print("🚀 Starting PDF analysis using run_paper_analysis()")
        
        # Time the actual analysis function
        analysis_step_start = time.time()
        keywords, semantic_keywords, paper_json = run_paper_analysis(pdf_path=temp_path)
        timer.log_step("run_paper_analysis", start_time=analysis_step_start)
        
        # Time keyword processing
        keyword_proc_start = time.time()
        final_keywords = set(i.lower() for i in keywords) 
        final_keywords.update(i.lower() for i in semantic_keywords)
        final_keywords = list(final_keywords)
        timer.log_step("keyword processing", start_time=keyword_proc_start)
        
    
        paper_json = paper_to_json(paper_json)
        
        timer.log_section("PDF Analysis & Keyword Extraction", pdf_analysis_start)

        # Parallel Processing Section
        parallel_start = timer.log_section("Parallel Claims & Citation Gap Analysis")
        
        # Variables to store results
        claims = None
        categorized_gaps = None
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Function to extract claims
        def extract_claims_thread():
            nonlocal claims
            print("🔄 Starting claims extraction in parallel...")
            claim_extract_start = time.time()
            claims = extract_claims_from_paper_rag(paper_json, groq_api_key)
            timer.log_step("extract_claims_from_paper_rag", start_time=claim_extract_start)
            
            shortlist_start = time.time()
            claims = shortlist_top_claims(claims)
            timer.log_step("shortlist_top_claims", start_time=shortlist_start)
            
            print(f"📌 Extracted {len(claims)} claims")
        
        # Function to analyze citation gaps
        def analyze_gaps_thread():
            nonlocal categorized_gaps
            print("🔄 Starting citation gap analysis in parallel...")
            gap_analysis_start = time.time()
            categorized_gaps = analyze_citation_gaps(paper_json, semantic_keywords)
            timer.log_step("analyze_citation_gaps", start_time=gap_analysis_start)
            
            if is_quick:
                gap_shortlist_start = time.time()
                categorized_gaps = shortlist_gaps(categorized_gaps)
                timer.log_step("shortlist_gaps", start_time=gap_shortlist_start)
            
            print(f"📌 Extracted {len(categorized_gaps)} gaps")
        
        # Create and start threads
        claims_thread = threading.Thread(target=extract_claims_thread)
        gaps_thread = threading.Thread(target=analyze_gaps_thread)
        
        claims_thread.start()
        gaps_thread.start()
        
        # Wait for both threads to complete
        claims_thread.join()
        gaps_thread.join()
        
        timer.log_section("Parallel Claims & Citation Gap Analysis", parallel_start)
        
        # Argumentation Analysis (depends on claims, so runs after parallel section)
        arg_analysis_start = timer.log_section("Argumentation Analysis")
        
        arg_step_start = time.time()
        claim_analysis = argumentation_analysis(paper_json, claims)
        timer.log_step("argumentation_analysis", start_time=arg_step_start)
        
        timer.log_section("Argumentation Analysis", arg_analysis_start)
        
        # Literary Score Analysis (depends on gaps and claim_analysis, so runs after)
        literary_start = timer.log_section("Literary Score Analysis")
        
        literary_step_start = time.time()
        literary_score = analyze_literary_score(paper_json, categorized_gaps, claim_analysis)
        timer.log_step("analyze_literary_score", start_time=literary_step_start)
        
        print("📌 Generated literary score analysis")
        timer.log_section("Literary Score Analysis", literary_start)
        
        conversion_start = timer.log_section("Data Type Conversion")
        
        conv_start = time.time()
        analysis_data = {
            "keywords": final_keywords,
            "paper_json": paper_json,
            "claims": convert_numpy_types(claims),
            "categorized_gaps": convert_numpy_types(categorized_gaps),
            "claim_analysis": convert_numpy_types(claim_analysis),
            "literary_score_report": convert_numpy_types(literary_score)
        }
        timer.log_step("convert_numpy_types for all data", start_time=conv_start)
        
        timer.log_section("Data Type Conversion", conversion_start)
        
        # Generate Report (Optional)
        report_bytes = None
        if generate_report:
            report_start = timer.log_section("Report Generation")
            
            try:
                report_gen_start = time.time()
                print("📊 Generating citation report...")
                
                # Generate report as bytes
                report_bytes = create_citation_report(
                    json_data=analysis_data,
                    groq_api_key=groq_api_key,
                    return_bytes=True  # Make sure your function supports this
                )
                
                timer.log_step("create_citation_report", start_time=report_gen_start)
                
                if report_bytes:
                    print(f"✅ Report generated successfully: {len(report_bytes)} bytes")
                else:
                    print("⚠️ Report generation returned None")
                    
            except Exception as e:
                print(f"❌ Report generation failed: {str(e)}")
                # Don't fail the entire request if report generation fails
                report_bytes = None
            
            timer.log_section("Report Generation", report_start)
        
        # End total timing
        total_time = timer.end_total()
        
        # Prepare final response
        result_data = {
            "analysis": analysis_data,
            "report_available": report_bytes is not None,
            "_timing_info": {
                "total_duration_seconds": total_time,
                "total_duration_minutes": total_time / 60,
                "analysis_completed_at": datetime.now().isoformat(),
                "report_generated": report_bytes is not None
            }
        }
        
        # Add report data if generated
        if report_bytes:
            import base64
            result_data["report_pdf"] = base64.b64encode(report_bytes).decode('utf-8')
            result_data["report_size_bytes"] = len(report_bytes)
        
        return result_data

    except Exception as e:
        print(f"🔥 Exception occurred during analysis: {str(e)}")
        timer.end_total()  # Still log total time even on failure
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cleanup_start = time.time()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🧹 Temporary file deleted: {temp_path}")
        else:
            print("⚠️ Temporary file not found for deletion.")
        
        cleanup_duration = time.time() - cleanup_start
        print(f"   ⏱️  Cleanup: {cleanup_duration:.2f}s")

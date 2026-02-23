from citationedge.models.paper import *
from typing import *
import os
import subprocess
import json
import threading
from citationedge.services.keyword_extractor import *

def run_paper_analysis(pdf_path: str, num_keywords: int = 10) -> Tuple[List[str], List[str], Paper]:
    """
    Run the complete paper analysis pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        num_keywords: Number of keywords to extract
        
    Returns:
        Tuple containing (keywords, semantic_keywords, paper_data)
    """
    
    try:
        
        paper_data = process_paper(pdf_path)
        
        
        
        # Variables to store results
        keywords = None
        sem_keywords = None
        
        # Function to run extract_keywords
        def extract_keywords_thread():
            nonlocal keywords
            keywords = extract_keywords(paper_data, num_keywords)
            
        
        # Function to run generate_semantic_keywords
        def generate_semantic_keywords_thread():
            nonlocal sem_keywords
            sem_keywords = generate_semantic_keywords(paper_data, num_keywords)
            
        
        # Create and start threads
        thread1 = threading.Thread(target=extract_keywords_thread)
        thread2 = threading.Thread(target=generate_semantic_keywords_thread)
        
        thread1.start()
        thread2.start()
        
        # Wait for both threads to complete
        thread1.join()
        thread2.join()
        
        
        return keywords, sem_keywords, paper_data
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed with error: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise

# Rest of your code remains the same...
def process_paper(file_path: str, is_pdf: bool = True) -> Paper:
    """
    Process a paper file, either PDF or JSON.
    
    Args:
        file_path: Path to the paper file
        is_pdf: Whether the file is a PDF (True) or JSON (False)
        
    Returns:
        Dict containing structured paper content
    """
    
    
    if is_pdf:
        
        try:
            json_path = extract_text_from_pdf(file_path)
            
            
            if not json_path:
                print(f"[ERROR] Failed to extract text from PDF")
                raise ValueError("Failed to extract text from PDF")
                
                
            
            paper = process_json_from_science_parser(json_path)
            
            return paper
            
        except Exception as e:
            print(f"[ERROR] Error in PDF processing: {str(e)}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            raise
    else:
        pass
        # Handle JSON file processing if needed


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract structured text from PDF using Science Parse.
    
    Args:
        pdf_path: Path to a PDF file
    
    Returns:
        Path to the generated JSON file
    """
    
    
    output_dir = "./json"
    

    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    

    # Get the folder and filename from the full path
    folder = os.path.dirname(pdf_path)
    pdf_file = os.path.basename(pdf_path)
    

    base_filename = os.path.splitext(pdf_file)[0]
    expected_json_path = os.path.join(output_dir, base_filename + ".pdf.json")
    

    # If already parsed
    if os.path.exists(expected_json_path):
        return expected_json_path

    
    # Run science-parse on the folder
    command = [
    "Java/jdk-1.8/bin/java.exe", 
    "-Xmx8g",
    "-jar", 
    "science-parse-cli-assembly-2.0.3.jar", 
    folder, 
    "-o", 
    output_dir
    ]
    
    print(f"[DEBUG] Running science-parse...")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        print(f"[DEBUG] Science-parse completed")
        
        if result.returncode == 0:
            print(f"[DEBUG] Science-parse executed successfully")
            if os.path.exists(expected_json_path):
                file_size = os.path.getsize(expected_json_path)
                return expected_json_path
            else:
                print(f"[ERROR] Expected JSON file was not created")
                return ""
        else:
            print(f"[ERROR] Science-parse failed with return code: {result.returncode}")
            return ""
            
    except Exception as e:
        print(f"[ERROR] Exception during science-parse execution: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        return ""

def process_json_from_science_parser(json_path: str) -> Paper:
    """
    Process JSON output from Science Parse and structure the data using Pydantic.
    
    Args:
        json_path: Path to the JSON file from Science Parse
        
    Returns:
        Paper object containing structured paper content
    """
    print(f"[DEBUG] Processing JSON from science parser: {json_path}")
    print(f"[DEBUG] JSON file exists: {os.path.exists(json_path)}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        root = data
        metadata = root.get("metadata", {})
        
        # Process authors
        authors = []
        author_data_list = metadata.get("authors", [])
        
        for i, author_data in enumerate(author_data_list):
            if isinstance(author_data, str):
                authors.append(Author(name=author_data))
            elif isinstance(author_data, dict):
                authors.append(Author(**author_data))
        
        
        sections = {}
        full_text = []
        
        if "sections" in metadata:
            sections_data = metadata["sections"]
            print(f"[DEBUG] Number of sections to process: {len(sections_data)}")
            
            for i, section_data in enumerate(sections_data):
                heading = section_data.get("heading", "")
                text = section_data.get("text", "")
                
                heading = "Abstract" if heading is None else heading
                sections[heading] = Section(
                    text=text,
                    original_section=section_data
                )
                
                if heading:
                    full_text.append(f"{heading}\n{text}")
                else:
                    full_text.append(text)
        else:
            pass
        
        
        
        # Process references
        
        references = []
        if "references" in metadata:
            references_data = metadata["references"]
            
            
            for i, ref_data in enumerate(references_data):
                
                
                # Handle None values by providing default empty strings
                reference = Reference(
                    title=ref_data.get("title", "") or "",
                    authors=ref_data.get("author", []) or [],
                    venue=ref_data.get("venue", "") or "",  # Fix: Handle None values
                    cite_regex=ref_data.get("citeRegEx", "") or "",
                    short_cite_regex=ref_data.get("shortCiteRegEx", "") or "",
                    year=ref_data.get("year", 0) or 0,
                    original_reference=ref_data
                )
                references.append(reference)
        else:
            print(f"[DEBUG] No references found in metadata")
        
        print(f"[DEBUG] Total references processed: {len(references)}")
        
        # Process reference mentions
        
        reference_mentions = []
        mention_data_list = metadata.get("referenceMentions", [])
        
        
        for i, mention_data in enumerate(mention_data_list):
            if isinstance(mention_data, dict):
                reference_mentions.append(ReferenceMention(**mention_data))
            else:
                reference_mentions.append(ReferenceMention())

        title = metadata.get("title", "") or ""
        abstract = metadata.get("abstractText", "") or ""
        full_text_combined = "\n\n".join(full_text)
       

        paper = Paper(
            file_name=root.get("name", "") or "",
            title=title,
            abstract=abstract,
            full_text=full_text_combined,
            sections=sections,
            references=references,
            authors=authors,
            emails=metadata.get("emails", []) or [],
            year=metadata.get("year", 0) or 0,
            creator=metadata.get("creator", "") or "",
            source=metadata.get("source", "") or "",
            reference_mentions=reference_mentions
        )

        return paper
        
    except Exception as e:
        print(f"[ERROR] Error processing JSON file: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise

from typing import *
import json
import re

def robust_json_parse(response_content: str, function_name: str) -> Dict[str, Any]:
    """
    Robust JSON parsing with multiple fallback strategies.
    
    Args:
        response_content: Raw response content from API
        function_name: Name of calling function for error reporting
    
    Returns:
        Parsed JSON dictionary or empty dict if parsing fails
    """
    try:
        content = response_content.strip()
        
        # Remove common markdown formatting
        content = content.replace('```json', '').replace('```', '')
        content = content.strip()
        
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            pass
        
        # Handle multiple JSON objects - extract the first valid one
        json_objects = []
        brace_count = 0
        current_json = ""
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                current_json += char
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                current_json += char
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                
            current_json += char
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                    # Complete JSON object found
                    if brace_count == 0 and current_json.strip():
                        try:
                            parsed = json.loads(current_json.strip())
                            json_objects.append(parsed)
                            current_json = ""
                        except json.JSONDecodeError:
                            pass
        
        # If we found multiple JSON objects, merge evidence arrays
        if json_objects:
            if len(json_objects) == 1:
                return json_objects[0]
            else:
                # Merge multiple evidence arrays
                merged_evidence = []
                for obj in json_objects:
                    if isinstance(obj, dict) and "evidence" in obj:
                        evidence = obj["evidence"]
                        if isinstance(evidence, list):
                            merged_evidence.extend(evidence)
                
                return {"evidence": merged_evidence}
        
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Clean up potential formatting issues
        if '{' in content and '}' in content:
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_content = content[json_start:json_end + 1]
                try:
                    result = json.loads(cleaned_content)
                    return result
                except json.JSONDecodeError:
                    pass
        return {}
        
    except Exception as e:
        print(f"Unexpected error in {function_name}: {e}")
        return {}

def determine_section_type(section_name: str) -> str:
    """
    Determine the type of section based on its name.
    
    Args:
        section_name: Name of the section
    
    Returns:
        Section type classification
    """
    section_lower = section_name.lower()
    
    type_mappings = {
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "background", "overview"],
        "method": ["method", "approach", "technique", "implementation", "model", "architecture"],
        "results": ["result", "evaluation", "experiment", "performance", "validation"],
        "discussion": ["discuss", "conclusion", "implication", "future", "limitation"],
        "related_work": ["related", "literature", "prior", "previous"]
    }
    
    for section_type, keywords in type_mappings.items():
        if any(keyword in section_lower for keyword in keywords):
            return section_type
    
    return "other"

def get_section_content(paper_json: Dict[str, Any], section_name: str) -> str:
    """
    Get the content of a specific section with fallback mechanisms.
    
    Args:
        paper_json: Complete paper JSON
        section_name: Name of the section to retrieve
    
    Returns:
        Section content as string
    """
    if section_name == "Abstract" and "abstractText" in paper_json["metadata"]:
        return paper_json["metadata"]["abstractText"]
    
    for section in paper_json["metadata"]["sections"]:
        curr_heading = section.get("heading", "")
        if curr_heading and (curr_heading == section_name or curr_heading.endswith(". " + section_name)):
            return section.get("text", "")
    
    # Fuzzy matching
    best_match = ""
    best_score = 0
    
    for section in paper_json["metadata"]["sections"]:
        curr_heading = section.get("heading", "")
        if not curr_heading:
            continue
            
        section_words = set(curr_heading.lower().split())
        target_words = set(section_name.lower().split())
        
        if target_words:
            overlap = len(section_words.intersection(target_words))
            score = overlap / len(target_words)
            
            if score > best_score:
                best_score = score
                best_match = section.get("text", "")
    
    return best_match if best_score > 0.5 else ""

def normalize(name):
    return name.lower().replace('.', '').strip()

def extract_lastname(name):
    parts = normalize(name).split()
    return parts[-1] if parts else ""

def extract_initials(name):
    parts = normalize(name).split()
    return ''.join(part[0] for part in parts[:-1]) 

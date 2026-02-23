from citationedge.utils.date_helpers import *
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

def create_llm_prompt(score_report, paper_json):
    """Create a prompt for the LLM to generate literary analysis."""
    overall_score = score_report["overall_score"]
    component_scores = score_report["component_scores"]
    metrics = score_report["metrics"]
    rating = score_report["rating"]
    
    paper_title = paper_json["metadata"].get("title", "Unknown Paper")
    paper_abstract = paper_json["metadata"].get("abstract", "")
    paper_year = paper_json["metadata"].get("year", get_current_year())
    
    prompt = f"""
    You are an academic literature review expert tasked with providing an insightful analysis of a scientific paper's citation practices.
    
    ## Paper Information:
    - Title: {paper_title}
    - Publication Year: {paper_year}
    - Abstract: {paper_abstract[:500]}...
    
    ## Citation Metrics:
    - Overall Literary Score: {overall_score:.1f}% ({rating})
    - Total References: {metrics['total_references']}
    - Citation Gaps: {metrics['citation_gaps']}
    - Citation Coverage Ratio: {metrics['citation_coverage_ratio']*100:.1f}%
    - Average Citation Age: {metrics['average_citation_age'] if metrics['average_citation_age'] is not None else 'N/A'} years
    - Self-Citation Percentage: {metrics['self_citation_percentage']:.1f}%
    
    ## Component Scores:
    - Citation Completeness: {component_scores['citation_completeness']:.1f}%
    - Citation Relevance: {component_scores['citation_relevance']:.1f}%
    - Citation Diversity: {component_scores['citation_diversity']:.1f}%
    - Citation Recency: {component_scores['citation_recency']:.1f}%
    - Self-Citation Ratio: {component_scores['self_citation_ratio']:.1f}%
    
    Based on this data, provide a comprehensive analysis in JSON format with the following structure:
    {{
        "overall_assessment": "A paragraph summarizing the overall citation quality of the paper, highlighting key strengths and weaknesses.",
        "strengths": ["List key strengths of the paper's citation practices (3-5 items)"],
        "weaknesses": ["List key weaknesses of the paper's citation practices (3-5 items)"],
        "recommendations": ["Provide specific recommendations for improving the paper's citations (3-5 items)"]
    }}
    
    Ensure your analysis is specific, actionable, and focused on citation practices rather than the paper's content.
    """
    
    return prompt

def call_groq_api(prompt, api_key=None):
    """Call the Groq API with the given prompt."""
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-70b-8192",  # Use Llama 3 70B model
        "messages": [
            {"role": "system", "content": "You are an academic literature review expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def parse_llm_response(llm_response, score_report):
    """Parse the LLM response and ensure it has the correct structure."""
    try:
        # Try to parse the JSON response
        analysis = json.loads(llm_response)
        
        required_keys = ["overall_assessment", "strengths", "weaknesses", "recommendations"]
        for key in required_keys:
            if key not in analysis:
                analysis[key] = []
        
        for key in ["strengths", "weaknesses", "recommendations"]:
            if not isinstance(analysis[key], list):
                analysis[key] = [analysis[key]]
        
        return analysis
        
    except json.JSONDecodeError:
        analysis = {
            "overall_assessment": "",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        if "overall_assessment" in llm_response.lower():
            parts = llm_response.split("overall_assessment", 1)[1].split("\n", 1)
            if len(parts) > 0:
                assessment = parts[0].strip(": \"',")
                analysis["overall_assessment"] = assessment
        
        # Extract lists
        for section in ["strengths", "weaknesses", "recommendations"]:
            if section in llm_response.lower():
                section_text = llm_response.split(section, 1)[1]
                items = []
                for line in section_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("*"):
                        items.append(line.strip("- *"))
                    elif line.startswith('"') and line.endswith('"'):
                        items.append(line.strip('"'))
                
                if items:
                    analysis[section] = items[:5]  # Limit to 5 items
        return analysis

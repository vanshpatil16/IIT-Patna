from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import spacy
import torch
from typing import List, Dict, Any, Set
import json
from dotenv import load_dotenv
import os
load_dotenv()
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from citationedge.utils.date_helpers import *

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model.eval()
nlp = spacy.load("en_core_web_lg")
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
SEMANTIC_SCHOLAR_API = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Initialize LLM for generating explanations
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def embed_text(keywords: List[str]) -> List[float]:
    """Embed text using SciBERT model."""
    text = " ".join(keywords)  # combine into a single string
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        out = model(**encoded, return_dict=True)
    return out.last_hidden_state.mean(dim=1).squeeze().tolist()

def parse_search_results(raw_results):
    """
    Parse the raw search results from Semantic Scholar into a structured format
    """
    papers = []
    
    paper_blocks = raw_results.strip().split("\n\n\n")
    
    for block in paper_blocks:
        if not block.strip():
            continue
            
        paper_info = {}
        lines = block.strip().split("\n")
        
        for line in lines:
            if not line.strip():
                continue
                
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "Published year":
                    # Handle None or invalid year values
                    try:
                        paper_info["year"] = int(value) if value and value != 'None' else 2024
                    except (ValueError, TypeError):
                        paper_info["year"] = 2024  # Default year
                elif key == "Title":
                    paper_info["title"] = value
                elif key == "Authors":
                    paper_info["authors"] = [author.strip() for author in value.split(",")]
                elif key == "Abstract":
                    paper_info["abstract"] = value
        
        if paper_info:
            papers.append(paper_info)
    return papers

def safe_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity with safety checks for empty vectors."""
    # Clean and validate input text
    text1 = str(text1).strip() if text1 else ""
    text2 = str(text2).strip() if text2 else ""
    
    # Return 0 similarity if either text is empty or too short
    if len(text1) < 5 or len(text2) < 5:
        return 0.0
    
    try:
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        
        # Check if vectors exist and are not empty
        if doc1.has_vector and doc2.has_vector and doc1.vector_norm > 0 and doc2.vector_norm > 0:
            return doc1.similarity(doc2)
        else:
            return 0.0
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def safe_year_conversion(year_value) -> int:
    
    fallback_year = get_current_year()
    if year_value is None:
        return fallback_year
    
    if isinstance(year_value, int):
        return year_value
    
    if isinstance(year_value, str):
        # Handle common string cases
        year_str = year_value.strip().lower()
        if year_str in ['none', 'null', '', 'unknown']:
            return fallback_year
        
        try:
            return int(year_str)
        except ValueError:
            # Try to extract year from string (e.g., "Published in 2023")
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
            if year_match:
                return int(year_match.group())
            return fallback_year
    
    return fallback_year

def generate_citation_explanation(source_paper: Dict[str, Any], cited_paper: Dict[str, Any], 
                                relevant_section: str, keywords: List[str]) -> Dict[str, str]:
    """
    Generate explanations for why and how a citation helps the source paper.
    
    Args:
        source_paper: The user's paper
        cited_paper: The paper being recommended for citation
        relevant_section: The section where this citation is most relevant
        keywords: Keywords that led to finding this paper
        
    Returns:
        Dictionary containing different types of explanations
    """
    
    # Create prompt template for generating explanations
    explanation_prompt = PromptTemplate(
        input_variables=["source_title", "source_abstract", "cited_title", "cited_abstract", 
                        "relevant_section", "keywords"],
        template="""
        You are an expert academic researcher helping to explain why a citation would be valuable for a research paper.
        
        SOURCE PAPER:
        Title: {source_title}
        Abstract: {source_abstract}
        
        RECOMMENDED CITATION:
        Title: {cited_title}
        Abstract: {cited_abstract}
        
        MOST RELEVANT SECTION: {relevant_section}
        MATCHING KEYWORDS: {keywords}
        
        Please provide a comprehensive explanation of how this citation helps the source paper by addressing:
        
        1. WHY this citation is relevant (what specific aspects connect to the source paper)
        2. HOW it can be used (what it contributes - methodology, evidence, theory, etc.)
        3. WHERE it fits best (specific section or context within the paper)
        4. WHAT it adds (gap it fills, support it provides, perspective it offers)
        
        Format your response as a JSON object with these keys:
        - "relevance_explanation": Why this citation is relevant
        - "contribution_type": What type of contribution it makes (methodology, evidence, theory, etc.)
        - "usage_suggestion": How to use this citation in the paper
        - "section_fit": Where it fits best and why
        - "value_added": What specific value it adds to the paper
        - "citation_context": Suggested context for citing this paper
        
        Keep explanations concise but informative (2-3 sentences each).
        """
    )
    
    # Create the new RunnableSequence chain (replaces LLMChain)
    explanation_chain = explanation_prompt | llm
    
    try:
        # Generate explanation using invoke method
        result = explanation_chain.invoke({
            "source_title": source_paper["metadata"].get("title", ""),
            "source_abstract": source_paper["metadata"].get("abstractText", ""),
            "cited_title": cited_paper.get("title", ""),
            "cited_abstract": cited_paper.get("abstract", ""),
            "relevant_section": relevant_section,
            "keywords": ", ".join(keywords)
        })
        
        # Extract content from AIMessage object
        result_content = result.content if hasattr(result, 'content') else str(result)

        if '```' in result_content:
            # Find the JSON between the backticks
            start = result_content.find('```') + 3
            end = result_content.rfind('```')
            result_content = result_content[start:end].strip()
        try:
            explanation_dict = json.loads(result_content)
            return explanation_dict
        except json.JSONDecodeError:
            
            return {
                "json_parsing":"False",
                "relevance_explanation": f"This paper is relevant based on shared concepts and methodology related to {', '.join(keywords)}.",
                "contribution_type": "Supporting evidence",
                "usage_suggestion": f"Can be cited in the {relevant_section} section to support key arguments.",
                "section_fit": f"Best fits in {relevant_section} section due to thematic alignment.",
                "value_added": "Provides additional perspective and empirical support for the research claims.",
                "citation_context": "Use to strengthen theoretical foundation or provide comparative analysis."
            }
    
    except Exception as e:
        print(f"Error generating explanation: {e}")
        # Return fallback explanation
        return {
            "relevance_explanation": f"This paper shares key concepts with your work, particularly around {', '.join(keywords)}.",
            "contribution_type": "Supporting research",
            "usage_suggestion": f"Consider citing in {relevant_section} section for additional support.",
            "section_fit": f"Aligns well with {relevant_section} section themes.",
            "value_added": "Provides complementary research perspective and findings.",
            "citation_context": "Use to support your arguments with additional empirical evidence."
        }

def analyze_citation_relationship(source_paper: Dict[str, Any], cited_paper: Dict[str, Any]) -> Dict[str, str]:
    """
    Analyze the specific relationship between source and cited papers.
    
    Returns:
        Dictionary with relationship analysis
    """
    source_text = f"{source_paper['metadata'].get('title', '')} {source_paper['metadata'].get('abstractText', '')}"
    cited_text = f"{cited_paper.get('title', '')} {cited_paper.get('abstract', '')}"
    
    # Extract key concepts using NLP
    source_doc = nlp(source_text)
    cited_doc = nlp(cited_text)
    
    # Extract entities and key phrases
    source_entities = [ent.text.lower() for ent in source_doc.ents]
    cited_entities = [ent.text.lower() for ent in cited_doc.ents]
    
    # Find common entities
    common_entities = set(source_entities) & set(cited_entities)
    
    # Determine relationship type
    relationship_type = "supporting"  # default
    if len(common_entities) > 3:
        relationship_type = "highly_related"
    elif any(method_term in cited_text.lower() for method_term in ["method", "approach", "technique", "algorithm"]):
        relationship_type = "methodological"
    elif any(theory_term in cited_text.lower() for theory_term in ["theory", "framework", "model", "concept"]):
        relationship_type = "theoretical"
    
    return {
        "relationship_type": relationship_type,
        "common_concepts": list(common_entities)[:5],  # Top 5 common concepts
        "strength": "strong" if len(common_entities) > 2 else "moderate"
    }

def generate_contextual_citation_suggestions(source_paper: Dict[str, Any], cited_paper: Dict[str, Any], 
                                           relevant_section: str) -> List[str]:
    """
    Generate specific citation context suggestions.
    
    Returns:
        List of suggested citation contexts
    """
    suggestions = []
    
    # Analyze the relationship
    relationship = analyze_citation_relationship(source_paper, cited_paper)
    
    if relationship["relationship_type"] == "methodological":
        suggestions.extend([
            f"Cite when discussing methodology in {relevant_section} section",
            "Reference for comparative analysis of methods",
            "Use to support chosen methodological approach"
        ])
    elif relationship["relationship_type"] == "theoretical":
        suggestions.extend([
            f"Cite for theoretical foundation in {relevant_section} section",
            "Reference for conceptual framework development",
            "Use to support theoretical arguments"
        ])
    elif relationship["relationship_type"] == "highly_related":
        suggestions.extend([
            f"Cite as closely related work in {relevant_section} section",
            "Reference for comparative discussion",
            "Use to highlight research gap or contribution"
        ])
    else:
        suggestions.extend([
            f"Cite as supporting evidence in {relevant_section} section",
            "Reference for additional context",
            "Use to broaden literature review"
        ])
    
    return suggestions

def analyze_citation_gaps(paper_json: Dict[str, Any], keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Main function to find relevant papers that should be cited but aren't.
    Now includes detailed explanations for each citation.
    
    Args:
        paper_json: Dictionary containing paper metadata and content
        keywords: List of keywords extracted from the paper
        
    Returns:
        List of categorized citation gaps with explanations
    """
    # Extract paper metadata
    paper_id = paper_json["metadata"].get("paper_id", "unknown")
    paper_title = paper_json["metadata"].get("title", "")
    paper_abstract = paper_json["metadata"].get("abstractText", "")
    
    # Get current citations
    current_citations = extract_current_citations(paper_json)
    
    # Find relevant papers based on the keywords
    relevant_papers = find_relevant_papers(keywords, paper_title)
    
    # Identify citation gaps (relevant papers that aren't cited)
    citation_gaps = []
    for paper in relevant_papers:
        if paper["title"].lower() in current_citations:
            continue
        
        # Calculate relevance
        relevance_score = calculate_relevance_score(paper, paper_json)
        
        # Add to gaps if relevant enough
        if relevance_score > 0.5:  # Threshold for relevance
            paper["relevance_score"] = relevance_score
            citation_gaps.append(paper)
    
    # Sort gaps by relevance
    citation_gaps.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Categorize gaps and add explanations
    categorized_gaps = categorize_citation_gaps_with_explanations(citation_gaps, paper_json, keywords)

    return categorized_gaps

def categorize_citation_gaps_with_explanations(citation_gaps: List[Dict[str, Any]], 
                                             paper_json: Dict[str, Any], 
                                             keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Categorize citation gaps by relevance to different paper sections and add detailed explanations.
    """
    # Extract sections from paper
    sections = {}
    if "sections" in paper_json["metadata"]:
        for section in paper_json["metadata"]["sections"]:
            heading = section.get("heading", "Unknown")
            text = section.get("text", "")
            if text:
                sections[heading] = text
    
    # Categorize each gap with explanations
    categorized_gaps = []
    for gap in citation_gaps:
        # Find most relevant section
        best_section = "General"
        best_score = 0
        
        for section_name, section_text in sections.items():
            # Limit section text for performance
            limited_section_text = section_text[:5000] if section_text else ""
            
            # Create text for gap paper
            gap_text = f"{gap.get('title', '')} {gap.get('abstract', '')}"
            
            # Calculate similarity using safe function
            similarity = safe_text_similarity(limited_section_text, gap_text)
            
            if similarity > best_score:
                best_score = similarity
                best_section = section_name
        
        # Categorize by importance
        relevance_score = gap.get("relevance_score", 0)
        importance = "high" if relevance_score > 0.75 else "medium" if relevance_score > 0.6 else "low"
        
        # Generate detailed explanation
        explanation = generate_citation_explanation(paper_json, gap, best_section, keywords)
        
        # Generate contextual suggestions
        context_suggestions = generate_contextual_citation_suggestions(paper_json, gap, best_section)
        
        # Analyze relationship
        relationship = analyze_citation_relationship(paper_json, gap)
        
        # Create enhanced categorized gap object
        categorized_gap = gap.copy()
        categorized_gap.update({
            "relevant_section": best_section,
            "importance": importance,
            "explanation": explanation,
            "context_suggestions": context_suggestions,
            "relationship": relationship,
            "section_similarity_score": best_score
        })
        
        categorized_gaps.append(categorized_gap)
    
    return categorized_gaps

def extract_current_citations(paper_json: Dict[str, Any]) -> Set[str]:
    """Extract currently cited papers from the paper JSON."""
    current_citations = set()
    
    # Extract from references field
    if "references" in paper_json["metadata"]:
        for ref in paper_json["metadata"]["references"]:
            # Also add by title matching (in case IDs don't match)
            ref_title = ref.get("title", "").lower()
            if ref_title:
                current_citations.add(ref_title)
    return current_citations

def find_relevant_papers(keywords: List[str], paper_title: str) -> List[Dict[str, Any]]:
    """Find relevant papers using Neo4j and/or Semantic Scholar."""
    relevant_papers = []
    
    neo4j_papers = find_papers_in_neo4j(keywords)
    relevant_papers.extend(neo4j_papers)
    
    if SEMANTIC_SCHOLAR_API:
        ss_papers = find_papers_in_semantic_scholar(keywords, paper_title)
        # Add unique papers from Semantic Scholar
        existing_titles = {p["title"] for p in relevant_papers}
        for paper in ss_papers:
            if paper["title"] not in existing_titles:
                relevant_papers.append(paper)
    
    print(f"FOUND {len(relevant_papers)} RELEVANT PAPERS")
    return relevant_papers

def find_papers_in_neo4j(keywords: List[str]) -> List[Dict[str, Any]]:
    """Find relevant papers in Neo4j knowledge graph."""
    # Query for papers matching keywords
    query_embedding = embed_text(keywords)
    query = """
    MATCH (p:Paper)-[:HAS_KEYWORD_GROUP]->(kg:KeywordGroup)
    WHERE any(searchTerm IN $keywords WHERE 
        toLower(p.title) CONTAINS toLower(searchTerm) OR 
        toLower(p.abstractText) CONTAINS toLower(searchTerm) OR 
        toLower(kg.extracted_keywords) CONTAINS toLower(searchTerm)
    )
    WITH p, 
        reduce(score = 0, searchTerm IN $keywords | 
        score + 
        CASE WHEN toLower(p.title) CONTAINS toLower(searchTerm) THEN 3 ELSE 0 END +
        CASE WHEN toLower(p.abstractText) CONTAINS toLower(searchTerm) THEN 1 ELSE 0 END +
        CASE WHEN toLower(kg.extracted_keywords) CONTAINS toLower(searchTerm) THEN 2 ELSE 0 END
        ) AS relevance_score
    WHERE relevance_score > 0
    RETURN DISTINCT p, relevance_score
    ORDER BY relevance_score DESC
    LIMIT 50
    """

    with driver.session() as session:
        results = list(session.run(query, keywords=keywords))
    
    # Format the results
    papers = []
    for result in results:
        paper_node = result["p"]  # Access the node from the result
        papers.append({
            "title": paper_node.get("title", "No Title"),
            "abstract": paper_node.get("abstractText", "No Abstract"),
            "year": safe_year_conversion(paper_node.get("year", 2024)),
            "source": paper_node.get("source", "Unknown")
        })
    print(f"FOUND {len(papers)} PAPERS FROM NEO4J")
    return papers

def find_papers_in_semantic_scholar(keywords: List[str], paper_title: str) -> List[Dict[str, Any]]:
    """Find relevant papers using Semantic Scholar API with robust error handling."""
    papers = []
    keywords = keywords[:10]
    
    # Configure session with retries and timeout
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    for i, query in enumerate(keywords):
        try:            
            # Option 1: If using SemanticScholarQueryRun with timeout configuration
            semantic_scholar = SemanticScholarQueryRun()
            
            # Add timeout handling
            search_results = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Set a reasonable timeout (30 seconds)
                    search_results = semantic_scholar.invoke(query, timeout=30)
                    break
                except Exception as timeout_error:
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        continue
            
            if search_results:
                structured_results = parse_search_results(search_results)
                
                for result in structured_results:
                    papers.append({
                        "title": result.get("title", ""),
                        "abstract": result.get("abstract", ""),
                        "year": safe_year_conversion(result.get("year", 2024)),
                        "authors": result.get("authors", ""),
                        "source": "semantic_scholar",
                        "query": query  # Track which query found this paper
                    })
            
            # Rate limiting - be respectful to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing keyword '{query}': {e}")
            continue  # Continue with next keyword instead of returning error
    return papers

def calculate_relevance_score(paper: Dict[str, Any], source_paper: Dict[str, Any]) -> float:
    """Calculate how relevant a paper is to the source paper."""
    # Extract text from source paper
    source_title = source_paper["metadata"].get("title", "")
    source_abstract = source_paper["metadata"].get("abstractText", "")
    source_text = f"{source_title} {source_abstract}"
    
    # Extract text from candidate paper
    candidate_title = paper.get("title", "")
    candidate_abstract = paper.get("abstract", "")
    candidate_text = f"{candidate_title} {candidate_abstract}"
    
    # Calculate text similarity using safe function
    text_similarity = safe_text_similarity(source_text, candidate_text)
    
    # Consider other factors
    
    # 1. Year recency factor (more recent papers get a boost)
    current_year = get_current_year()
    paper_year = safe_year_conversion(paper.get("year", current_year))
    year_factor = 1.0 - (0.05 * (current_year - paper_year))  # 5% penalty per year
    year_factor = max(0.5, min(1.0, year_factor))  # Cap between 0.5 and 1.0
    
    # 2. Citation count factor (if available)
    citation_count = paper.get("citation_count", 0)
    if citation_count and isinstance(citation_count, (int, float)):
        citation_factor = min(1 + (citation_count / 1000), 1.5)  # Cap at 1.5 boost
    else:
        citation_factor = 1.0
    
    # 3. Title-specific match factor using safe similarity
    title_similarity = safe_text_similarity(source_title, candidate_title)
    title_factor = 1.0 + (title_similarity * 0.5)  # Up to 1.5 boost
    
    # Calculate overall relevance
    relevance_score = text_similarity * year_factor * citation_factor * title_factor
    return relevance_score

def categorize_citation_gaps(citation_gaps: List[Dict[str, Any]], paper_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Categorize citation gaps by relevance to different paper sections."""
    # Extract sections from paper
    sections = {}
    if "sections" in paper_json["metadata"]:
        for section in paper_json["metadata"]["sections"]:
            heading = section.get("heading", "Unknown")
            text = section.get("text", "")
            if text:
                sections[heading] = text
    
    # Categorize each gap
    categorized_gaps = []
    for gap in citation_gaps:
        # Find most relevant section
        best_section = "General"
        best_score = 0
        
        for section_name, section_text in sections.items():
            # Limit section text for performance
            limited_section_text = section_text[:5000] if section_text else ""
            
            # Create text for gap paper
            gap_text = f"{gap.get('title', '')} {gap.get('abstract', '')}"
            
            # Calculate similarity using safe function
            similarity = safe_text_similarity(limited_section_text, gap_text)
            
            if similarity > best_score:
                best_score = similarity
                best_section = section_name
        
        # Categorize by importance
        relevance_score = gap.get("relevance_score", 0)
        importance = "high" if relevance_score > 0.75 else "medium" if relevance_score > 0.6 else "low"
        
        # Create categorized gap object
        categorized_gap = gap.copy()
        categorized_gap["relevant_section"] = best_section
        categorized_gap["importance"] = importance
        
        categorized_gaps.append(categorized_gap)
    return categorized_gaps

def format_citation_recommendations(citation_gaps: List[Dict[str, Any]]) -> str:
    """
    Format citation recommendations into a readable format for the user.
    """
    if not citation_gaps:
        return "No citation gaps found."
    
    formatted_output = "# Citation Gap Analysis Results\n\n"
    
    for i, gap in enumerate(citation_gaps, 1):
        formatted_output += f"## {i}. {gap.get('title', 'Unknown Title')}\n"
        formatted_output += f"**Authors:** {', '.join(gap.get('authors', ['Unknown']))}\n"
        formatted_output += f"**Year:** {gap.get('year', 'Unknown')}\n"
        formatted_output += f"**Relevance Score:** {gap.get('relevance_score', 0):.2f}\n"
        formatted_output += f"**Importance:** {gap.get('importance', 'Unknown').title()}\n"
        formatted_output += f"**Most Relevant Section:** {gap.get('relevant_section', 'Unknown')}\n\n"
        
        # Add explanations if available
        if 'explanation' in gap:
            exp = gap['explanation']
            formatted_output += "### Why This Citation Helps:\n"
            formatted_output += f"**Relevance:** {exp.get('relevance_explanation', 'Not specified')}\n"
            formatted_output += f"**Contribution Type:** {exp.get('contribution_type', 'Not specified')}\n"
            formatted_output += f"**How to Use:** {exp.get('usage_suggestion', 'Not specified')}\n"
            formatted_output += f"**Where to Cite:** {exp.get('section_fit', 'Not specified')}\n"
            formatted_output += f"**Value Added:** {exp.get('value_added', 'Not specified')}\n\n"
        
        # Add context suggestions if available
        if 'context_suggestions' in gap:
            formatted_output += "### Citation Context Suggestions:\n"
            for suggestion in gap['context_suggestions']:
                formatted_output += f"- {suggestion}\n"
            formatted_output += "\n"
        
        # Add relationship analysis if available
        if 'relationship' in gap:
            rel = gap['relationship']
            formatted_output += f"**Relationship Type:** {rel.get('relationship_type', 'Unknown').title()}\n"
            if rel.get('common_concepts'):
                formatted_output += f"**Common Concepts:** {', '.join(rel['common_concepts'])}\n"
            formatted_output += f"**Relationship Strength:** {rel.get('strength', 'Unknown').title()}\n\n"
        
        formatted_output += "---\n\n"
    
    return formatted_output

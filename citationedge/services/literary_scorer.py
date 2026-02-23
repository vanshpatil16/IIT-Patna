import numpy as np
from collections import Counter
from datetime import datetime
import networkx as nx
from py2neo import Graph
import json
import pandas as pd
import requests
import os
from citationedge.constants.config import DEFAULT_WEIGHTS
from citationedge.utils.model_initializer import *
from citationedge.utils.text_processing import *
from dotenv import load_dotenv
load_dotenv()

from citationedge.api.llm_interface import *
from citationedge.utils.date_helpers import get_current_year, calculate_average_citation_age
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Keep the original DEFAULT_WEIGHTS unchanged
DEFAULT_WEIGHTS = {
    "citation_completeness": 0.35,
    "citation_relevance": 0.25,
    "citation_diversity": 0.15,
    "citation_recency": 0.15,
    "self_citation_ratio": 0.10
}

def analyze_literary_score(paper_json, citation_gaps, claim_analysis):
    """
    Main function to calculate literary score and generate analysis.
    This function replicates the main functionality of the original class.
    """
    return generate_comprehensive_literary_report(
        paper_json=paper_json,
        citation_gaps=citation_gaps,
        claim_analysis=claim_analysis
    )

def calculate_completeness_score_dynamic(references, citation_gaps):
    """Calculate citation completeness score with dynamic scaling."""
    if not references and not citation_gaps:
        return 0.0
    
    total_relevant = len(references) + len(citation_gaps)
    actual_refs = len(references)
    
    # Base completeness ratio
    completeness_ratio = actual_refs / total_relevant if total_relevant > 0 else 0
    
    # Dynamic scaling based on gap severity
    gap_severity_penalty = 0
    if citation_gaps:
        high_relevance_gaps = sum(1 for gap in citation_gaps if gap.get("relevance_score", 0) > 0.8)
        medium_relevance_gaps = sum(1 for gap in citation_gaps if 0.5 <= gap.get("relevance_score", 0) <= 0.8)
        
        # Scale penalty based on proportion of high/medium relevance gaps
        total_gaps = len(citation_gaps)
        gap_severity_penalty = (
            (high_relevance_gaps / total_gaps) * 0.3 +
            (medium_relevance_gaps / total_gaps) * 0.1
        ) if total_gaps > 0 else 0
    
    # Apply sigmoid scaling to make scores more dynamic
    raw_score = completeness_ratio - gap_severity_penalty
    
    # Sigmoid transformation: makes middle scores more sensitive to changes
    def sigmoid_scale(x, steepness=5):
        return 1 / (1 + np.exp(-steepness * (x - 0.5)))
    
    scaled_score = sigmoid_scale(raw_score)
    
    return max(0.0, min(scaled_score, 1.0))

def calculate_relevance_score_dynamic(references, claim_analysis):
    """Calculate citation relevance score with dynamic assessment."""
    if not references or not claim_analysis:
        return 0.5
    
    claim_arguments = claim_analysis.get("claim_arguments", [])
    
    # Dynamic quality assessment based on evidence distribution
    quality_weights = {"strong": 1.0, "moderate": 0.7, "weak": 0.3}
    
    weighted_support = 0
    total_weight = 0
    
    for claim in claim_arguments:
        evidence = claim.get("evidence", [])
        quality = claim.get("argument_quality", "moderate")
        weight = quality_weights.get(quality, 0.5)
        
        if evidence:
            weighted_support += weight
        total_weight += weight
    
    support_ratio = weighted_support / total_weight if total_weight > 0 else 0
    
    # Dynamic relevance calculation
    relevance_scores = []
    for claim in claim_arguments:
        for evidence in claim.get("evidence", []):
            relevance_scores.append(evidence.get("relevance", 0))
    
    if relevance_scores:
        # Use weighted average with emphasis on higher relevance scores
        sorted_scores = sorted(relevance_scores, reverse=True)
        
        # Give more weight to top relevance scores
        weights = [1.0 / (i + 1) for i in range(len(sorted_scores))]
        weighted_avg = sum(score * weight for score, weight in zip(sorted_scores, weights)) / sum(weights)
        
        # Combine support ratio and weighted relevance
        relevance_score = (support_ratio * 0.6) + (weighted_avg * 0.4)
    else:
        relevance_score = support_ratio
    
    
    return max(0.0, min(relevance_score ** 0.8, 1.0))

def calculate_diversity_score_dynamic(references):
    """Calculate citation diversity score with dynamic scaling."""
    if not references:
        return 0.0
    
    total_refs = len(references)
    
    
    venues = [ref.get("venue", "Unknown") for ref in references]
    years = [ref.get("year", 0) for ref in references if ref.get("year", 0) > 0]
    
    
    author_groups = []
    for ref in references:
        authors = ref.get("author", [])
        if authors:
            author_groups.append(frozenset(authors))
    
    
    unique_venues = len(set(venues))
    unique_years = len(set(years)) if years else 0
    unique_author_groups = len(set(author_groups))
    
    
    ref_count_factor = min(total_refs / 20, 1.0) 
    
    
    venue_diversity = (unique_venues / total_refs) * (1 + (1 - ref_count_factor) * 0.3)
    
    
    author_diversity = (unique_author_groups / total_refs) * (1 + (1 - ref_count_factor) * 0.2)
    
    # Year diversity (less sensitive to reference count)

    year_diversity = (unique_years / total_refs) if years else 0
    
    # Combine with dynamic weights based on achievement
    diversity_score = (
        venue_diversity * 0.4 +
        author_diversity * 0.4 +
        year_diversity * 0.2
    )
    
    return max(0.0, min(np.log(diversity_score + 1) / np.log(2), 1.0))

def calculate_recency_score_dynamic(references, paper_year):
    """Calculate citation recency score with dynamic age assessment."""
    if not references:
        return 0.0
    
    if paper_year is None or paper_year == 0:
        paper_year = get_current_year()
    
    # Extract valid years
    years = [ref.get("year", 0) for ref in references if ref.get("year", 0) > 0]
    
    if not years:
        return 0.0
    
    # Calculate age distribution
    ages = [max(0, paper_year - year) for year in years]
    
    # Dynamic scoring based on age distribution
    age_scores = []
    for age in ages:
        if age <= 1:
            age_scores.append(1.0)
        elif age <= 3:
            age_scores.append(0.9 - (age - 1) * 0.1)
        elif age <= 7:
            age_scores.append(0.7 - (age - 3) * 0.05)
        elif age <= 15:
            age_scores.append(0.5 - (age - 7) * 0.02)
        else:
            age_scores.append(max(0.1, 0.34 - (age - 15) * 0.01))
    
    # Calculate weighted average with emphasis on recent citations
    total_score = sum(age_scores)
    avg_score = total_score / len(age_scores)
    
    # Bonus for having recent citations
    recent_citations = sum(1 for age in ages if age <= 3)
    recent_bonus = (recent_citations / len(ages)) * 0.1
    
    # Penalty for having too many old citations
    old_citations = sum(1 for age in ages if age > 10)
    old_penalty = (old_citations / len(ages)) * 0.15
    
    final_score = avg_score + recent_bonus - old_penalty
    
    return max(0.0, min(final_score, 1.0))

def calculate_self_citation_score_dynamic(references, paper_authors):
    """Calculate self-citation score with dynamic expectations."""
    if not references or not paper_authors:
        return 0.5
    
    # Extract author names
    paper_author_names = {author.lower().strip() for author in paper_authors}
    
    # Count self-citations with fuzzy matching
    self_citations = 0
    for ref in references:
        ref_authors = ref.get("author", [])
        ref_author_names = {author.lower().strip() for author in ref_authors}
        
        if paper_author_names.intersection(ref_author_names):
            self_citations += 1
    
    self_citation_percentage = self_citations / len(references) if references else 0
    
    # Dynamic scoring based on citation context
    total_refs = len(references)
    
    # Adjust expectations based on total reference count
    if total_refs < 15:
        # Smaller papers, more tolerance
        optimal_range = (0.0, 0.25)
        penalty_threshold = 0.35
    elif total_refs < 30:
        # Medium papers
        optimal_range = (0.02, 0.20)
        penalty_threshold = 0.30
    else:
        # Larger papers, stricter standards
        optimal_range = (0.03, 0.15)
        penalty_threshold = 0.25
    
    # Calculate score based on optimal range
    if optimal_range[0] <= self_citation_percentage <= optimal_range[1]:
        score = 1.0
    elif self_citation_percentage < optimal_range[0]:
        # Too few self-citations
        score = 0.8 + (self_citation_percentage / optimal_range[0]) * 0.2
    elif self_citation_percentage <= penalty_threshold:
        # Moderately high self-citations
        excess = self_citation_percentage - optimal_range[1]
        penalty_range = penalty_threshold - optimal_range[1]
        score = 1.0 - (excess / penalty_range) * 0.3
    else:
        # Too many self-citations
        excess = self_citation_percentage - penalty_threshold
        score = max(0.1, 0.7 - excess * 2)
    
    return max(0.0, min(score, 1.0))

def calculate_self_citation_percentage(references, paper_authors):
    """Calculate percentage of self-citations using smart matching (initials + last name)."""
    if not references or not paper_authors:
        return 0.0

    # Normalize paper authors into (initials, lastname) tuples
    paper_author_tuples = []
    for pa in paper_authors:
        initials = extract_initials(pa)
        lastname = extract_lastname(pa)
        paper_author_tuples.append((initials, lastname))

    self_citations = 0
    for ref in references:
        ref_authors = ref.get("author", [])
        matched = False
        for ref_author in ref_authors:
            ref_initials = extract_initials(ref_author)
            ref_lastname = extract_lastname(ref_author)

            for paper_initials, paper_lastname in paper_author_tuples:
                if ref_lastname == paper_lastname and ref_initials == paper_initials:
                    matched = True
                    break
            if matched:
                break

        if matched:
            self_citations += 1

    return (self_citations / len(references))

def get_score_rating(score):
    """Convert numeric score to descriptive rating with 50% threshold."""
    if score >= 50:
        return "Acceptable"
    else:
        return "Needs Improvement"

def calculate_literary_score_dynamic(paper_json, citation_gaps, claim_analysis, weights=None):
    """Calculate the overall literary score with dynamic component scoring."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    references = paper_json["metadata"].get("references", [])
    paper_year = paper_json["metadata"].get("year", get_current_year())
    paper_authors = paper_json["metadata"].get("authors", [])
    
    # Use dynamic scoring functions
    completeness_score = calculate_completeness_score_dynamic(references, citation_gaps)
    relevance_score = calculate_relevance_score_dynamic(references, claim_analysis)
    diversity_score = calculate_diversity_score_dynamic(references)
    recency_score = calculate_recency_score_dynamic(references, paper_year)
    self_citation_score = calculate_self_citation_score_dynamic(references, paper_authors)
    
    # Calculate weighted overall score
    overall_score = (
        weights["citation_completeness"] * completeness_score +
        weights["citation_relevance"] * relevance_score +
        weights["citation_diversity"] * diversity_score +
        weights["citation_recency"] * recency_score +
        weights["self_citation_ratio"] * self_citation_score
    )
    
    percentage_score = min(overall_score * 100, 100)
    
    detailed_report = {
        "overall_score": percentage_score,
        "component_scores": {
            "citation_completeness": completeness_score * 100,
            "citation_relevance": relevance_score * 100,
            "citation_diversity": diversity_score * 100,
            "citation_recency": recency_score * 100,
            "self_citation_ratio": self_citation_score * 100
        },
        "component_weights": weights,
        "metrics": {
            "total_references": len(references),
            "citation_gaps": len(citation_gaps),
            "citation_coverage_ratio": len(references) / (len(references) + len(citation_gaps)) if references or citation_gaps else 0,
            "average_citation_age": calculate_average_citation_age(references, paper_year),
            "self_citation_percentage": calculate_self_citation_percentage(references, paper_authors)
        },
        "rating": get_score_rating(percentage_score)
    }
    
    return detailed_report

def generate_literary_score_analysis_with_llm(score_report, paper_json, groq_api_key=None):
    """Generate a detailed textual analysis of the literary score using Groq LLM."""
    prompt = create_llm_prompt(score_report, paper_json)
    
    try:
        analysis = call_groq_api(prompt, groq_api_key)
        return parse_llm_response(analysis, score_report)
    except Exception as e:
        print(f"Error calling Groq API: {e}")

def create_score_visualization_data(score_report):
    """Create visualization data for score breakdown."""
    components = list(score_report["component_scores"].keys())
    scores = list(score_report["component_scores"].values())
    
    radar_data = [
        {
            "category": component.replace("_", " ").title(),
            "value": score,
            "fullMark": 100
        }
        for component, score in zip(components, scores)
    ]
    
    return radar_data

def generate_comprehensive_literary_report(paper_json, citation_gaps, claim_analysis, 
                                         weights=None, groq_api_key=None, 
                                         neo4j_credentials=None, semantic_scholar_api=None):
    """Generate a comprehensive literary score report with dynamic scoring."""
    # Calculate score using dynamic functions
    score_report = calculate_literary_score_dynamic(paper_json, citation_gaps, claim_analysis, weights)
    
    # Generate analysis using LLM if available
    analysis = generate_literary_score_analysis_with_llm(score_report, paper_json, groq_api_key)
    
    # Prepare visualization data
    visualization_data = create_score_visualization_data(score_report)
    
    # Combine into comprehensive report
    comprehensive_report = {
        "paper_title": paper_json["metadata"].get("title", "Unknown Paper"),
        "score_report": score_report,
        "analysis": analysis,
        "visualization_data": visualization_data,
        "citation_gaps": citation_gaps[:10],
        "timestamp": datetime.now().isoformat()
    }
    
    return comprehensive_report

# Keep the original function name for backward compatibility
def calculate_literary_score(paper_json, citation_gaps, claim_analysis, weights=None):
    """Backward compatibility wrapper for dynamic scoring."""
    return calculate_literary_score_dynamic(paper_json, citation_gaps, claim_analysis, weights)

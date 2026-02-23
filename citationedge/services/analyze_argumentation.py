from dotenv import load_dotenv
import os
from typing import *
import spacy
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from py2neo import Graph
import re
from groq import Groq
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from citationedge.utils.model_initializer import *
from citationedge.utils.text_processing import *
load_dotenv()

def argumentation_analysis(paper_json: Dict[str, Any], claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Complete workflow for enhanced argumentation analysis.
    
    Args:
        paper_json: JSON representation of the paper
        claims: List of extracted claims
        groq_api_key: API key for Groq
    
    Returns:
        Complete argumentation analysis results
    """
    models = initialize_models(os.getenv("GROQ_API_KEY"))
    
    analysis_results = analyze_argumentation(paper_json, claims, models)
    
    return analysis_results

def extract_paper_content(paper_json: Dict[str, Any]) -> str:
    """
    Extract the full text content of the paper.
    
    Args:
        paper_json: JSON representation of the paper
    
    Returns:
        Complete paper content as string
    """
    content = ""
    
    # Add abstract
    if "abstractText" in paper_json["metadata"]:
        abstract = paper_json["metadata"]["abstractText"]
        content += "Abstract\n" + abstract + "\n\n"
    
    for section in paper_json["metadata"]["sections"]:
        heading = section.get("heading")
        if heading is None:
            heading = "Abstract"
        else:
            heading_parts = heading.split(".")
            heading = heading_parts[-1].strip() if len(heading_parts) > 1 else heading
        
        text = section.get("text", "")
        
        if heading:  
            content += heading + "\n"
        if text:
            content += text + "\n\n"
                
    return content

def identify_premises_with_genai(section_content: str, claim_text: str, section_type: str, 
                                groq_client: Groq) -> List[Dict[str, Any]]:
    """
    Use GenAI to identify premises for a given claim with high precision.
    
    Args:
        section_content: Content of the section containing the claim
        claim_text: The claim text to find premises for
        section_type: Type of section (method, results, etc.)
        groq_client: Initialized Groq client
    
    Returns:
        List of identified premises with metadata
    """
    prompt = f"""
    TASK: Identify premises that support or relate to the given claim in academic text.
    
    SECTION TYPE: {section_type}
    
    CLAIM: "{claim_text}"
    
    SECTION CONTENT:
    {section_content[:2000]}
    
    INSTRUCTIONS:
    1. Find sentences that serve as premises (evidence, reasoning, methodology) for the claim
    2. For each premise, determine its relationship to the claim (support, contrast, neutral)
    3. Assess confidence level (0.0-1.0) based on logical connection strength
    4. Consider section-specific context:
       - Method sections: Algorithm descriptions, implementation details
       - Results sections: Data findings, experimental outcomes
       - Discussion sections: Interpretations, implications
    
    You must respond with ONLY valid JSON in exactly this format:
    {{"premises": [{{"text": "exact sentence text", "relation": "support", "confidence": 0.8, "reasoning": "brief explanation"}}]}}
    
    If no premises found, return: {{"premises": []}}
    
    Do not include any text before or after the JSON.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=1500
        )
        
        result = robust_json_parse(response.choices[0].message.content, "identify_premises_with_genai")
        
        if not result or "premises" not in result:
            return []
        
        premises = result.get("premises", [])
        
        # Validate each premise
        validated_premises = []
        for premise in premises:
            if (isinstance(premise, dict) and 
                all(key in premise for key in ["text", "relation", "confidence"]) and
                premise["relation"] in ["support", "contrast", "neutral"] and
                isinstance(premise["confidence"], (int, float)) and
                0.0 <= premise["confidence"] <= 1.0):
                validated_premises.append(premise)
            else:
                print(f"Invalid premise format: {premise}")
        
        return validated_premises
    
    except Exception as e:
        print(f"GenAI premise identification failed: {e}")
        return []

def extract_evidence_with_genai(section_content: str, claim_text: str, paper_json: Dict[str, Any], 
                               groq_client) -> List[Dict[str, Any]]:
    """
    Use GenAI to extract evidence (citations, data references) for claims.
    
    Args:
        section_content: Content of the section
        claim_text: The claim to find evidence for
        paper_json: Complete paper JSON
        groq_client: Initialized Groq client
    
    Returns:
        List of evidence items with metadata
    """
    # Extract references for context
    references_text = ""
    if "references" in paper_json.get("metadata", {}):
        refs = paper_json["metadata"]["references"][:10]  # First 10 refs
        references_text = "\n".join([
            f"[{i+1}] {ref.get('title', 'Unknown')} ({ref.get('year', 'N/A')})"
            for i, ref in enumerate(refs)
        ])
    
    prompt = f"""
    TASK: Extract evidence supporting the given claim from academic text.
    
    CLAIM: "{claim_text}"
    
    SECTION CONTENT:
    {section_content[:1500]}
    
    AVAILABLE REFERENCES:
    {references_text}
    
    INSTRUCTIONS:
    1. Find explicit citations [1], [2], (Author, Year) that support the claim
    2. Identify references to tables, figures, or data that serve as evidence
    3. Look for experimental results, statistical data, or empirical findings
    4. Assess relevance (0.0-1.0) of each evidence to the claim
    5. Distinguish between explicit citations and implicit references
    
    CRITICAL: Respond with ONLY ONE valid JSON object. Do not include multiple JSON objects.
    Use this EXACT format:
    {{"evidence": [{{"type": "citation", "text": "exact text", "relevance": 0.8, "context": "surrounding context", "reference_id": "1"}}]}}
    
    If no evidence found, return: {{"evidence": []}}
    
    Do not include any text before or after the JSON. Do not repeat the JSON structure multiple times.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200
        )
        
        result = robust_json_parse(response.choices[0].message.content, "extract_evidence_with_genai")
        
        if not result:
            return []
        
        evidence = result.get("evidence", [])
        
        # Validate evidence items
        validated_evidence = []
        for ev in evidence:
            if (isinstance(ev, dict) and 
                "text" in ev and "relevance" in ev and
                isinstance(ev.get("relevance"), (int, float)) and
                0.0 <= ev.get("relevance", 0) <= 1.0):
                
                # Ensure all required fields exist
                validated_ev = {
                    "type": ev.get("type", "citation"),
                    "text": str(ev["text"]),
                    "relevance": float(ev["relevance"]),
                    "context": ev.get("context", ""),
                    "reference_id": ev.get("reference_id", "")
                }
                validated_evidence.append(validated_ev)
            else:
                print(f"Invalid evidence format: {ev}")
        
        # Remove duplicates based on text content
        unique_evidence = []
        seen_texts = set()
        for ev in validated_evidence:
            if ev["text"] not in seen_texts:
                unique_evidence.append(ev)
                seen_texts.add(ev["text"])
        
        return unique_evidence
    
    except Exception as e:
        print(f"GenAI evidence extraction failed: {e}")
        return []
def compute_semantic_similarity(text1: str, text2: str, embedder: SentenceTransformer) -> float:
    """
    Compute semantic similarity between two texts using sentence transformers.
    
    Args:
        text1: First text
        text2: Second text  
        embedder: Sentence transformer model
    
    Returns:
        Similarity score (0.0-1.0)
    """
    try:
        embeddings = embedder.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception:
        return 0.5  # Fallback similarity

def classify_argument_components(text: str, claim: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use deep learning to classify if text is a premise, evidence, or neither.
    
    Args:
        text: Text to classify
        claim: Related claim text
        models: Dictionary of initialized models
    
    Returns:
        Classification results with confidence
    """
    # Use BART for natural language inference
    premise_hypothesis = f"This text provides support or reasoning for: {claim}"
    
    try:
        inputs = models['argument_tokenizer'](
            f"{text} </s> {premise_hypothesis}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = models['argument_classifier'](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
        # BART MNLI: 0=contradiction, 1=neutral, 2=entailment
        entailment_prob = probs[0][2].item()
        
        return {
            "is_premise": entailment_prob > 0.6,
            "confidence": entailment_prob,
            "classification": "premise" if entailment_prob > 0.6 else "neutral"
        }
    
    except Exception as e:
        return {
            "is_premise": False,
            "confidence": 0.5,
            "classification": "neutral"
        }

def evaluate_argument_strength_with_ai(premises: List[Dict], evidence: List[Dict], 
                                      section_type: str, groq_client: Groq) -> Dict[str, Any]:
    """
    Use AI to evaluate the overall strength of an argument.
    
    Args:
        premises: List of identified premises
        evidence: List of evidence items
        section_type: Type of section containing the argument
        groq_client: Groq client for AI evaluation
    
    Returns:
        Argument strength evaluation
    """
    premises_text = "\n".join([f"- {p.get('text', '')}" for p in premises[:5]])
    evidence_text = "\n".join([f"- {e.get('text', '')}" for e in evidence[:5]])
    
    prompt = f"""
    TASK: Evaluate the strength of an academic argument based on its premises and evidence.
    
    SECTION TYPE: {section_type}
    
    PREMISES:
    {premises_text}
    
    EVIDENCE:
    {evidence_text}
    
    EVALUATION CRITERIA:
    1. Logical coherence of premises
    2. Quality and quantity of evidence
    3. Relevance to section type:
       - Method: Technical rigor, implementation details
       - Results: Empirical support, statistical significance
       - Discussion: Logical reasoning, implications
    4. Overall persuasiveness
    
    INSTRUCTIONS:
    Rate the argument strength on a scale of 0.0-1.0 and provide reasoning.
    Consider section-specific standards and academic rigor.
    
    You must respond with ONLY valid JSON in exactly this format:
    {{"strength_score": 0.8, "quality_label": "good", "reasoning": "detailed explanation", "strengths": ["strength1"], "weaknesses": ["weakness1"]}}
    
    Quality labels must be: excellent, good, adequate, weak, or unsupported
    
    Do not include any text before or after the JSON.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        
        result = robust_json_parse(response.choices[0].message.content, "evaluate_argument_strength_with_ai")
        
        if not result:
            return {
                "strength_score": 0.5,
                "quality_label": "adequate",
                "reasoning": "Evaluation failed, using default score",
                "strengths": [],
                "weaknesses": ["Unable to evaluate automatically"]
            }
        
        # Validate required fields
        required_fields = ["strength_score", "quality_label", "reasoning"]
        for field in required_fields:
            if field not in result:
                result[field] = "Unknown" if field != "strength_score" else 0.5
        
        # Validate strength_score
        if not isinstance(result.get("strength_score"), (int, float)):
            result["strength_score"] = 0.5
        else:
            result["strength_score"] = max(0.0, min(1.0, result["strength_score"]))
        
        # Validate quality_label
        valid_labels = ["excellent", "good", "adequate", "weak", "unsupported"]
        if result.get("quality_label") not in valid_labels:
            result["quality_label"] = "adequate"
        
        # Ensure lists exist
        result["strengths"] = result.get("strengths", [])
        result["weaknesses"] = result.get("weaknesses", [])
        
        return result
    
    except Exception as e:
        print(f"AI argument evaluation failed: {e}")
        return {
            "strength_score": 0.5,
            "quality_label": "adequate",
            "reasoning": "Evaluation failed, using default score",
            "strengths": [],
            "weaknesses": ["Unable to evaluate automatically"]
        }


def create_argument_graph(claim: str, premises: List[Dict], evidence: List[Dict]) -> Dict[str, Any]:
    """
    Create a graph representation of the argument structure.
    
    Args:
        claim: The main claim
        premises: List of premises
        evidence: List of evidence
    
    Returns:
        Graph structure as dictionary
    """
    nodes = [{"id": "claim", "text": claim, "type": "claim"}]
    edges = []
    
    # Add premise nodes and edges
    for i, premise in enumerate(premises):
        node_id = f"premise_{i}"
        nodes.append({
            "id": node_id,
            "text": premise.get("text", ""),
            "type": "premise"
        })
        
        edges.append({
            "source": node_id,
            "target": "claim",
            "relation": premise.get("relation", "neutral"),
            "confidence": premise.get("confidence", 0.5)
        })
    
    # Add evidence nodes and edges
    for i, ev in enumerate(evidence):
        node_id = f"evidence_{i}"
        nodes.append({
            "id": node_id,
            "text": ev.get("text", ""),
            "type": "evidence"
        })
        
        edges.append({
            "source": node_id,
            "target": "claim",
            "relation": "support",
            "confidence": ev.get("relevance", 0.5)
        })
    
    return {"nodes": nodes, "edges": edges}

def generate_overall_assessment_with_ai(claim_analyses: List[Dict], groq_client: Groq) -> Dict[str, Any]:
    """
    Generate an overall assessment of the paper's argumentation quality using AI.
    
    Args:
        claim_analyses: List of individual claim analyses
        groq_client: Groq client for AI assessment
    
    Returns:
        Overall assessment dictionary
    """
    if not claim_analyses:
        return {
            "overall_strength": 0,
            "quality_label": "insufficient",
            "summary": "No claims were analyzed.",
            "key_strengths": [],
            "key_weaknesses": ["No claims found"],
            "recommendations": ["Add clear claims and supporting evidence."],
            "section_analysis": {"strongest_sections": [], "weakest_sections": []}
        }
    
    # Prepare summary for AI
    analyses_summary = []
    for i, analysis in enumerate(claim_analyses[:5]):  # Limit to avoid token limits
        analyses_summary.append({
            "claim": analysis.get("claim", "")[:100],
            "quality": analysis.get("argument_quality", "unknown"),
            "section": analysis.get("section", "unknown"),
            "premises_count": len(analysis.get("premises", [])),
            "evidence_count": len(analysis.get("evidence", []))
        })
    
    prompt = f"""
    TASK: Provide an overall assessment of a research paper's argumentation quality.
    
    CLAIM ANALYSES SUMMARY:
    {json.dumps(analyses_summary, indent=2)}
    
    TOTAL CLAIMS ANALYZED: {len(claim_analyses)}
    
    INSTRUCTIONS:
    1. Evaluate the overall argumentation strength across all claims
    2. Identify patterns in argument quality by section
    3. Assess the paper's scholarly rigor and evidence support
    4. Provide specific, actionable recommendations for improvement
    5. Consider academic standards for research papers
    
    You must respond with ONLY valid JSON in exactly this format:
    {{"overall_strength": 0.8, "quality_label": "good", "summary": "comprehensive summary", "key_strengths": ["strength1"], "key_weaknesses": ["weakness1"], "recommendations": ["rec1"], "section_analysis": {{"strongest_sections": ["section1"], "weakest_sections": ["section2"]}}}}
    
    Quality labels must be: excellent, good, adequate, needs_improvement, or poor
    
    Do not include any text before or after the JSON.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        
        result = robust_json_parse(response.choices[0].message.content, "generate_overall_assessment_with_ai")
        
        if not result:
            # Fallback to simple calculation
            strengths = [analysis.get("argument_strength", 0) 
                        for analysis in claim_analyses if "argument_strength" in analysis]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0
            
            return {
                "overall_strength": avg_strength,
                "quality_label": "adequate" if avg_strength > 0.5 else "needs_improvement",
                "summary": f"Analyzed {len(claim_analyses)} claims with average strength {avg_strength:.2f}",
                "key_strengths": ["Automated analysis completed"],
                "key_weaknesses": ["Manual review recommended"],
                "recommendations": ["Strengthen evidence support", "Improve logical reasoning"],
                "section_analysis": {"strongest_sections": [], "weakest_sections": []}
            }
        
        # Validate and set defaults for required fields
        result["overall_strength"] = result.get("overall_strength", 0.5)
        if not isinstance(result["overall_strength"], (int, float)):
            result["overall_strength"] = 0.5
        else:
            result["overall_strength"] = max(0.0, min(1.0, result["overall_strength"]))
        
        valid_labels = ["excellent", "good", "adequate", "needs_improvement", "poor"]
        if result.get("quality_label") not in valid_labels:
            result["quality_label"] = "adequate"
        
        # Ensure all required fields exist
        required_fields = {
            "summary": "Analysis completed successfully",
            "key_strengths": [],
            "key_weaknesses": [],
            "recommendations": [],
            "section_analysis": {"strongest_sections": [], "weakest_sections": []}
        }
        
        for field, default_value in required_fields.items():
            if field not in result:
                result[field] = default_value
        
        return result
    
    except Exception as e:
        print(f"AI overall assessment failed: {e}")
        # Fallback to simple calculation
        strengths = [analysis.get("argument_strength", 0) 
                    for analysis in claim_analyses if "argument_strength" in analysis]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0
        
        return {
            "overall_strength": avg_strength,
            "quality_label": "adequate" if avg_strength > 0.5 else "needs_improvement",
            "summary": f"Analyzed {len(claim_analyses)} claims with average strength {avg_strength:.2f}",
            "key_strengths": ["Automated analysis completed"],
            "key_weaknesses": ["Manual review recommended"],
            "recommendations": ["Strengthen evidence support", "Improve logical reasoning"],
            "section_analysis": {"strongest_sections": [], "weakest_sections": []}
        }

def save_argument_graph(claim_analysis: Dict[str, Any], claim_index: int, output_folder: str = "argument_graphs"):
    """
    Save argument graph as image file.
    
    Args:
        claim_analysis: Single claim analysis dict
        claim_index: Index number of the claim
        output_folder: Folder to save graphs
    """
    # Create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    graph_dict = claim_analysis["argument_graph"]
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in graph_dict["nodes"]:
        G.add_node(node["id"], text=node["text"], type=node["type"])
    
    for edge in graph_dict["edges"]:
        G.add_edge(edge["source"], edge["target"])
    
    # Layout and colors
    pos = nx.spring_layout(G, seed=42)
    colors = {"claim": "red", "premise": "blue", "evidence": "green"}
    node_colors = [colors[G.nodes[node]["type"]] for node in G.nodes()]
    
    # Plot
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=node_colors, with_labels=True, 
            node_size=800, arrowsize=20, font_size=8)
    
    # Title with claim info
    claim_text = claim_analysis["claim"][:50] + "..." if len(claim_analysis["claim"]) > 50 else claim_analysis["claim"]
    plt.title(f"Claim {claim_index + 1}: {claim_text}")
    
    # Save
    filename = f"{output_folder}/argument_graph_claim_{claim_index + 1}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() 


def analyze_argumentation(paper_json: Dict[str, Any], extracted_claims: List[Dict], 
                         models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to analyze the argumentation structure of a paper using GenAI and deep learning.
    
    Args:
        paper_json: JSON representation of the paper
        extracted_claims: List of claims extracted from the paper
        models: Dictionary of initialized models and clients
    
    Returns:
        Complete argumentation analysis
    """
    argumentation_analysis = []
    
    for i, claim in enumerate(extracted_claims):  # Added enumerate to get index
        try:
            claim_text = claim["text"]
            section_name = claim["section"]
            
            # Get section content and type
            section_content = get_section_content(paper_json, section_name)
            section_type = determine_section_type(section_name)
            
            # Use GenAI to identify premises
            premises = identify_premises_with_genai(
                section_content, claim_text, section_type, models['groq_client']
            )
            
            # Use GenAI to extract evidence
            evidence = extract_evidence_with_genai(
                section_content, claim_text, paper_json, models['groq_client']
            )
            
            # Enhanced premise validation using deep learning
            validated_premises = []
            for premise in premises:
                validation = classify_argument_components(
                    premise.get("text", ""), claim_text, models
                )
                
                if validation["is_premise"]:
                    premise["dl_confidence"] = validation["confidence"]
                    validated_premises.append(premise)
            
            # Create argument graph
            argument_graph = create_argument_graph(claim_text, validated_premises, evidence)
            
            # Evaluate argument strength using AI
            strength_evaluation = evaluate_argument_strength_with_ai(
                validated_premises, evidence, section_type, models['groq_client']
            )
            
            # Build comprehensive analysis
            claim_analysis = {
                "claim": claim_text,
                "claim_type": claim.get("claim_type", "unknown"),
                "section": section_name,
                "section_type": section_type,
                "premises": validated_premises,
                "evidence": evidence,
                "argument_graph": argument_graph,
                "argument_strength": strength_evaluation.get("strength_score", 0.5),
                "argument_quality": strength_evaluation.get("quality_label", "adequate"),
                "ai_evaluation": strength_evaluation
            }
            
            argumentation_analysis.append(claim_analysis)
            save_argument_graph(claim_analysis, i, "arg_graphs")  # Now using proper index
            
        except Exception as e:
            print(f"Error analyzing claim '{claim.get('text', 'Unknown')}': {e}")
            continue
    
    # Generate overall assessment using AI
    overall_assessment = generate_overall_assessment_with_ai(
        argumentation_analysis, models['groq_client']
    )
    
    return {
        "claim_arguments": argumentation_analysis,
        "overall_assessment": overall_assessment,
        "analysis_metadata": {
            "total_claims": len(extracted_claims),
            "successfully_analyzed": len(argumentation_analysis),
            "models_used": ["groq_mixtral", "sentence_transformers", "bart_mnli"],
            "analysis_type": "genai_enhanced"
        }
    }

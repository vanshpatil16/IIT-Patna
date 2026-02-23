from dotenv import load_dotenv
from citationedge.models.rag_graph import rag_context
from typing import *
import networkx as nx
import os
import json
from citationedge.services.rag_service import *
from citationedge.services.knowledge_graph_services import *
load_dotenv()

def extract_claims_with_rag(text: str, context_data: List[Dict], section: str = None) -> List[Dict]:
    if not text or not text.strip():
        return []
    
    # FIXED: Handle None values in context data
    context_str = ""
    for ctx in context_data[:3]:  # Use top 3 relevant contexts
        heading = ctx.get('heading', '') or ''
        concepts = [c for c in ctx.get('concepts', []) if c and c.strip()][:5]
        entities = [e for e in ctx.get('entities', []) if e and e.strip()][:3]
        
        context_str += f"Section: {heading}\n"
        context_str += f"Key concepts: {', '.join(concepts)}\n"
        context_str += f"Entities: {', '.join(entities)}\n\n"
    
    prompt = f"""
    You are a research claim extraction expert. Extract research claims from the text and score them accurately.

    CONTEXT FROM DOCUMENT:
    {context_str}

    TARGET TEXT TO ANALYZE:
    "{text}"

    SECTION: {section or 'Unknown'}

    Extract research claims and classify them into these MUTUALLY EXCLUSIVE categories:

    CLAIM TYPES & STRICT SCORING RANGES:
    1. "breakthrough" - Major novel discovery/method that hasn't been done before
    - Confidence: 0.3-0.6 (lower because breakthrough claims are harder to verify)
    - Novelty: 0.8-1.0 (very high novelty by definition)

    2. "significant" - Important finding/improvement over existing work
    - Confidence: 0.5-0.8 (moderate to high confidence)
    - Novelty: 0.6-0.9 (high novelty)

    3. "incremental" - Minor improvement/variation of existing methods
    - Confidence: 0.7-0.95 (high confidence for incremental work)
    - Novelty: 0.3-0.7 (moderate novelty)

    4. "supportive" - Supporting evidence, validation, or implementation details
    - Confidence: 0.8-0.95 (high confidence for established facts)
    - Novelty: 0.1-0.5 (low novelty - mostly confirming existing knowledge)

    5. "methodological" - Process/technique description without major novelty claims
    - Confidence: 0.6-0.9 (varies based on how well-established the method is)
    - Novelty: 0.2-0.6 (low to moderate novelty)

    CONFIDENCE SCORING (how certain/well-supported the claim is):
    - 0.9-1.0: Definitive statements with strong empirical evidence
    - 0.7-0.9: Well-supported claims with good evidence
    - 0.5-0.7: Moderate claims with some uncertainty
    - 0.3-0.5: Tentative claims or preliminary findings
    - 0.1-0.3: Speculative or highly uncertain claims

    NOVELTY SCORING (how new/original the contribution is):
    - 0.9-1.0: Completely new approach, never done before
    - 0.7-0.9: Significant novel contribution or major improvement
    - 0.5-0.7: Moderate novelty, some new elements
    - 0.3-0.5: Minor novelty, small improvements
    - 0.1-0.3: Low novelty, mostly established knowledge

    CONTEXT RELEVANCE (how central the claim is to the section):
    - 0.9-1.0: Central to the section's main argument
    - 0.7-0.9: Important supporting point
    - 0.5-0.7: Relevant but secondary
    - 0.3-0.5: Tangentially related
    - 0.1-0.3: Barely connected to context

    CRITICAL: Your novelty score MUST align with your claim type classification. 
    - If claim_type is "supportive", novelty MUST be 0.1-0.5
    - If claim_type is "breakthrough", novelty MUST be 0.8-1.0
    - Follow the ranges strictly for each type

    Return a JSON array with this exact structure:
    [
        {{
            "text": "exact sentence containing the claim",
            "claim_type": "breakthrough|significant|incremental|supportive|methodological",
            "confidence": 0.XX,
            "novelty": 0.XX,
            "context_relevance": 0.XX,
            "key_terms": ["term1", "term2"],
            "reasoning": "Brief explanation of claim type classification and why these specific scores"
        }}
    ]

    RETURN ONLY THE JSON ARRAY. NO MARKDOWN, NO EXPLANATIONS, NO CODE BLOCKS.
    """
    
    try:
        response = rag_context.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        claims_data = json.loads(response.choices[0].message.content)
        
        # Add section information and ensure no None values
        for claim in claims_data:
            claim['section'] = section or 'Unknown'
            claim['extraction_method'] = 'rag'
            # Ensure key_terms is not None
            if not claim.get('key_terms'):
                claim['key_terms'] = []
            # Filter out None values from key_terms
            claim['key_terms'] = [term for term in claim.get('key_terms', []) if term and term.strip()]
            
        return claims_data
        
    except Exception as e:
        #print(f"Error extracting claims with RAG: {e}")
        # Fallback to simpler extraction
        return extract_claims_fallback(text, section)

def extract_claims_fallback(text: str, section: str = None) -> List[Dict]:
    """Fallback claim extraction using pattern matching and NLP."""
    if not text or not text.strip():
        return []
    
    doc = rag_context.nlp(text)
    claims = []
    
    # Look for sentences with claim indicators
    claim_indicators = [
        "we propose", "we present", "we introduce", "we develop", "we show",
        "results show", "demonstrates", "we found", "our method", "our approach",
        "this paper", "this work", "we conclude", "we demonstrate", "we establish",
        "novel", "new", "improved", "better", "outperforms", "achieves", "enables"
    ]
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text.split()) < 5:  # Too short
            continue
            
        # Check for claim indicators
        has_indicator = any(indicator in sent_text.lower() for indicator in claim_indicators)
        
        # Check for technical terms (entities, noun phrases)
        has_technical_terms = len([ent for ent in sent.ents]) > 0 or len(list(sent.noun_chunks)) > 2
        
        if has_indicator or has_technical_terms:
            # FIXED: Handle None values in entities
            key_terms = []
            for ent in sent.ents:
                if ent.text and ent.text.strip():
                    key_terms.append(ent.text.strip())
            
            claims.append({
                "text": sent_text,
                "claim_type": "general",
                "confidence": 0.7 if has_indicator else 0.5,
                "key_terms": key_terms,
                "context_relevance": 0.6,
                "section": section or 'Unknown',
                "extraction_method": "fallback"
            })
    
    return claims[:5]  # Return top 5

def augment_claims_with_graph_knowledge(claims: List[Dict], doc_graph: nx.DiGraph, 
                                       external_knowledge: List[Dict] = None) -> List[Dict]:
    """Augment claims with knowledge from graph and external sources."""
    augmented_claims = []
    
    for claim in claims:
        # Fix: Access claim text properly from dictionary
        claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
        if not claim_text or not claim_text.strip():
            continue
        
        try:
            related_contexts = retrieve_relevant_context(claim_text, doc_graph, k=3)
            
            similar_claims = query_similar_claims_neo4j(claim_text)
            
            novelty_score = calculate_graph_based_novelty(claim, related_contexts, similar_claims)
            
            category_analysis = categorize_claim_with_rag(claim, related_contexts, similar_claims)
            
            # Augment claim with all information
            if isinstance(claim, dict):
                augmented_claim = claim.copy()
            else:
                augmented_claim = {"text": str(claim)}
                
            augmented_claim.update({
                "novelty_score": novelty_score,
                "category": category_analysis.get("category", "unknown") if isinstance(category_analysis, dict) else "unknown",
                "evidence": category_analysis.get("evidence", "No evidence provided") if isinstance(category_analysis, dict) else "No evidence provided",
                "confidence_score": category_analysis.get("confidence", 0.5) if isinstance(category_analysis, dict) else 0.5,
                "related_contexts": related_contexts,
                "similar_claims": similar_claims[:3] if similar_claims else [],
                "graph_connections": len(related_contexts)
            })
            
            augmented_claims.append(augmented_claim)
            
        except Exception as e:
            print(f"Error augmenting claim: {e}")
            # Add the claim with minimal augmentation
            if isinstance(claim, dict):
                augmented_claim = claim.copy()
            else:
                augmented_claim = {"text": str(claim)}
                
            augmented_claim.update({
                "novelty_score": 0.5,
                "category": "unknown",
                "evidence": "Error during augmentation",
                "confidence_score": 0.5,
                "related_contexts": [],
                "similar_claims": [],
                "graph_connections": 0
            })
            augmented_claims.append(augmented_claim)
    
    return augmented_claims
    
def extract_claims_from_paper_rag(paper_json: Dict, groq_api_key: str) -> List[Dict]:
    """Enhanced RAG-based claim extraction with literature validation."""
    EXCLUDE_KEYWORDS = [
        "conclusion", "conclusions", "related work", "acknowledgement", 
        "acknowledgements", "acknowledgments", "abstract"
    ]
    
    if not initialize_rag_system(groq_api_key):
        return []
    
    print("🔄 Building document knowledge graph...")
    doc_graph = build_document_knowledge_graph(paper_json)
    
    # Extract text sections
    sections_data = []
    abstract = paper_json.get("metadata", {}).get("abstractText", "")
    if abstract and abstract.strip():
        sections_data.append({"text": abstract, "section": "Abstract", "priority": "high"})
    
    for i, section in enumerate(paper_json.get("metadata", {}).get("sections", [])):
        section_text = section.get("text", "")
        section_heading = section.get("heading", f"Section_{i}")
        
        if section_text and len(section_text.strip()) > 100:
            sections_data.append({
                "text": section_text,
                "section": section_heading,
                "priority": "medium"
            })
    
    all_claims = []
    for section_data in sections_data:
        if section_data['section'] is not None:
            raw_title = (section_data.get("section") or "")
            title = re.sub(r'^\d+\.?\s*', '', raw_title).strip().lower()
            if any(keyword in title for keyword in EXCLUDE_KEYWORDS):
                continue
        
        relevant_contexts = retrieve_relevant_context(section_data["text"], doc_graph)
        section_claims = extract_claims_with_rag(
            section_data["text"], 
            relevant_contexts, 
            section_data["section"]
        )
        
        print(f"📝 Found {len(section_claims)} claims in {section_data['section']}")
        all_claims.extend(section_claims)
    
    if not all_claims:
        for section_data in sections_data[:2]:
            fallback_claims = extract_claims_fallback(section_data["text"], section_data["section"])
            all_claims.extend(fallback_claims)
    
    if not all_claims:
        return []
    
    # Batch query Semantic Scholar for literature validation
    print("🔍 Querying Semantic Scholar for literature validation...")
    ss_results = query_semantic_scholar_batch(all_claims)
    
    # Enhance claims with literature analysis
    enhanced_claims = []
    for claim in all_claims:
        claim_text = claim.get("text", "")
        
        # Get literature results for this claim
        literature_results = ss_results.get(claim_text, [])
        
        # Calculate literature prevalence
        prevalence_score = calculate_literature_prevalence(claim_text, literature_results)
        
        # Expand context with literature
        original_contexts = retrieve_relevant_context(claim_text, doc_graph)
        expanded_contexts = expand_context_with_literature(
            original_contexts, claim_text, literature_results
        )
        
        # Recalculate novelty with enhanced context
        similar_claims = query_similar_claims_neo4j(claim_text)
        enhanced_novelty = calculate_graph_based_novelty(claim, expanded_contexts, similar_claims)
        
        # Calibrate confidence with literature
        calibrated_confidence = calibrate_confidence_with_literature(
            claim, literature_results, prevalence_score
        )
        
        # Update claim with enhanced scores
        claim.update({
            "novelty_score": enhanced_novelty,
            "confidence_score": calibrated_confidence,
            "literature_prevalence": prevalence_score,
            "literature_papers_count": len(literature_results),
            "enhanced_context_count": len(expanded_contexts)
        })
        
        enhanced_claims.append(claim)
    
    # Augment with graph knowledge
    if enhanced_claims:
        augmented_claims = augment_claims_with_graph_knowledge(enhanced_claims, doc_graph)
        
        # Sort by adjusted novelty score
        augmented_claims.sort(
            key=lambda x: (x.get("novelty_score", 0) * x.get("confidence_score", 0)), 
            reverse=True
        )
        
        return augmented_claims
    
    return []

def main_rag_pipeline(paper_json_path: str):
    """Complete RAG-based pipeline for claim extraction."""
    try:
        print("🚀 Starting RAG-based Claim Extraction Pipeline...")
        groq_api_key=os.getenv("GROQ_API_KEY")
        # Load paper
        with open(paper_json_path, 'r', encoding="utf-8") as f:
            paper_json = json.load(f)
        
        title = paper_json.get('metadata', {}).get('title', 'Unknown Title')
        
        # Extract claims using RAG
        claims = extract_claims_from_paper_rag(paper_json, groq_api_key)
        
        return claims
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline: {e}")
        return []

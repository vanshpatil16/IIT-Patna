from citationedge.models.rag_graph import rag_context
from citationedge.services.knowledge_graph_services import *
from neo4j import GraphDatabase
from groq import Groq
import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
import os
from typing import *
from dotenv import load_dotenv
import neo4j
from urllib.parse import urlparse
import json
import re
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import statistics
from citationedge.utils.cache import *

load_dotenv()

def initialize_rag_system(groq_api_key: str, neo4j_credentials: Dict = None) -> bool:
    """Initialize RAG system with all components."""
    try:
        rag_context.groq_client = Groq(api_key=groq_api_key)
        rag_context.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        rag_context.nlp = spacy.load("en_core_web_lg")
        
        rag_context.graph = GraphDatabase.driver(
                os.getenv("NEO4J_URI"), 
                auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )

        rag_context.knowledge_graph = nx.DiGraph()
        
        print("✅ RAG system initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

def calculate_graph_based_novelty(claim: Dict, contexts: List[Dict], similar_claims: List[Dict]) -> float:
    """Enhanced novelty calculation with adaptive thresholds."""
    base_score = 0.4
    
    # Collect all similarities for adaptive thresholds
    all_similarities = []
    
    # Add similarities from similar claims
    for sc in similar_claims:
        sim = sc.get("similarity")
        if sim is not None:
            all_similarities.append(sim)
    
    # Add similarities from contexts (literature)
    for ctx in contexts:
        sim = ctx.get("similarity")
        if sim is not None:
            all_similarities.append(sim)
    
    # Calculate adaptive thresholds
    thresholds = adaptive_similarity_thresholds(all_similarities)
    
    # Factor 1: Context diversity
    if contexts:
        unique_concepts = set()
        literature_count = 0
        
        for ctx in contexts:
            if ctx.get("source") == "semantic_scholar":
                literature_count += 1
            
            concepts = ctx.get("concepts", [])
            if concepts:
                valid_concepts = [c for c in concepts if c and c.strip()]
                unique_concepts.update(valid_concepts)
        
        # Diversity bonus
        concept_count = len(unique_concepts)
        if concept_count > 8:
            base_score += 0.2
        elif concept_count > 5:
            base_score += 0.1
        elif concept_count > 2:
            base_score += 0.05
        
        # Literature context penalty (more literature = less novel)
        if literature_count > 0:
            literature_penalty = min(0.3, literature_count * 0.1)
            base_score -= literature_penalty
    
    # Factor 2: Similarity penalties with adaptive thresholds
    if all_similarities:
        max_similarity = max(all_similarities)
        avg_similarity = statistics.mean(all_similarities)
        
        # Adaptive penalty based on thresholds
        if max_similarity > thresholds["high"]:
            base_score -= 0.4
        elif max_similarity > thresholds["medium"]:
            base_score -= 0.25
        elif max_similarity > thresholds["low"]:
            base_score -= 0.15
        
        # Additional penalty for high average similarity
        if avg_similarity > 0.6:
            base_score -= 0.1
    else:
        # No similarities found - moderate novelty bonus
        base_score += 0.1
    
    # Factor 3: Technical complexity
    key_terms = claim.get("key_terms", [])
    valid_terms = [t for t in key_terms if t and t.strip()]
    
    if len(valid_terms) > 6:
        base_score += 0.12
    elif len(valid_terms) > 3:
        base_score += 0.06
    
    # Factor 4: Claim strength indicators
    claim_text = claim.get("text", "").lower()
    
    # Novelty indicators
    novelty_indicators = 0
    innovation_terms = ["novel", "new", "first", "unprecedented", "innovative", "original"]
    for term in innovation_terms:
        if term in claim_text:
            novelty_indicators += 1
    
    if novelty_indicators > 2:
        base_score += 0.15
    elif novelty_indicators > 0:
        base_score += 0.08
    
    # Improvement indicators
    improvement_terms = ["improved", "enhanced", "better", "optimized", "efficient"]
    improvement_count = sum(1 for term in improvement_terms if term in claim_text)
    
    if improvement_count > 0:
        base_score += min(0.1, improvement_count * 0.03)
    
    # Add controlled randomness
    base_score += np.random.uniform(-0.03, 0.03)
    
    return max(0.1, min(0.9, round(base_score, 3)))

def categorize_claim_with_rag(claim: Dict, contexts: List[Dict], similar_claims: List[Dict]) -> Dict:
    """Categorize claim using RAG approach with robust error handling."""
    claim_text = claim.get("text", "")
    if not claim_text or not claim_text.strip():
        return {"category": "unknown", "confidence": 0.1, "evidence": "Empty claim text", "key_novelty_aspects": []}
    
    novelty_score = claim.get("novelty_score", 0.5)
    
    # Prepare context for categorization
    context_info = ""
    if contexts:
        headings = [ctx.get('heading', '') for ctx in contexts[:2] if ctx.get('heading')]
        context_info = f"Related sections: {', '.join(headings)}\n"
        
        all_concepts = []
        for ctx in contexts:
            concepts = ctx.get('concepts', [])
            if concepts:
                valid_concepts = [c for c in concepts[:3] if c and c.strip()]
                all_concepts.extend(valid_concepts)
        
        unique_concepts = list(set(all_concepts))
        context_info += f"Key concepts: {', '.join(unique_concepts)}\n"
    
    similar_info = ""
    if similar_claims:
        valid_similarities = [sc.get('similarity', 0) for sc in similar_claims if sc.get('similarity') is not None]
        if valid_similarities:
            max_sim = max(valid_similarities)
            similar_info = f"Found {len(similar_claims)} similar claims with max similarity: {max_sim:.2f}\n"
            
            first_similar = similar_claims[0]
            first_text = first_similar.get('text', '')
            if first_text and first_text.strip():
                similar_info += f"Most similar: {first_text[:100]}...\n"
    
    # Clean and escape the claim text more thoroughly
    escaped_claim_text = clean_text_for_json(claim_text)
    
    prompt = f"""Analyze and categorize this research claim based on its novelty and relationship to existing work.

        CLAIM TO ANALYZE:
        "{escaped_claim_text}"

        DOCUMENT CONTEXT:
        {context_info}

        SIMILAR EXISTING CLAIMS:
        {similar_info if similar_info else "None identified"}

        COMPUTED NOVELTY SCORE: {novelty_score:.2f}

        CATEGORIZATION OPTIONS:
        - novel: Genuinely new contribution, method, or finding not seen before
        - incremental: Meaningful improvement, extension, or refinement of existing work
        - non-novel: Very similar or nearly identical to existing claims
        - supportive: Provides supporting evidence, validation, or replication of known results

        RESPONSE REQUIREMENTS:
        Return ONLY a valid JSON object with no additional text, explanations, or formatting.

        Required JSON structure:
        {{"category": "novel", "confidence": 0.85, "evidence": "brief justification without quotes", "key_novelty_aspects": ["aspect1", "aspect2"]}}

        VALIDATION RULES:
        - category: must be exactly one of "novel", "incremental", "non-novel", "supportive"
        - confidence: float between 0.0 and 1.0 representing classification certainty
        - evidence: concise reasoning (under 80 words) without direct quotes
        - key_novelty_aspects: array of 1-5 strings describing what makes this claim distinct"""
    
    try:
        response = rag_context.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Zero temperature for maximum consistency
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON with multiple fallback strategies
        result = parse_json_response(response_text)
        
        # Validate and fix the result
        result = validate_and_fix_result(result, novelty_score)
        return result
        
    except Exception as e:
        print(f"Error categorizing claim: {e}")
        # Fallback categorization based on novelty score
        return get_fallback_categorization(novelty_score)

def clean_text_for_json(text: str) -> str:
    """Clean text to prevent JSON parsing issues."""
    if not text:
        return ""
    
    # Replace problematic characters
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.replace('\r', ' ')  # Replace carriage returns with spaces
    text = text.replace('\t', ' ')  # Replace tabs with spaces
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = text.strip()
    
    # Limit length to prevent overly long strings
    if len(text) > 200:
        text = text[:200] + "..."
    
    return text

def parse_json_response(response_text: str) -> Dict:
    """Parse JSON response with multiple fallback strategies."""
    
    # Strategy 1: Try parsing as-is
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to fix common JSON issues
    try:
        cleaned_json = fix_json_formatting(response_text)
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract values using regex
    return extract_values_with_regex(response_text)

def fix_json_formatting(json_text: str) -> str:
    """Fix common JSON formatting issues."""
    # Remove markdown formatting
    json_text = re.sub(r'```json\s*', '', json_text)
    json_text = re.sub(r'```\s*', '', json_text)
    
    # Extract just the JSON object
    json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if json_match:
        json_text = json_match.group()
    
    # Remove trailing commas
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    # Fix unescaped quotes in string values
    # This is a more robust approach
    def fix_quotes_in_value(match):
        key = match.group(1)
        value = match.group(2)
        
        # If value contains unescaped quotes, escape them
        if '"' in value:
            # Don't escape if already escaped
            value = re.sub(r'(?<!\\)"', '\\"', value)
        
        return f'"{key}": "{value}"'
    
    # Match key-value pairs with string values
    json_text = re.sub(r'"([^"]+)":\s*"([^"]*(?:\\"[^"]*)*)"', fix_quotes_in_value, json_text)
    
    return json_text

def extract_values_with_regex(response_text: str) -> Dict:
    """Extract values using regex as a last resort."""
    result = {
        "category": "incremental",
        "confidence": 0.5,
        "evidence": "Could not parse LLM response",
        "key_novelty_aspects": []
    }
    
    # Extract category
    category_match = re.search(r'"category":\s*"([^"]+)"', response_text)
    if category_match:
        category = category_match.group(1).lower()
        if category in ['novel', 'incremental', 'non-novel', 'supportive']:
            result["category"] = category
    
    # Extract confidence
    confidence_match = re.search(r'"confidence":\s*([0-9.]+)', response_text)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            result["confidence"] = max(0.0, min(1.0, confidence))
        except ValueError:
            pass
    
    # Extract evidence
    evidence_match = re.search(r'"evidence":\s*"([^"]+)"', response_text)
    if evidence_match:
        result["evidence"] = evidence_match.group(1)
    
    # Extract key novelty aspects
    aspects_match = re.search(r'"key_novelty_aspects":\s*\[(.*?)\]', response_text, re.DOTALL)
    if aspects_match:
        aspects_str = aspects_match.group(1)
        aspects = re.findall(r'"([^"]+)"', aspects_str)
        result["key_novelty_aspects"] = aspects
    
    return result

def validate_and_fix_result(result: Dict, novelty_score: float) -> Dict:
    """Validate and fix the categorization result."""
    valid_categories = ['novel', 'incremental', 'non-novel', 'supportive']
    
    # Ensure required keys exist
    if 'category' not in result:
        result['category'] = 'incremental'
    if 'confidence' not in result:
        result['confidence'] = 0.5
    if 'evidence' not in result:
        result['evidence'] = 'Automatic categorization'
    if 'key_novelty_aspects' not in result:
        result['key_novelty_aspects'] = []
    
    # Validate category
    if result['category'] not in valid_categories:
        # Try to map similar categories
        category_lower = str(result['category']).lower()
        if 'novel' in category_lower and 'non' not in category_lower:
            result['category'] = 'novel'
        elif 'increment' in category_lower:
            result['category'] = 'incremental'
        elif 'support' in category_lower:
            result['category'] = 'supportive'
        else:
            result['category'] = 'incremental'
    
    # Validate confidence
    try:
        confidence = float(result['confidence'])
        result['confidence'] = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        result['confidence'] = 0.5
    
    # Ensure evidence is a string and clean it
    if not isinstance(result['evidence'], str):
        result['evidence'] = str(result['evidence'])
    
    # Clean evidence text
    result['evidence'] = clean_text_for_json(result['evidence'])
    
    # Ensure key_novelty_aspects is a list
    if not isinstance(result['key_novelty_aspects'], list):
        if isinstance(result['key_novelty_aspects'], str):
            # Try to parse as a simple list
            aspects = [aspect.strip() for aspect in result['key_novelty_aspects'].split(',')]
            result['key_novelty_aspects'] = aspects
        else:
            result['key_novelty_aspects'] = []
    
    # Clean novelty aspects
    result['key_novelty_aspects'] = [
        clean_text_for_json(aspect) for aspect in result['key_novelty_aspects']
        if aspect and str(aspect).strip()
    ]
    
    return result

def get_fallback_categorization(novelty_score: float) -> Dict:
    """Get fallback categorization based on novelty score."""
    if novelty_score > 0.7:
        return {
            "category": "novel", 
            "confidence": 0.6, 
            "evidence": f"High novelty score ({novelty_score:.2f}) suggests novel contribution", 
            "key_novelty_aspects": []
        }
    elif novelty_score > 0.4:
        return {
            "category": "incremental", 
            "confidence": 0.6, 
            "evidence": f"Moderate novelty score ({novelty_score:.2f}) suggests incremental improvement", 
            "key_novelty_aspects": []
        }
    else:
        return {
            "category": "non-novel", 
            "confidence": 0.6, 
            "evidence": f"Low novelty score ({novelty_score:.2f}) suggests limited novelty", 
            "key_novelty_aspects": []
        }

def query_semantic_scholar_batch(claims: List[Dict], batch_size: int = 5) -> Dict[str, List[Dict]]:
    """Batch query Semantic Scholar API for similar papers/claims."""
    cache = load_cache()
    results = {}
    uncached_claims = []
    
    # Check cache first
    for claim in claims:
        claim_text = claim.get("text", "")
        if not claim_text.strip():
            continue
            
        cache_key = generate_cache_key(claim_text)
        if cache_key in cache:
            results[claim_text] = cache[cache_key][0]
        else:
            uncached_claims.append(claim)
    
    if not uncached_claims:
        return results
    
    # Process uncached claims in batches
    for i in range(0, len(uncached_claims), batch_size):
        batch = uncached_claims[i:i + batch_size]
        
        for claim in batch:
            claim_text = claim.get("text", "")
            if not claim_text.strip():
                continue
                
            try:
                # Extract key terms for search
                key_terms = extract_search_terms(claim_text)
                search_query = " ".join(key_terms[:6])  # Limit query length
                
                # Query Semantic Scholar
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": search_query,
                    "limit": 20,
                    "fields": "title,abstract,year,citationCount,authors,venue"
                }
                
                headers = {}
                if hasattr(rag_context, 'semantic_scholar_api_key') and os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
                    headers["x-api-key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                
                response = requests.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("data", [])
                    
                    # Filter and process papers
                    processed_papers = []
                    for paper in papers:
                        if paper.get("abstract") and paper.get("title"):
                            processed_papers.append({
                                "title": paper["title"],
                                "abstract": paper["abstract"],
                                "year": paper.get("year", 0),
                                "citations": paper.get("citationCount", 0),
                                "venue": paper.get("venue", ""),
                                "authors": [a.get("name", "") for a in paper.get("authors", [])]
                            })
                    
                    results[claim_text] = processed_papers
                    
                    # Cache result
                    cache_key = generate_cache_key(claim_text)
                    cache[cache_key] = (processed_papers, datetime.now())
                    
                elif response.status_code == 429:
                    time.sleep(5)
                    continue
                else:
                    print(f"SS API error: {response}")
                    results[claim_text] = []
                    
            except Exception as e:
                print(f"Error querying SS for claim: {e}")
                results[claim_text] = []
            
            # Rate limiting
            time.sleep(1)
    
    # Save updated cache
    save_cache(cache)
    return results

def extract_search_terms(text: str) -> List[str]:
    """Extract meaningful search terms from claim text."""
    if not text.strip():
        return []
    
    # Use spaCy for intelligent term extraction
    doc = rag_context.nlp(text)
    
    terms = []
    
    # Extract noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and len(chunk.text) > 2:
            terms.append(chunk.text.lower())
    
    # Extract technical terms (proper nouns, capitalized words)
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"] and len(token.text) > 3:
            terms.append(token.text.lower())
    
    # Extract entities
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "PERSON", "EVENT"] and len(ent.text) > 2:
            terms.append(ent.text.lower())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms

def calculate_literature_prevalence(claim_text: str, ss_results: List[Dict]) -> float:
    """Calculate how prevalent this claim/concept is in literature."""
    if not ss_results:
        return 0.3  # Neutral score when no literature found
    
    claim_embedding = rag_context.sentence_model.encode([claim_text])[0]
    
    # Calculate similarity to abstracts
    similarities = []
    weighted_similarities = []
    current_year = datetime.now().year
    
    for paper in ss_results:
        abstract = paper.get("abstract", "")
        if not abstract.strip():
            continue
            
        try:
            abstract_embedding = rag_context.sentence_model.encode([abstract])[0]
            similarity = float(cosine_similarity([claim_embedding], [abstract_embedding])[0][0])
            similarities.append(similarity)
            
            # Weight by recency and citations
            year = paper.get("year", 0)
            citations = paper.get("citations", 0)
            
            # Recency weight (more recent = higher weight)
            recency_weight = 1.0
            if year > 0:
                years_ago = current_year - year
                recency_weight = max(0.3, 1.0 - (years_ago * 0.05))
            
            # Citation weight (more cited = higher weight)
            citation_weight = min(2.0, 1.0 + (citations / 1000))
            
            weighted_sim = similarity * recency_weight * citation_weight
            weighted_similarities.append(weighted_sim)
            
        except Exception as e:
            continue
    
    if not similarities:
        return 0.3
    
    max_similarity = max(similarities)
    avg_similarity = statistics.mean(similarities)
    high_sim_count = sum(1 for s in similarities if s > 0.6)
    
    weighted_avg = statistics.mean(weighted_similarities) if weighted_similarities else avg_similarity
    
    prevalence = (max_similarity * 0.4 + weighted_avg * 0.4 + 
                 (high_sim_count / len(similarities)) * 0.2)
    
    return min(0.95, max(0.05, prevalence))

def adaptive_similarity_thresholds(all_similarities: List[float]) -> Dict[str, float]:
    """Calculate adaptive thresholds based on similarity distribution."""
    if not all_similarities:
        return {"high": 0.8, "medium": 0.6, "low": 0.4}
    
    # Calculate percentiles for adaptive thresholds
    similarities = sorted(all_similarities)
    n = len(similarities)
    
    if n < 5:
        return {"high": 0.8, "medium": 0.6, "low": 0.4}
    
    high_threshold = similarities[int(n * 0.85)]
    medium_threshold = similarities[int(n * 0.65)] 
    low_threshold = similarities[int(n * 0.4)] 
    
    high_threshold = max(0.7, min(0.9, high_threshold))
    medium_threshold = max(0.5, min(0.8, medium_threshold))
    low_threshold = max(0.3, min(0.6, low_threshold))
    
    return {
        "high": high_threshold,
        "medium": medium_threshold,
        "low": low_threshold
    }

def expand_context_with_literature(contexts: List[Dict], claim_text: str, ss_results: List[Dict]) -> List[Dict]:
    """Expand document context with literature findings."""
    if not ss_results:
        return contexts
    
    claim_embedding = rag_context.sentence_model.encode([claim_text])[0]
    relevant_papers = []
    
    for paper in ss_results[:10]: 
        abstract = paper.get("abstract", "")
        if not abstract.strip():
            continue
            
        try:
            abstract_embedding = rag_context.sentence_model.encode([abstract])[0]
            similarity = float(cosine_similarity([claim_embedding], [abstract_embedding])[0][0])
            
            if similarity > 0.4: 
                relevant_papers.append({
                    "title": paper.get("title", ""),
                    "abstract": abstract,
                    "year": paper.get("year", 0),
                    "similarity": similarity,
                    "citations": paper.get("citations", 0)
                })
        except Exception:
            continue
    
    # Sort by similarity and add to contexts
    relevant_papers.sort(key=lambda x: x["similarity"], reverse=True)
    
    literature_contexts = []
    for paper in relevant_papers[:3]:
        literature_contexts.append({
            "section_id": f"literature_{paper['title'][:20]}",
            "heading": f"Literature: {paper['title'][:50]}...",
            "text": paper["abstract"],
            "similarity": paper["similarity"],
            "entities": [],
            "concepts": [],
            "source": "semantic_scholar",
            "year": paper["year"],
            "citations": paper["citations"]
        })
    
    expanded_contexts = contexts + literature_contexts
    expanded_contexts.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    return expanded_contexts

def calibrate_confidence_with_literature(claim: Dict, ss_results: List[Dict], prevalence_score: float) -> float:
    """Calibrate confidence based on literature validation."""
    original_confidence = claim.get("confidence", 0.5)
    
    if not ss_results:
        # No literature found - moderate confidence reduction
        return max(0.2, original_confidence * 0.8)
    
    # Adjust confidence based on literature prevalence
    if prevalence_score > 0.8:
        # High prevalence = well-established = high confidence in low novelty
        return min(0.95, original_confidence * 1.2)
    elif prevalence_score > 0.6:
        # Medium prevalence = some support = moderate confidence
        return original_confidence
    else:
        # Low prevalence = novel claim = confidence depends on claim strength
        claim_type = claim.get("claim_type", "")
        if claim_type in ["breakthrough", "significant"]:
            return max(0.3, original_confidence * 0.9)  # Slight reduction
        else:
            return max(0.4, original_confidence * 0.95)  # Minimal reduction

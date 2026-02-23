from citationedge.models.paper import Paper
from typing import List, Dict, Tuple, Set
import re
from collections import defaultdict, Counter
from transformers import pipeline
from citationedge.utils.model_initializer import _get_models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def extract_keywords(paper: Paper, num_keywords: int = 10) -> List[str]:
    """
    Advanced dynamic keyword extraction using multiple complementary approaches:
    1. Domain-specific entity recognition
    2. Contextual embedding clustering
    3. Semantic coherence scoring
    4. Cross-validation with paper structure
    
    Args:
        paper: Pydantic Paper object
        num_keywords: Number of keywords to extract
        
    Returns:
        List of high-quality extracted keywords
    """
    
    try:
        models = _get_models()
        
        # Combine relevant text sources with weights
        text_sources = _prepare_weighted_text(paper)
        
        # Extract candidate terms using multiple methods
        candidates = _extract_candidate_terms(text_sources, models)
        
        # Score candidates using contextual embeddings
        scored_candidates = _score_candidates_contextually(candidates, text_sources, models)

        # Apply semantic filtering and deduplication
        filtered_candidates = _semantic_filter_and_deduplicate(scored_candidates, models)
        
        # Cross-validate with paper structure
        final_keywords = _cross_validate_with_structure(filtered_candidates, paper, models)
        
        # Select top keywords with diversity
        result = _select_diverse_keywords(final_keywords, num_keywords, models)
        
        print(f"[DEBUG] Final keywords: {result}")
        return result
        
    except Exception as e:
        print(f"[ERROR] Keyword extraction failed: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise

def _prepare_weighted_text(paper: Paper) -> Dict[str, Tuple[str, float]]:
    """Prepare text sources with importance weights"""
    sources = {}
    
    # Title (highest weight)
    if paper.title:
        sources['title'] = (paper.title, 3.0)
    
    # Abstract (high weight)
    if paper.abstract:
        sources['abstract'] = (paper.abstract, 2.5)
    
    # Introduction (high weight)
    intro_text = paper.get_section_text('Introduction')
    if intro_text:
        sources['introduction'] = (intro_text, 2.0)
    
    # Conclusion (medium-high weight)
    conclusion_text = paper.get_section_text('Conclusion') or paper.get_section_text('Conclusions')
    if conclusion_text:
        sources['conclusion'] = (conclusion_text, 1.8)
    
    # Method/Methodology (medium weight)
    method_text = (paper.get_section_text('Method') or 
                   paper.get_section_text('Methodology') or 
                   paper.get_section_text('Methods'))
    if method_text:
        sources['method'] = (method_text, 1.5)
    
    # Full text (base weight, truncated for efficiency)
    if paper.full_text:
        # Use middle portion to avoid references/boilerplate
        text_len = len(paper.full_text)
        start_idx = min(text_len // 4, 2000)
        end_idx = max(text_len * 3 // 4, text_len - 2000)
        sources['full_text'] = (paper.full_text[start_idx:end_idx], 1.0)
    
    return sources

def _extract_candidate_terms(text_sources: Dict[str, Tuple[str, float]], models) -> List[Tuple[str, float, str]]:
    """Extract candidate terms using multiple NLP approaches"""
    candidates = []
    
    for source_name, (text, weight) in text_sources.items():
        
        # Method 1: Named Entity Recognition with scientific focus
        entities = _extract_scientific_entities(text, models['nlp'])
        for entity, confidence in entities:
            candidates.append((entity, confidence * weight, f"entity_{source_name}"))
        
        # Method 2: Noun phrase extraction with filtering
        noun_phrases = _extract_meaningful_noun_phrases(text, models['nlp'])
        for phrase, score in noun_phrases:
            candidates.append((phrase, score * weight, f"phrase_{source_name}"))
        
        # Method 3: KeyBERT with domain adaptation
        keybert_terms = _extract_keybert_terms(text, models['keyword_model'], weight)
        candidates.extend(keybert_terms)
    
    return candidates

def _extract_scientific_entities(text: str, nlp_model) -> List[Tuple[str, float]]:
    """Extract scientifically relevant named entities"""
    doc = nlp_model(text)
    entities = []
    
    # Scientific entity types with confidence scores
    scientific_labels = {
        'ORG': 0.8,      # Organizations, institutions
        'PRODUCT': 0.9,   # Software, tools, methods
        'EVENT': 0.7,     # Conferences, workshops
        'GPE': 0.6,       # Geographic entities (for datasets)
        'PERSON': 0.3,    # Authors (lower priority)
        'WORK_OF_ART': 0.8, # Papers, books
        'LAW': 0.9,       # Algorithms, principles
        'LANGUAGE': 0.9,  # Programming languages
    }
    
    for ent in doc.ents:
        if ent.label_ in scientific_labels:
            # Filter out common noise
            if _is_valid_scientific_entity(ent.text):
                confidence = scientific_labels[ent.label_]
                entities.append((ent.text.strip(), confidence))
    
    return entities

def _is_valid_scientific_entity(text: str) -> bool:
    """Validate if entity is scientifically meaningful"""
    text = text.strip().lower()
    
    # Filter out noise patterns
    noise_patterns = [
        r'^\d+$',                    # Pure numbers
        r'^[a-z]$',                  # Single letters
        r'^(figure|table|section)\s*\d*$',  # Figure/table references
        r'^(the|a|an)\s+',           # Articles at start
        r'^\W+$',                    # Pure punctuation
        r'^(et|al|etc)\.?$',         # Common abbreviations
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, text):
            return False
    
    # Must have reasonable length
    if len(text) < 3 or len(text) > 100:
        return False
    
    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    return True

def _extract_meaningful_noun_phrases(text: str, nlp_model) -> List[Tuple[str, float]]:
    """Extract meaningful noun phrases with quality scoring"""
    doc = nlp_model(text)
    phrases = []
    
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        
        # Quality scoring based on linguistic features
        score = _score_noun_phrase_quality(chunk, doc)
        
        if score > 0.3 and _is_valid_scientific_entity(phrase):
            phrases.append((phrase, score))
    
    # Also extract compound terms
    compound_terms = _extract_compound_terms(doc)
    phrases.extend(compound_terms)
    
    return phrases

def _score_noun_phrase_quality(chunk, doc) -> float:
    """Score noun phrase quality based on linguistic features"""
    score = 0.5  # Base score
    
    # Length penalty/bonus
    if 2 <= len(chunk) <= 4:
        score += 0.2
    elif len(chunk) > 6:
        score -= 0.3
    
    # POS pattern scoring
    pos_pattern = [token.pos_ for token in chunk]
    
    # Prefer ADJ + NOUN patterns
    if any(pos == 'ADJ' for pos in pos_pattern) and any(pos == 'NOUN' for pos in pos_pattern):
        score += 0.3
    
    # Prefer compound nouns
    if pos_pattern.count('NOUN') >= 2:
        score += 0.2
    
    # Penalty for determiners
    if any(token.pos_ == 'DET' for token in chunk):
        score -= 0.2
    
    # Bonus for technical-sounding terms
    text = chunk.text.lower()
    if any(suffix in text for suffix in ['tion', 'sion', 'ment', 'ness', 'ing', 'ity']):
        score += 0.1
    
    # Penalty for very common words
    common_starts = {'this', 'that', 'these', 'those', 'some', 'many', 'most', 'all'}
    if chunk[0].text.lower() in common_starts:
        score -= 0.4
    
    return max(0.0, min(1.0, score))

def _extract_compound_terms(doc) -> List[Tuple[str, float]]:
    """Extract compound technical terms"""
    compounds = []
    
    # Look for patterns like "machine learning", "neural network"
    for i in range(len(doc) - 1):
        if (doc[i].pos_ in ['NOUN', 'ADJ'] and 
            doc[i+1].pos_ == 'NOUN' and 
            not doc[i].is_stop and 
            not doc[i+1].is_stop):
            
            compound = f"{doc[i].text} {doc[i+1].text}"
            if _is_valid_scientific_entity(compound):
                compounds.append((compound, 0.7))
    
    return compounds

def _extract_keybert_terms(text: str, keybert_model, weight: float) -> List[Tuple[str, float, str]]:
    """Extract terms using KeyBERT with improved parameters"""
    try:
        keywords = keybert_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words=None,  # Handle stopwords dynamically
            use_mmr=True,
            diversity=0.8,
            top_n=min(20, len(text.split()) // 5)
        )
        
        result = []
        for term, score in keywords:
            if _is_valid_scientific_entity(term):
                result.append((term, score * weight, "keybert"))
        
        return result
    except Exception as e:
        print(f"[WARNING] KeyBERT extraction failed: {e}")
        return []

def _score_candidates_contextually(candidates: List[Tuple[str, float, str]], 
                                 text_sources: Dict[str, Tuple[str, float]], 
                                 models) -> List[Tuple[str, float]]:
    """Score candidates using contextual embeddings"""
    
    # Group candidates by term
    term_scores = defaultdict(list)
    for term, score, source in candidates:
        term_scores[term].append((score, source))
    
    # Calculate embeddings for context
    all_text = " ".join([text for text, _ in text_sources.values()])
    context_embedding = models['sentence_model'].encode([all_text])[0]
    
    scored_terms = []
    for term, scores in term_scores.items():
        # Aggregate scores from different sources
        total_score = sum(score for score, _ in scores)
        source_diversity = len(set(source for _, source in scores))
        
        # Contextual relevance score
        term_embedding = models['sentence_model'].encode([term])[0]
        relevance = cosine_similarity([term_embedding], [context_embedding])[0][0]
        
        # Combined score
        final_score = total_score * (1 + 0.3 * source_diversity) * (0.5 + 0.5 * relevance)
        scored_terms.append((term, final_score))
    
    return sorted(scored_terms, key=lambda x: x[1], reverse=True)

def _semantic_filter_and_deduplicate(scored_candidates: List[Tuple[str, float]], 
                                   models) -> List[Tuple[str, float]]:
    """Remove semantically similar terms and filter noise"""
    
    if not scored_candidates:
        return []
    
    # Calculate embeddings for all candidates
    terms = [term for term, _ in scored_candidates]
    embeddings = models['sentence_model'].encode(terms)
    
    # Semantic deduplication
    filtered = []
    used_indices = set()
    
    for i, (term, score) in enumerate(scored_candidates):
        if i in used_indices:
            continue
            
        # Find semantically similar terms
        similarities = cosine_similarity([embeddings[i]], embeddings)[0]
        similar_indices = [j for j, sim in enumerate(similarities) 
                          if sim > 0.8 and j != i and j not in used_indices]
        
        # Keep the highest scoring among similar terms
        best_score = score
        best_term = term
        best_idx = i
        
        for j in similar_indices:
            if scored_candidates[j][1] > best_score:
                best_score = scored_candidates[j][1]
                best_term = scored_candidates[j][0]
                best_idx = j
        
        filtered.append((best_term, best_score))
        used_indices.add(best_idx)
        used_indices.update(similar_indices)
    
    return filtered

def _cross_validate_with_structure(candidates: List[Tuple[str, float]], 
                                 paper: Paper, 
                                 models) -> List[Tuple[str, float]]:
    """Cross-validate keywords with paper structure"""
    
    validated = []
    
    for term, score in candidates:
        # Check frequency across sections
        section_presence = 0
        total_sections = len(paper.sections)
        
        if total_sections > 0:
            for section_text in [s.text for s in paper.sections.values()]:
                if term.lower() in section_text.lower():
                    section_presence += 1
            
            presence_ratio = section_presence / total_sections
            
            # Boost terms that appear in multiple sections
            if presence_ratio > 0.3:
                score *= 1.3
            elif presence_ratio > 0.1:
                score *= 1.1
        
        # Check if term appears in references (boost for established concepts)
        reference_boost = 0
        for ref in paper.references:
            if term.lower() in ref.title.lower():
                reference_boost += 0.1
        
        score += min(reference_boost, 0.3)
        
        validated.append((term, score))
    
    return sorted(validated, key=lambda x: x[1], reverse=True)

def _select_diverse_keywords(candidates: List[Tuple[str, float]], 
                           num_keywords: int, 
                           models) -> List[str]:
    """Select diverse keywords avoiding redundancy"""
    
    if len(candidates) <= num_keywords:
        return [term for term, _ in candidates]
    
    selected = []
    remaining = candidates.copy()
    
    # Always include the top candidate
    selected.append(remaining[0][0])
    selected_embeddings = [models['sentence_model'].encode([remaining[0][0]])[0]]
    remaining = remaining[1:]
    
    # Greedily select diverse terms
    while len(selected) < num_keywords and remaining:
        best_idx = 0
        best_diversity_score = -1
        
        for i, (term, score) in enumerate(remaining):
            term_embedding = models['sentence_model'].encode([term])[0]
            
            # Calculate minimum similarity to already selected terms
            if selected_embeddings:
                similarities = [cosine_similarity([term_embedding], [emb])[0][0] 
                              for emb in selected_embeddings]
                max_similarity = max(similarities)
                diversity_bonus = 1 - max_similarity
            else:
                diversity_bonus = 1
            
            # Combined score: original score + diversity bonus
            combined_score = score * 0.7 + diversity_bonus * 0.3
            
            if combined_score > best_diversity_score:
                best_diversity_score = combined_score
                best_idx = i
        
        # Add the most diverse term
        selected_term = remaining[best_idx][0]
        selected.append(selected_term)
        selected_embeddings.append(models['sentence_model'].encode([selected_term])[0])
        remaining.pop(best_idx)
    
    print(f"[DEBUG] Selected diverse keywords: {selected}")
    return selected

def generate_semantic_keywords(paper: Paper, num_keywords: int = 10) -> List[str]:
    """
    Generate semantic keywords using advanced clustering and topic modeling
    """
    
    try:
        models = _get_models()
        
        # Use multi-level text analysis
        text_hierarchy = _create_text_hierarchy(paper)
        
        # Generate semantic clusters
        clusters = _generate_semantic_clusters(text_hierarchy, models, num_keywords)
        
        # Extract representative terms from each cluster
        keywords = _extract_cluster_representatives(clusters, models)
        
        # Ensure diversity and quality
        final_keywords = _ensure_semantic_quality(keywords, paper, models)
        
        return final_keywords[:num_keywords]
        
    except Exception as e:
        print(f"[ERROR] Semantic keyword generation failed: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # Fallback to simple extraction
        return _fallback_keyword_extraction(paper, num_keywords)

def _create_text_hierarchy(paper: Paper) -> Dict[str, List[str]]:
    """Create hierarchical text representation"""
    hierarchy = {
        'sentences': [],
        'paragraphs': [],
        'sections': []
    }
    
    # Combine high-value text
    combined_text = ""
    if paper.title:
        combined_text += paper.title + ". "
    if paper.abstract:
        combined_text += paper.abstract + " "
    
    # Add key sections
    for section_name, section in paper.sections.items():
        if section.text and len(section.text) > 100:
            combined_text += section.text + " "
    
    # Split into hierarchy levels
    import nltk
    try:
        sentences = nltk.sent_tokenize(combined_text)
        hierarchy['sentences'] = [s for s in sentences if len(s.split()) > 5]
        
        # Group sentences into paragraphs (every 3-5 sentences)
        paragraph_size = 4
        for i in range(0, len(sentences), paragraph_size):
            para = " ".join(sentences[i:i+paragraph_size])
            if len(para.split()) > 20:
                hierarchy['paragraphs'].append(para)
        
        # Use sections as-is
        hierarchy['sections'] = [s.text for s in paper.sections.values() 
                               if s.text and len(s.text) > 200]
        
    except Exception as e:
        print(f"[WARNING] Text hierarchy creation failed: {e}")
        # Fallback to simple splitting
        hierarchy['sentences'] = combined_text.split('. ')
    
    return hierarchy

def _generate_semantic_clusters(text_hierarchy: Dict[str, List[str]], 
                              models, 
                              num_clusters: int) -> List[List[str]]:
    """Generate semantic clusters from text hierarchy"""
    from sklearn.cluster import KMeans
    
    # Use paragraphs for clustering (good balance of context and granularity)
    texts = text_hierarchy.get('paragraphs', text_hierarchy.get('sentences', []))
    
    if len(texts) < num_clusters:
        return [[text] for text in texts]
    
    # Generate embeddings
    embeddings = models['sentence_model'].encode(texts)
    
    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Group texts by cluster
    clusters = [[] for _ in range(num_clusters)]
    for text, label in zip(texts, cluster_labels):
        clusters[label].append(text)
    
    return clusters

def _extract_cluster_representatives(clusters: List[List[str]], models) -> List[str]:
    """Extract representative keywords from each cluster"""
    keywords = []
    
    for cluster in clusters:
        if not cluster:
            continue
            
        # Combine cluster text
        cluster_text = " ".join(cluster)
        
        # Extract key terms using KeyBERT
        try:
            cluster_keywords = models['keyword_model'].extract_keywords(
                cluster_text,
                keyphrase_ngram_range=(1, 2),
                use_mmr=True,
                diversity=0.6,
                top_n=3
            )
            
            for kw, score in cluster_keywords:
                if _is_valid_scientific_entity(kw) and score > 0.3:
                    keywords.append(kw)
                    
        except Exception as e:
            print(f"[WARNING] Cluster keyword extraction failed: {e}")
    
    return keywords

def _ensure_semantic_quality(keywords: List[str], paper: Paper, models) -> List[str]:
    """Ensure semantic quality and remove low-quality terms"""
    if not keywords:
        return []
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_keywords.append(kw)
    
    # Score by relevance to paper
    scored_keywords = []
    paper_context = f"{paper.title} {paper.abstract}"
    
    for kw in unique_keywords:
        # Simple relevance scoring
        relevance = 0
        kw_lower = kw.lower()
        
        # Boost if in title/abstract
        if kw_lower in paper_context.lower():
            relevance += 0.5
        
        # Boost if appears multiple times
        text_lower = paper.full_text.lower()
        count = text_lower.count(kw_lower)
        relevance += min(count * 0.1, 0.5)
        
        # Boost technical-sounding terms
        if any(suffix in kw_lower for suffix in ['tion', 'ment', 'ing', 'ness', 'ity', 'ism']):
            relevance += 0.2
        
        scored_keywords.append((kw, relevance))
    
    # Sort by relevance and return
    scored_keywords.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in scored_keywords]

def _fallback_keyword_extraction(paper: Paper, num_keywords: int) -> List[str]:
    """Simple fallback keyword extraction"""
    text = f"{paper.title} {paper.abstract} {paper.full_text[:5000]}"
    words = text.lower().split()
    
    # Simple frequency counting with filtering
    word_freq = Counter()
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if (len(clean_word) > 3 and 
            clean_word not in STOP_WORDS and 
            not clean_word.isdigit()):
            word_freq[clean_word] += 1
    
    return [word for word, _ in word_freq.most_common(num_keywords)]

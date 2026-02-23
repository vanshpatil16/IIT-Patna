import json
import json
from typing import List, Dict, Any

def shortlist_top_claims(claims: List[Dict], top_k: int = 3, debug: bool = True) -> List[Dict]:
    """
    Shortlist top K claims per section using various strategies.
    Rejects claims with 8 words or less.
    
    Args:
        claims: List of extracted claims
        top_k: Number of top claims to select per section (used only for sections without subsections)
        debug: Print debugging information
    
    Returns:
        List of top claims per section
    """
    
    if debug:
        print(f"📊 Total input claims: {len(claims)}")
    
    # Filter out claims with 8 words or less
    filtered_claims = []
    for claim in claims:
        claim_text = claim.get('text', '').strip()
        word_count = len(claim_text.split()) if claim_text else 0
        
        if word_count > 8:
            filtered_claims.append(claim)
    
    if debug:
        rejected_count = len(claims) - len(filtered_claims)
        print(f"🚫 Rejected {rejected_count} claims with ≤8 words")
        print(f"✅ Remaining claims after filtering: {len(filtered_claims)}")
    
    # Use filtered_claims instead of claims for the rest of the function
    claims = filtered_claims
    
    # Group claims by main section and subsections
    sectioned_claims = {}
    section_structure = {}  # Track which sections have subsections
    
    for claim in claims:
        section = claim.get('section', 'unknown')
        
        # Extract main section number (e.g., "5.1" -> "5", "3.4" -> "3")
        if '.' in section:
            main_section = section.split('.')[0]
        else:
            main_section = section
        
        # Initialize main section if not exists
        if main_section not in sectioned_claims:
            sectioned_claims[main_section] = []
            section_structure[main_section] = {'subsections': set(), 'has_main': False}
        
        # Track subsections and main sections
        if '.' in section:
            section_structure[main_section]['subsections'].add(section)
        else:
            section_structure[main_section]['has_main'] = True
        
        sectioned_claims[main_section].append(claim)
    
    if debug:
        print(f"Main sections found: {list(sectioned_claims.keys())}")
        for main_section, section_claims in sectioned_claims.items():
            subsections = section_structure[main_section]['subsections']
            has_main = section_structure[main_section]['has_main']
            print(f"   - {main_section}: {len(section_claims)} claims")
            if subsections:
                print(f"     Subsections: {sorted(subsections)}")
            if has_main:
                print(f"     Has main section content: Yes")
    
    shortlisted_claims = []
    
    for main_section, section_claims in sectioned_claims.items():
        subsections = section_structure[main_section]['subsections']
        has_main = section_structure[main_section]['has_main']
        
        def composite_score(claim):
            confidence = claim.get('confidence', 0) * 0.3
            novelty = claim.get('novelty_score', 0) * 0.25
            context_rel = claim.get('context_relevance', 0) * 0.2
            
            # Section weighting
            section_weights = {
                'Abstract': 1.0, 'Introduction': 0.9, 'Conclusions': 0.95,
                'Results': 0.8, 'Discussion': 0.7, 'Methods': 0.6
            }
            section_weight = section_weights.get(main_section, 0.4) * 0.15
            
            # Graph connectivity bonus
            connections = min(claim.get('graph_connections', 0) / 10, 0.1)
            
            return confidence + novelty + context_rel + section_weight + connections
        
        # Sort all claims by composite score
        sorted_claims = sorted(section_claims, key=composite_score, reverse=True)
        
        if subsections:
            # Section has subsections
            total_units = len(subsections)
            if has_main:
                total_units += 1  # Add 1 for main section
            
            target_claims = total_units + 2
            
            # Group claims by subsection/main section
            subsection_claims = {}
            for claim in section_claims:
                claim_section = claim.get('section', 'unknown')
                if claim_section not in subsection_claims:
                    subsection_claims[claim_section] = []
                subsection_claims[claim_section].append(claim)
            
            selected = []
            
            # Select 1 claim from each subsection
            for subsection in sorted(subsections):
                if subsection in subsection_claims:
                    subsection_sorted = sorted(subsection_claims[subsection], key=composite_score, reverse=True)
                    if subsection_sorted:
                        selected.append(subsection_sorted[0])
            
            # Select 1 claim from main section if it exists
            if has_main and main_section in subsection_claims:
                main_sorted = sorted(subsection_claims[main_section], key=composite_score, reverse=True)
                if main_sorted:
                    selected.append(main_sorted[0])
            
            # Select 2 additional claims from anywhere in the section (excluding already selected)
            selected_ids = {id(claim) for claim in selected}
            remaining_claims = [claim for claim in sorted_claims if id(claim) not in selected_ids]
            
            additional_needed = target_claims - len(selected)
            if additional_needed > 0:
                selected.extend(remaining_claims[:additional_needed])
            
            if debug:
                print(f"Section '{main_section}' has {len(subsections)} subsections + {'main section' if has_main else 'no main section'}")
                print(f"Target claims: {target_claims}, Selected: {len(selected)}")
        
        else:
            # Section has no subsections, select exactly top_k claims
            selected = sorted_claims[:top_k]
            
            if debug:
                print(f"Section '{main_section}' has no subsections, selected {len(selected)} claims")
        
        shortlisted_claims.extend(selected)
    
    if debug:
        print(f"Total shortlisted claims: {len(shortlisted_claims)}")
    
    return shortlisted_claims
  
def shortlist_gaps(gaps_json: List[Dict]) -> List[Dict]:
    s_gaps = []
    high_count = 0
    medium_count = 0 
    low_count = 0
    
    for gap in gaps_json:

        if 'json_parsing' in gap:
            continue
        importance = gap.get("importance", "").lower()
        
        if importance == 'high' and high_count < 10:
            s_gaps.append(gap)
            high_count += 1
        elif importance == 'medium' and medium_count < 10:
            s_gaps.append(gap)
            medium_count += 1
        elif importance == 'low' and low_count < 5:
            s_gaps.append(gap)
            low_count += 1
    
    return s_gaps

from citationedge.models.rag_graph import rag_context
from citationedge.services.rag_service import *
from typing import *
import networkx as nx
import os
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_context(query_text: str, doc_graph: nx.DiGraph, k: int = 5) -> List[Dict]:
    """Enhanced context retrieval with literature expansion."""
    if not query_text or not query_text.strip():
        return []
    
    # Get document-internal context (original function)
    query_embedding = rag_context.sentence_model.encode([query_text])[0]
    relevant_contexts = []
    
    for node_id, node_data in doc_graph.nodes(data=True):
        if node_data.get("type") == "section":
            section_text = node_data.get("text", "")
            if section_text and section_text.strip():
                try:
                    section_embedding = rag_context.sentence_model.encode([section_text])[0]
                    similarity = float(cosine_similarity([query_embedding], [section_embedding])[0][0])
                    
                    connected_entities = []
                    connected_concepts = []
                    
                    for neighbor in doc_graph.neighbors(node_id):
                        neighbor_data = doc_graph.nodes[neighbor]
                        neighbor_name = neighbor_data.get("name", "")
                        
                        if neighbor_name and neighbor_name.strip():
                            if neighbor_data.get("type") == "entity":
                                connected_entities.append(neighbor_name.strip())
                            elif neighbor_data.get("type") == "concept":
                                connected_concepts.append(neighbor_name.strip())
                    
                    relevant_contexts.append({
                        "section_id": node_id,
                        "heading": node_data.get("heading", ""),
                        "text": section_text,
                        "similarity": similarity,
                        "entities": connected_entities,
                        "concepts": connected_concepts,
                        "source": "document"
                    })
                except Exception:
                    continue
    
    # Sort by similarity and return top k
    relevant_contexts.sort(key=lambda x: x["similarity"], reverse=True)
    return relevant_contexts[:k]

def query_similar_claims_neo4j(claim_text: str) -> List[Dict]:
    """Query Neo4j for similar claims."""
    if not claim_text or not claim_text.strip():
        return []
    
    try:
        # First try to get claim embedding
        claim_embedding = rag_context.sentence_model.encode([claim_text])[0]
        
        # Query Neo4j for existing claims - using only properties that exist
        query = """
        MATCH (p:Paper)-[:MAKES_CLAIM]->(c:Claim)
        RETURN c.text as text, p.title as paper_title
        LIMIT 50
        """
        
        # Use session to run query
        with rag_context.graph.session() as session:
            result = session.run(query)
            results = list(result)
        
        if not results:
            return []
        
        # Calculate similarities - Handle None values
        candidate_texts = []
        valid_results = []
        
        for record in results:
            text = record["text"] if record["text"] else ""
            if text and text.strip():
                candidate_texts.append(text.strip())
                valid_results.append(record)
        
        if not candidate_texts:
            return []
        
        candidate_embeddings = rag_context.sentence_model.encode(candidate_texts)
        similarities = cosine_similarity([claim_embedding], candidate_embeddings)[0]
        
        similar_claims = []
        for i, record in enumerate(valid_results):
            similarity_score = float(similarities[i])
            if similarity_score > 0.4:  # Lower threshold
                similar_claims.append({
                    "text": record["text"] if record["text"] else "",
                    "paper_title": record["paper_title"] if record["paper_title"] else "",
                    "similarity": similarity_score
                })
        
        similar_claims.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_claims[:5]   
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        return []
    
        
def extract_entities_and_concepts(text: str) -> Tuple[List[str], List[str], List[Dict]]:
    """Extract entities, concepts, and relationships from text."""
    if not text or not text.strip():
        return [], [], []
    
    doc = rag_context.nlp(text)
    
    entities = []
    for ent in doc.ents:
        if ent.text and ent.text.strip() and ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
            entities.append(ent.text.strip())
    
    concepts = []
    for chunk in doc.noun_chunks:
        if chunk.text and chunk.text.strip() and len(chunk.text.split()) <= 4 and len(chunk.text) > 3:
            concepts.append(chunk.text.lower().strip())
    
    relationships = []
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"] and token.head.pos_ == "VERB":
            object_texts = []
            for child in token.head.children:
                if child.dep_ in ["dobj", "pobj"] and child.text and child.text.strip():
                    object_texts.append(child.text.strip())
            
            if token.text and token.text.strip() and token.head.text and token.head.text.strip():
                relationships.append({
                    "subject": token.text.strip(),
                    "predicate": token.head.text.strip(),
                    "object": object_texts
                })
    
    return list(set([e for e in entities if e])), list(set([c for c in concepts if c])), relationships


def build_document_knowledge_graph(paper_json: Dict) -> nx.DiGraph:
    """Build a knowledge graph from the document for better retrieval."""
    doc_graph = nx.DiGraph()
    
    # Extract entities and relationships from the paper
    paper_id = paper_json.get("metadata", {}).get("id", "unknown")
    title = paper_json.get("metadata", {}).get("title", "")
    abstract = paper_json.get("metadata", {}).get("abstractText", "")
    
    # Add paper node
    doc_graph.add_node(f"paper_{paper_id}", 
                      type="paper", 
                      title=title, 
                      abstract=abstract)
    
    # Process each section
    sections = paper_json.get("metadata", {}).get("sections", [])
    
    for i, section in enumerate(sections):
        section_text = section.get("text", "")
        section_heading = section.get("heading", f"Section_{i}")
        
        if section_text and len(section_text.strip()) > 50:
            # Extract entities and concepts from section
            entities, concepts, relationships = extract_entities_and_concepts(section_text)
            
            # Add section node
            section_id = f"section_{i}"
            doc_graph.add_node(section_id, 
                              type="section", 
                              heading=section_heading,
                              text=section_text[:500])  # Store first 500 chars
            
            # Connect paper to section
            doc_graph.add_edge(f"paper_{paper_id}", section_id, relation="contains")
            
            # Add entities and concepts - FIXED: Handle None values
            for entity in entities:
                if entity and entity.strip():  # Check for None and empty strings
                    entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                    doc_graph.add_node(entity_id, type="entity", name=entity)
                    doc_graph.add_edge(section_id, entity_id, relation="mentions")
            
            for concept in concepts:
                if concept and concept.strip():  # Check for None and empty strings
                    concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                    doc_graph.add_node(concept_id, type="concept", name=concept)
                    doc_graph.add_edge(section_id, concept_id, relation="discusses")
    
    return doc_graph

#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz
#!pip install yake keybert neo4j torch

import os
import subprocess
import re
import json
import glob
import numpy as np
import torch
from neo4j import GraphDatabase
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import yake

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
scibert_model = SentenceTransformer('allenai/scibert_scivocab_uncased', device=device)
keybert_model = KeyBERT()
nlp = spacy.load("en_core_sci_lg")
kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1)

use_amp = torch.cuda.is_available()

def pdf_to_json(pdf_path: str, output_dir: str = "./json") -> str:
    """Extract structured text from PDF using Science Parse"""
    os.makedirs(output_dir, exist_ok=True)
    
    folder = os.path.dirname(pdf_path)
    pdf_file = os.path.basename(pdf_path)
    
    base_filename = os.path.splitext(pdf_file)[0]
    expected_json_path = os.path.join(output_dir, base_filename + ".pdf.json")
    
    if os.path.exists(expected_json_path):
        return expected_json_path
    
    # FIXME: hardcoded java path, should make this configurable
    command = [
        "Java/jdk-1.8/bin/java.exe", 
        "-Xmx8g",
        "-jar", 
        "science-parse-cli-assembly-2.0.3.jar", 
        folder, 
        "-o", 
        output_dir
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            if os.path.exists(expected_json_path):
                return expected_json_path
            else:
                return ""
        else:
            return ""
            
    except Exception as e:
        return ""

def get_scibert_embedding(text: str) -> List[float]:
    """Get 768-dim embedding from SciBERT"""
    try:
        if not text or not text.strip():
            return [0.0] * 768
        
        # trim text to avoid memory issues
        text = text[:5000]
        
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                embedding = scibert_model.encode(
                    text, 
                    convert_to_numpy=True,
                    device=device,
                    batch_size=1,
                    show_progress_bar=False
                )
        else:
            embedding = scibert_model.encode(
                text, 
                convert_to_numpy=True,
                device=device,
                batch_size=1,
                show_progress_bar=False
            )
        
        if embedding.shape[0] != 768:
            return [0.0] * 768
        
        return embedding.tolist()
    except:
        return [0.0] * 768

def get_batch_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Process multiple texts at once for better GPU usage"""
    try:
        if not texts:
            return []
        
        processed_texts = []
        for text in texts:
            if text and text.strip():
                processed_texts.append(text[:5000])
            else:
                processed_texts.append("")
        
        all_embeddings = []
        
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i+batch_size]
            
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    batch_embeddings = scibert_model.encode(
                        batch,
                        convert_to_numpy=True,
                        device=device,
                        batch_size=len(batch),
                        show_progress_bar=False
                    )
            else:
                batch_embeddings = scibert_model.encode(
                    batch,
                    convert_to_numpy=True,
                    device=device,
                    batch_size=len(batch),
                    show_progress_bar=False
                )
            
            for embedding in batch_embeddings:
                if embedding.shape[0] != 768:
                    all_embeddings.append([0.0] * 768)
                else:
                    all_embeddings.append(embedding.tolist())
        
        # cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return all_embeddings
    except:
        return [[0.0] * 768] * len(texts)

def extract_keywords(text: str, title: str = "", abstract: str = "", author_keywords: List[str] = None) -> Dict[str, float]:
    """Get keywords using multiple extraction methods"""
    try:
        if not text or not text.strip():
            return {}
            
        results = {}
        
        # author provided keywords get highest priority
        if author_keywords:
            for kw in author_keywords:
                results[kw.lower().strip()] = 1.0
        
        context = ""
        if title:
            context += title + " "
        if abstract:
            context += abstract + " "
        
        # don't process too much text at once
        text_for_extraction = context + text[:20000]
        
        # keybert approach
        keybert_keywords = keybert_model.extract_keywords(text_for_extraction, 
                                                         keyphrase_ngram_range=(1, 3), 
                                                         stop_words='english', 
                                                         top_n=20)
        
        # yake approach  
        yake_keywords = kw_extractor.extract_keywords(text_for_extraction)
        
        for kw, score in keybert_keywords:
            results[kw.lower().strip()] = max(score, results.get(kw.lower().strip(), 0))
            
        for kw, score in yake_keywords:
            # yake scores are inverted (lower = better)
            normalized_score = 1 - (score / 1.0) if score < 1.0 else 0.1
            results[kw.lower().strip()] = max(normalized_score, results.get(kw.lower().strip(), 0))
        
        # spacy NER for technical terms
        doc = nlp(text_for_extraction[:10000])
        for ent in doc.ents:
            if ent.label_ in ["METHOD", "TASK", "TOOL", "METRIC"]:
                results[ent.text.lower().strip()] = max(0.8, results.get(ent.text.lower().strip(), 0))
        
        # keep top 10
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return results
    
    except:
        return {}

def extract_claims_batch(sections: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Find claims across all sections efficiently"""
    try:
        all_claims = []
        claim_texts = []
        claim_metadata = []
        
        # patterns that often indicate claims
        claim_patterns = ["we show", "we demonstrate", "we propose", "we present", 
                         "we introduce", "results indicate", "we find", "we conclude",
                         "data suggests", "evidence shows", "we argue", "this work"]
        
        for section_title, section_text in sections:
            if not section_text or not section_text.strip():
                continue
                
            doc = nlp(section_text[:10000])
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) < 20: 
                    continue
                    
                is_claim = False
                for pattern in claim_patterns:
                    if pattern in sent_text.lower():
                        is_claim = True
                        break
                
                # check for modal verbs and comparatives
                has_modality = any(token.dep_ == "aux" and token.lemma_ in ["can", "will", "should", "may"] for token in sent)
                has_comparative = any(token.tag_ in ["JJR", "RBR"] for token in sent)
                
                if is_claim or has_modality or has_comparative:
                    claim_texts.append(sent_text)
                    claim_metadata.append({
                        "location": section_title,
                        "has_citation": any(token.dep_ == "pobj" and token.head.lemma_ == "in" for token in sent)
                    })
        
        # process embeddings in batches
        if claim_texts:
            claim_embeddings = get_batch_embeddings(claim_texts, batch_size=32)
            
            for i, (text, metadata, embedding) in enumerate(zip(claim_texts, claim_metadata, claim_embeddings)):
                claim = {
                    "text": text,
                    "embedding": embedding,
                    "location": metadata["location"],
                    "has_citation": metadata["has_citation"]
                }
                all_claims.append(claim)
        
        # don't return too many claims
        return all_claims[:20]
        
    except:
        return []

def create_paper_schema(tx):
    """Setup database constraints"""
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:SectionTitle) REQUIRE (t.title, t.paper_name) IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Reference) REQUIRE (r.title, r.year) IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.text)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (kw:Keyword) ON (kw.text)")

def create_graph(tx, paper_data):
    """Insert paper data into graph database"""
    
    # main paper node
    tx.run("""
        MERGE (p:Paper {name: $name})
        SET p.title = $title,
            p.abstractText = $abstractText,
            p.year = $year,
            p.source = $source,
            p.embedding = $embedding
    """, **paper_data)
    
    # keywords
    tx.run("""
        MERGE (kg:KeywordGroup {paper_name: $paper_name})
        MERGE (p:Paper {name: $paper_name})
        MERGE (p)-[:HAS_KEYWORD_GROUP]->(kg)
        SET kg.author_keywords = $author_keywords,
            kg.extracted_keywords = $extracted_keywords
    """, paper_name=paper_data["name"], 
         author_keywords=json.dumps(paper_data.get("author_keywords", [])),
         extracted_keywords=json.dumps(paper_data.get("extracted_keywords", {})))
    
    # individual keyword nodes
    for keyword, score in paper_data.get("extracted_keywords", {}).items():
        if not keyword or len(keyword) < 3:
            continue
            
        tx.run("""
            MERGE (k:Keyword {text: $keyword})
            SET k.score = $score
            MERGE (kg:KeywordGroup {paper_name: $paper_name})
            MERGE (kg)-[:CONTAINS {score: $score}]->(k)
        """, keyword=keyword, score=score, paper_name=paper_data["name"])
    
    # sections
    for idx, (sec_title, sec_text, sec_embed) in enumerate(paper_data["sections"]):
        tx.run("""
            MERGE (t:SectionTitle {title: $title, paper_name: $paper_name})
            MERGE (c:SectionContent {text: $text})
            SET c.embedding = $embedding
            MERGE (p:Paper {name: $paper_name})
            MERGE (p)-[:HAS_SECTION {order: $order}]->(t)
            MERGE (t)-[:HAS_CONTENT]->(c)
        """, title=sec_title, text=sec_text, embedding=sec_embed, 
           paper_name=paper_data["name"], order=idx)
    
    # claims
    for claim in paper_data.get("claims", []):
        claim_text = claim.get("text", "")
        if not claim_text:
            continue
            
        claim_location = claim.get("location", "unknown")
            
        tx.run("""
            CREATE (c:Claim {text: $text})
            SET c.embedding = $embedding,
                c.has_citation = $has_citation,
                c.location = $location
            MERGE (p:Paper {name: $paper_name})
            MERGE (p)-[:MAKES_CLAIM]->(c)
            
            WITH c, p
            OPTIONAL MATCH (p)-[:HAS_SECTION]->(t:SectionTitle)
            WHERE t.title CONTAINS $location OR $location = 'unknown'
            WITH c, t LIMIT 1
            MERGE (t)-[:CONTAINS_CLAIM]->(c)
        """, text=claim_text, embedding=claim.get("embedding", [0.0]*768),
           has_citation=claim.get("has_citation", False), 
           location=claim_location,
           paper_name=paper_data["name"])
    
    # reference group
    tx.run("""
        MERGE (rg:ReferenceGroup {paper_name: $paper_name})
        MERGE (p:Paper {name: $paper_name})
        MERGE (p)-[:HAS_REFERENCES]->(rg)
    """, paper_name=paper_data["name"])

    # references
    for ref in paper_data["references"]:
        tx.run("""
            MERGE (r:Reference {title: $title, year: $year})
            SET r.authors = $authors,
                r.venue = $venue,
                r.citeRegEx = $citeRegEx,
                r.shortCiteRegEx = $shortCiteRegEx
            MERGE (rg:ReferenceGroup {paper_name: $paper_name})
            MERGE (rg)-[:CONTAINS]->(r)
            MERGE (p:Paper {name: $paper_name})
            MERGE (p)-[:CITES]->(r)
        """, **ref, paper_name=paper_data["name"])
        
    # citation contexts
    for citation, context_data in paper_data.get("citation_contexts", {}).items():
        tx.run("""
            MATCH (p:Paper {name: $paper_name})
            MATCH (r:Reference)
            WHERE r.citeRegEx = $citation OR r.shortCiteRegEx = $citation
            CREATE (ctx:CitationContext {
                context: $context,
                type: $type,
                embedding: $embedding
            })
            MERGE (p)-[:HAS_CONTEXT]->(ctx)
            MERGE (ctx)-[:REFERS_TO]->(r)
        """, paper_name=paper_data["name"],
           citation=citation,
           context=context_data["context"],
           type=context_data["type"],
           embedding=context_data["embedding"])
    
    # topic nodes for main keywords
    top_keywords = sorted(paper_data.get("extracted_keywords", {}).items(), 
                        key=lambda x: x[1], reverse=True)[:5]
    
    for kw, score in top_keywords:
        tx.run("""
            MERGE (t:Topic {name: $keyword})
            MERGE (p:Paper {name: $paper_name})
            MERGE (p)-[:BELONGS_TO {score: $score}]->(t)
        """, keyword=kw, score=score, paper_name=paper_data["name"])

def build_knowledge_base_heuristic_marker_based():
    """Main processing function"""
    
    json_files = glob.glob(os.path.join(os.getenv("DATA_DIR", "./json"), "*.json"))
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER") 
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise ValueError("Missing Neo4j credentials in environment variables")
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # setup db schema
    with driver.session() as session:
        session.write_transaction(create_paper_schema)
    
    for idx, file_path in enumerate(json_files):
        
        # handle PDF files
        if file_path.endswith('.pdf'):
            file_path = pdf_to_json(file_path)
            if not file_path:
                continue
                
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        meta = data.get("metadata") or {}
        paper_name = data.get("name") or os.path.basename(file_path)
        title = meta.get("title") or ""
        abstract = meta.get("abstractText") or ""
        sections_raw = meta.get("sections") or []
        refs_raw = meta.get("references") or []
        author_keywords = meta.get("keywords", [])

        # build full text
        full_text_parts = [title, abstract]
        for sec in sections_raw:
            if sec is None:
                continue
            sec_text = sec.get("text") or ""
            full_text_parts.append(sec_text)
        
        full_text = "\n".join(full_text_parts)
        
        # get paper embedding
        paper_embedding = get_scibert_embedding(full_text)
        
        # extract keywords
        extracted_keywords = extract_keywords(full_text, title, abstract, author_keywords)
        
        # merge subsections into parent sections
        parent_sections = []
        current_parent = None
        for sec in sections_raw:
            if sec is None:
                continue
            heading = sec.get("heading") or "Untitled Section"
            text = sec.get("text") or ""
            
            # check if this is a main section (like "1. Introduction") vs subsection ("1.1 Background")
            if re.match(r"^\d+\.\s", heading) and not re.search(r"\d+\.\d", heading):
                current_parent = {"heading": heading, "text": text}
                parent_sections.append(current_parent)
            else:
                if current_parent is not None:
                    current_parent["text"] += "\n" + text
                else:
                    current_parent = {"heading": heading, "text": text}
                    parent_sections.append(current_parent)

        # get section embeddings in batch
        section_texts = [parent["text"] for parent in parent_sections]
        section_titles = [parent["heading"] for parent in parent_sections]
        
        if section_texts:
            section_embeddings = get_batch_embeddings(section_texts, batch_size=16)
        else:
            section_embeddings = []
        
        sections = []
        for i, parent in enumerate(parent_sections):
            sec_title = parent["heading"]
            sec_text = parent["text"]
            sec_embedding = section_embeddings[i] if i < len(section_embeddings) else [0.0] * 768
            sections.append((sec_title, sec_text, sec_embedding))
            
        # extract claims
        section_pairs = [(title, text) for title, text, _ in sections]
        claims = extract_claims_batch(section_pairs)

        # process references
        references = []
        for ref in refs_raw:
            if ref is None:
                continue
            authors = ref.get("author") or []
            if not isinstance(authors, list):
                authors = [str(authors)]
            references.append({
                "title": ref.get("title") or "Untitled",
                "authors": ", ".join(authors),
                "venue": ref.get("venue") or "",
                "year": ref.get("year") or 0,
                "citeRegEx": ref.get("citeRegEx") or "",
                "shortCiteRegEx": ref.get("shortCiteRegEx") or ""
            })
            
        paper_data = {
            "name": paper_name,
            "title": title,
            "abstractText": abstract,
            "year": meta.get("year") or 0,
            "source": meta.get("source") or "",
            "embedding": paper_embedding,
            "author_keywords": author_keywords,
            "extracted_keywords": extracted_keywords,
            "sections": sections,
            "claims": claims,
            "references": references,
            "citation_contexts": {}
        }
        
        try:
            with driver.session() as session:
                session.write_transaction(create_graph, paper_data)
                
            # memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except:
            continue

    # cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    driver.close()

build_knowledge_base_heuristic_marker_based()
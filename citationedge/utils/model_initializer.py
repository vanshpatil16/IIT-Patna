from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pytextrank
from typing import *
from groq import Groq
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'
os.environ['HF_HOME'] = 'D:/huggingface_cache'

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
_cached_models = None

def _get_models():
    """Load models from local directory."""
    global _cached_models
    
    if _cached_models is not None:
        return _cached_models
    
    _cached_models = {}
    _cached_models['sentence_model'] = SentenceTransformer('./citationedge/services/models/scibert_sentence')
    _cached_models['keyword_model'] = KeyBERT(model=_cached_models['sentence_model'])
    _cached_models['nlp'] = spacy.load("en_core_web_lg")
    _cached_models['nlp'].add_pipe("textrank")
    _cached_models['claim_tokenizer'] = AutoTokenizer.from_pretrained('./citationedge/services/models/scibert_tokenizer')
    _cached_models['claim_model'] = AutoModelForSequenceClassification.from_pretrained('./citationedge/services/models/scibert_classifier')
    
    return _cached_models

def initialize_models(groq_api_key: str, neo4j_uri: str = None, neo4j_auth: Tuple[str, str] = None):
    """
    Initialize all required models and connections for argumentation analysis.
    
    Args:
        groq_api_key: API key for Groq
        neo4j_uri: Neo4j database URI (optional)
        neo4j_auth: Neo4j authentication tuple (username, password) (optional)
    
    Returns:
        Dict containing initialized models and connections
    """
    models = {
        'groq_client': Groq(api_key=groq_api_key),
        'nlp': spacy.load("en_core_web_lg"),
        'sentence_transformer': SentenceTransformer('all-MiniLM-L6-v2'),
        'argument_tokenizer': AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir="D:/huggingface_cache"),
        'argument_classifier': AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli", 
            from_flax=True, cache_dir="D:/huggingface_cache"
        ),
        'scientific_embedder': SentenceTransformer('allenai/scibert_scivocab_uncased')
    }
    
    if neo4j_uri and neo4j_auth:
        models['graph'] = Graph(neo4j_uri, auth=neo4j_auth)
    
    return models

def initialize_neo4j_connection(neo4j_credentials=None):
    """Initialize Neo4j database connection."""
    return Graph(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

    

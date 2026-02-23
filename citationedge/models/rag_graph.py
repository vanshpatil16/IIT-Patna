from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
import networkx as nx
from neo4j import GraphDatabase
import spacy
from sentence_transformers import SentenceTransformer
from groq import Groq  # Assuming groq module
from py2neo import Graph

class RAGContext(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    groq_client: Optional[Groq] = None
    sentence_model: Optional[SentenceTransformer] = None
    nlp: Optional[Any] = None
    graph: Optional[Graph] = None
    knowledge_graph: Optional[nx.DiGraph] = None

# A single global instance of RAGContext to hold all shared objects
rag_context = RAGContext()

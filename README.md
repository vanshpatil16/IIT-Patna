# CitationEdge

<!--[![Paper](https://img.shields.io/badge/Paper-WSDM%202026-blue)](#citation)
[![Demo](https://img.shields.io/badge/🚀%20Demo-Streamlit-yellow)](#running-the-streamlit-interface)
[![API](https://img.shields.io/badge/API-FastAPI-green)](#running-the-fastapi-server)-->

**An Automated Framework for Assessing Manuscript Argumentation Quality and Citation Relevance**

## 🌟 Overview

The scientific publishing ecosystem faces significant challenges in efficiently evaluating manuscript quality, particularly regarding how effectively papers engage with relevant literature. **CitationEdge** addresses these critical limitations by providing:

- **Automated Citation Gap Detection**: Identifies overlooked but relevant citations using semantic search and knowledge graphs
- **Argumentation Quality Assessment**: Evaluates claim strength, evidence support, and argumentative depth using modern NLP
- **Comprehensive Literature Analysis**: Generates quantitative literary scores and actionable feedback
- **Multi-layered Evaluation**: Combines keyword extraction, semantic search, knowledge graph construction, and argumentation theory

## 🎯 Key Features

### 📚 Citation Analysis
- **Gap Identification**: Discovers missing but relevant citations using hybrid retrieval strategies
- **Relevance Scoring**: Calculates multi-factor relevance scores considering semantic similarity, recency, and citation impact
- **Priority Classification**: Categorizes citation gaps into high, medium, and low importance levels

### 🔍 Claim Extraction & Validation  
- **RAG-based Extraction**: Uses Retrieval-Augmented Generation to identify research claims
- **Novelty Assessment**: Evaluates claim originality against existing literature
- **Confidence Scoring**: Provides granular confidence metrics (0.1-1.0) for each claim
- **Categorization**: Classifies claims as breakthrough, significant, incremental, supportive, or methodological

### 🧠 Argumentation Analysis
- **Toulmin Model Integration**: Applies established argumentation theory principles
- **Evidence Mapping**: Links claims to supporting evidence (both explicit citations and implicit data)
- **Section-specific Analysis**: Tailors evaluation approach for different manuscript sections
- **Argument Graph Construction**: Visualizes argumentative structures using directed graphs

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database
- Required API keys (configured in `.env`)

### Installation

```bash
git clone https://github.com/ChintanD2205/CitationEdge.git
cd CitationEdge
pip install -r citationedge/requirements.txt
```

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys and configuration
nano .env
```

### Setting Up Neo4j Database

```bash
# Start Neo4j database
neo4j start

# Access Neo4j interface at http://localhost:7474/

# Populate database with research papers
# From Terminal run
python citationedge.services.knowledge_base.py
```

## 💻 Usage

### Streamlit Web Interface

```bash
streamlit run citationedge.streamlit_app.py
```

Access the interface at `http://localhost:8501`

### FastAPI Server

```bash
uvicorn citationedge.app:app --reload
```

- **API Server**: `http://127.0.0.1:8000`
- **Interactive Docs**: `http://127.0.0.1:8000/docs`

### API Usage

```bash
curl -X 'POST' 'http://127.0.0.1:8000/citation-edge-paper' \
     -F 'file=@path/to/manuscript.pdf'
```

## 📁 Project Structure

```
CitationEdge/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment template
├── citationedge/
│   ├── app.py                             # FastAPI application
│   ├── streamlit_app.py                   # Streamlit web interface
│   ├── requirements.txt                   # Python dependencies
│   ├── api/                               # API endpoints and routing
│   ├── constants/                         # Configuration constants
│   ├── models/                            # Paper data models and schemas
│   ├── services/                          # Core business logic services
│   └── utils/                             # Utility functions and helpers
├── Java/jdk-1.8                           # Java library to run science-parse
├── science-parse-cli-assembly-2.0.3.jar   # Science Parse Jar file
```

## 🔧 Technical Architecture

### Four-Stage Pipeline

1. **Document Parsing & Keyword Extraction**
   - ScienceParse for academic document processing
   - Multi-layered keyword extraction (KeyBERT + SciBERT, TextRank, Enhanced TF-IDF)
   - Domain-specific NER with confidence scoring

2. **Claim Extraction & Analysis**
   - RAG-based claim identification using retrieval-augmented generation
   - Novelty assessment through semantic similarity analysis
   - Confidence scoring and categorical classification

3. **Citation Gap Analysis**
   - Hybrid retrieval strategy (Neo4j + Semantic Scholar API)
   - Multi-factor relevance scoring with recency and citation impact
   - Automated section assignment and priority classification

4. **Argumentation Quality Assessment**
   - Toulmin model decomposition of argument structures
   - Evidence mapping and semantic validation
   - Section-specific analysis for Methods, Results, Discussion, and Abstract

### Key Technologies

- **NLP Models**: SciBERT, KeyBERT, all-MiniLM-L6-v2, BART
- **Knowledge Graph**: Neo4j with custom schema
- **Retrieval**: Semantic Scholar API, RAG pipeline
- **Scoring**: MMR, cosine similarity, multi-factor relevance
- **Visualization**: NetworkX argument graphs

## 📊 Output & Reports

### Generated Artifacts

- **Comprehensive PDF Report**: Detailed analysis with visual insights
- **JSON Export**: Structured data for custom analysis
- **Citation Recommendations**: Prioritized list with relevance scores and usage context
- **Claim Validation Report**: Argumentation strengths and weaknesses assessment
- **Literary Score**: Quantitative measure of literature engagement

### Sample Outputs

```
📈 Literary Score: 7.2/10
🔍 Citation Gaps Found: 12 (5 High Priority, 4 Medium, 3 Low)
📝 Claims Extracted: 8 (2 Breakthrough, 3 Significant, 3 Incremental)
⚖️ Argumentation Score: 8.1/10
```

## **Contact**  

For queries, reach out to:  
- **Prabhat Kumar Bharti**: [dept.csprabhat@gmail.com](mailto:dept.csprabhat@gmail.com)
- **Chintan Dodia**: [chintan222005@gmail.com](mailto:chintan222005@gmail.com) 
- **Mihir Panchal**: [mihirpanchal5400@gmail.com](mailto:mihirpanchal5400@gmail.com) 

---

## **License**  

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.  

---
<!--**🔗 Quick Links**: [Demo](http://localhost:8501) | [API Docs](http://127.0.0.1:8000/docs) | [Paper](#citation) | [Neo4j Interface](http://localhost:7474/)-->

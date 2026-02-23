from typing import Dict, List, Any, Optional
from citationedge.models.paper import Paper, Author, Reference, Section, ReferenceMention
def paper_to_json(paper):
    """Convert Paper Pydantic object to JSON format for RAG pipeline."""
    
    sections_list = []
    for heading, section in paper.sections.items():
        sections_list.append({
            "heading": heading,
            "text": section.text
        })
    
    references_list = []
    for ref in paper.references:
        references_list.append({
            "title": ref.title,
            "author": ref.authors,
            "venue": ref.venue,
            "citeRegEx": ref.cite_regex,
            "year": ref.year
        })
    
    authors_list = []
    for author in paper.authors:
        if author.name and not author.email and not author.affiliation:
            authors_list.append(author.name)
        else:
            author_dict = {}
            if author.name:
                author_dict["name"] = author.name
            if author.email:
                author_dict["email"] = author.email
            if author.affiliation:
                author_dict["affiliation"] = author.affiliation
            authors_list.append(author_dict)
    
    return {
        "name": paper.file_name,
        "metadata": {
            "title": paper.title,
            "abstractText": paper.abstract,
            "authors": authors_list,
            "emails": paper.emails,
            "sections": sections_list,
            "references": references_list,
            "year": paper.year,
            "creator": paper.creator,
            "source": paper.source
        }
    }

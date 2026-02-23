from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import json


class Author(BaseModel):
    """Model for paper authors"""
    name: Optional[str] = None
    email: Optional[str] = None
    affiliation: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from original data


class Reference(BaseModel):
    """Model for paper references"""
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    venue: str = ""
    cite_regex: str = Field(default="", alias="citeRegEx")
    short_cite_regex: Optional[str] = Field(default=None, alias="shortCiteRegEx")
    year: Optional[int] = None
    original_reference: Optional[Dict[str, Any]] = None
    
    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class Section(BaseModel):
    """Model for paper sections"""
    text: str = ""
    original_section: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"


class ReferenceMention(BaseModel):
    """Model for reference mentions in the text"""
    reference_id: Optional[str] = None
    context: Optional[str] = None
    
    class Config:
        extra = "allow"


class Paper(BaseModel):
    """Complete model for a scientific paper"""
    file_name: str = ""
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    sections: Dict[str, Section] = Field(default_factory=dict)
    references: List[Reference] = Field(default_factory=list)
    authors: List[Author] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    year: int = 0
    creator: Optional[str] = None
    source: str = ""
    reference_mentions: List[ReferenceMention] = Field(default_factory=list)
    
    class Config:
        extra = "allow"
        validate_assignment = True
    
    def get_section_text(self, heading: str) -> Optional[str]:
        """Get text for a specific section by heading"""
        section = self.sections.get(heading)
        return section.text if section else None
    
    def get_authors_names(self) -> List[str]:
        """Get list of author names"""
        return [author.name for author in self.authors if author.name]
    
    def get_reference_titles(self) -> List[str]:
        """Get list of reference titles"""
        return [ref.title for ref in self.references if ref.title]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility"""
        return self.dict()

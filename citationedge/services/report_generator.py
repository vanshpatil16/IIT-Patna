import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import io
from groq import Groq

# ReportLab imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF


class CitationEdgeReportGenerator:
    """
    A comprehensive PDF report generator for research paper analysis.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            groq_api_key: Optional Groq API key for AI-powered recommendations
        """
        self.groq_client = None
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
        
        # Color scheme
        self.colors = {
            'primary': HexColor('#2E86AB'),
            'secondary': HexColor('#A23B72'),
            'accent': HexColor('#F18F01'),
            'success': HexColor('#C73E1D'),
            'light_grey': HexColor('#F5F5F5'),
            'dark_grey': HexColor('#4A4A4A'),
            'text': HexColor('#333333')
        }
        
        # Initialize styles
        self.styles = self._create_styles()
    
    def _create_styles(self) -> Dict:
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()
        
        custom_styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=self.colors['primary'],
                fontName='Helvetica-Bold'
            ),
            'subtitle': ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=self.colors['dark_grey'],
                fontName='Helvetica'
            ),
            'section_header': ParagraphStyle(
                'SectionHeader',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=15,
                spaceBefore=20,
                textColor=self.colors['primary'],
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=self.colors['primary'],
                borderPadding=10,
                backColor=self.colors['light_grey']
            ),
            'subsection_header': ParagraphStyle(
                'SubsectionHeader',
                parent=styles['Heading3'],
                fontSize=14,
                spaceAfter=10,
                spaceBefore=15,
                textColor=self.colors['secondary'],
                fontName='Helvetica-Bold'
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=10,
                alignment=TA_JUSTIFY,
                textColor=self.colors['text'],
                fontName='Helvetica'
            ),
            'caption': ParagraphStyle(
                'Caption',
                parent=styles['Normal'],
                fontSize=9,
                spaceAfter=5,
                alignment=TA_CENTER,
                textColor=self.colors['dark_grey'],
                fontName='Helvetica-Oblique'
            ),
            'footer': ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=self.colors['dark_grey'],
                fontName='Helvetica'
            )
        }
        
        return custom_styles
    
    def _create_citation_quality_chart(self, score_data: Dict) -> Drawing:
        """Create a radar chart for citation quality metrics."""
        drawing = Drawing(400, 300)
        
        # Create bar chart for citation metrics
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 200
        chart.width = 300
        
        # Prepare data
        categories = list(score_data.get('component_scores', {}).keys())
        values = list(score_data.get('component_scores', {}).values())
        
        chart.data = [values]
        chart.categoryAxis.categoryNames = [cat.replace('_', ' ').title() for cat in categories]
        chart.valueAxis.valueMax = 100
        chart.valueAxis.valueMin = 0
        
        # Styling
        chart.bars[0].fillColor = self.colors['primary']
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.fontSize = 8
        chart.valueAxis.labels.fontSize = 8
        
        drawing.add(chart)
        
        # Add title
        title = String(200, 270, 'Citation Quality Metrics', textAnchor='middle')
        title.fontSize = 12
        title.fillColor = self.colors['primary']
        drawing.add(title)
        
        return drawing
    
    def _create_insight_interpretation(self, json_data: Dict) -> List:
        """Create detailed interpretation of what the analysis results mean."""
        story = []
        
        story.append(Paragraph("🔍 What Your Analysis Reveals", self.styles['section_header']))
        
        # Get core metrics
        score_report = json_data.get('literary_score_report', {}).get('score_report', {})
        overall_score = score_report.get('overall_score', 0)
        component_scores = score_report.get('component_scores', {})
        claims = json_data.get('claims', [])
        gaps = json_data.get('categorized_gaps', [])
        
        # Overall standing interpretation
        story.append(Paragraph("Your Paper's Academic Standing", self.styles['subsection_header']))
        
        if overall_score >= 85:
            standing_text = f"""
            <b>Your paper demonstrates EXCELLENT citation practices</b> (Score: {overall_score:.1f}/100)
            <br/><br/>
            <b>What this means:</b> Your work meets high academic standards and shows rigorous engagement with existing literature. 
            Reviewers and readers will likely view your paper as well-researched and credible. Your citation practices are in the top 15% of academic papers.
            <br/><br/>
            <b>Academic Impact:</b> This level of citation quality significantly enhances your paper's chances of acceptance in high-impact journals 
            and increases its potential for future citations by other researchers.
            """
        elif overall_score >= 70:
            standing_text = f"""
            <b>Your paper shows GOOD citation practices</b> (Score: {overall_score:.1f}/100)
            <br/><br/>
            <b>What this means:</b> Your work demonstrates solid academic rigor but has room for improvement. Most of your citations are appropriate, 
            but there are opportunities to strengthen your literature foundation.
            <br/><br/>
            <b>Academic Impact:</b> Your paper is publishable but addressing the identified gaps could elevate it to a higher tier of academic journals 
            and increase its scholarly impact.
            """
        elif overall_score >= 50:
            standing_text = f"""
            <b>Your paper has MODERATE citation quality</b> (Score: {overall_score:.1f}/100)
            <br/><br/>
            <b>What this means:</b> Your literature review needs significant strengthening. While you have some relevant citations, 
            there are important gaps that could undermine your paper's credibility and impact.
            <br/><br/>
            <b>Academic Impact:</b> Reviewers may question the thoroughness of your literature review. Addressing these gaps is crucial 
            for publication in reputable journals.
            """
        else:
            standing_text = f"""
            <b>Your paper needs SUBSTANTIAL citation improvement</b> (Score: {overall_score:.1f}/100)
            <br/><br/>
            <b>What this means:</b> Your literature foundation is insufficient for academic publication. This suggests either incomplete 
            literature review or missed connections to relevant research.
            <br/><br/>
            <b>Academic Impact:</b> Significant revision is needed before submission. Reviewers will likely request major revisions 
            or reject based on inadequate literature engagement.
            """
        
        story.append(Paragraph(standing_text, self.styles['body']))
        story.append(Spacer(1, 0.3*inch))
        
        return story

    def _create_component_diagnosis(self, json_data: Dict) -> List:
        """Create user-friendly diagnosis with actual data and clear actions."""
        story = []
        
        story.append(Paragraph("📊 Your Citation Health Check", self.styles['subsection_header']))
        
        component_scores = json_data.get('literary_score_report', {}).get('score_report', {}).get('component_scores', {})
        metrics = json_data.get('literary_score_report', {}).get('score_report', {}).get('metrics', {})
        
        # Group issues by severity
        critical_issues = []
        moderate_issues = []
        strengths = []
        
        for component, score in component_scores.items():
            component_name = component.replace('_', ' ').title()
            
            
            actual_data = self._get_actual_data(component, metrics)
            action_needed = self._get_action_needed(component, score, metrics)
            
            issue_item = {
                'name': component_name,
                'score': score,
                'data': actual_data,
                'action': action_needed
            }
            
            if score < 60:
                critical_issues.append(issue_item)
            elif score < 80:
                moderate_issues.append(issue_item)
            else:
                strengths.append(issue_item)
        
        # Display critical issues first
        if critical_issues:
            story.append(Paragraph("🔴 Critical Issues (Fix These First)", self.styles['subsection_header']))
            for issue in critical_issues:
                story.extend(self._format_issue(issue, "critical"))
            story.append(Spacer(1, 0.2*inch))
        
        # Display moderate issues
        if moderate_issues:
            story.append(Paragraph("🟡 Moderate Issues (Fix Later)", self.styles['subsection_header']))
            for issue in moderate_issues:
                story.extend(self._format_issue(issue, "moderate"))
            story.append(Spacer(1, 0.2*inch))
        
        # Display strengths
        if strengths:
            story.append(Paragraph("✅ What's Working Well", self.styles['subsection_header']))
            for strength in strengths:
                story.extend(self._format_issue(strength, "strength"))
        
        return story

    def _get_actual_data(self, component: str, metrics: Dict, scores: Dict) -> str:
        """Generate adaptive insights based on component scores and metrics."""

        comp = component.lower()

        if 'coverage' in comp or 'completeness' in comp:
            total_refs = metrics.get('total_references', 0)
            gaps = metrics.get('citation_gaps', 0)
            score = scores.get('citation_completeness', 0)

            if score > 80:
                return f"Strong coverage — {total_refs} references with only {gaps} notable gaps."
            elif score > 50:
                return f"Moderate coverage — {total_refs} references, but {gaps} key papers are missing."
            else:
                return f"Coverage needs improvement — {gaps} important works missing across {total_refs} references."

        elif 'recency' in comp:
            avg_age = metrics.get('average_citation_age', 0)
            score = scores.get('citation_recency', 0)

            if score > 70:
                return f"Your citations are fairly up-to-date (avg age {avg_age:.1f} years)."
            elif score > 40:
                return f"Some citations are dated (avg age {avg_age:.1f} years) — consider adding newer studies."
            else:
                return f"Citations are largely outdated (avg age {avg_age:.1f} years) — update with recent work."

        elif 'self' in comp:
            self_pct = metrics.get('self_citation_percentage', 0) * 100
            score = scores.get('self_citation_ratio', 0)

            if score < 30:
                return f"Minimal self-citation ({self_pct:.1f}%) — healthy balance of external sources."
            elif score < 60:
                return f"Moderate self-citation ({self_pct:.1f}%) — acceptable but keep it balanced."
            else:
                return f"High self-citation ({self_pct:.1f}%) — reduce reliance on your own work for stronger credibility."

        elif 'diversity' in comp:
            score = scores.get('citation_diversity', 0)

            if score > 75:
                return "Excellent diversity — citing a broad range of sources."
            elif score > 50:
                return "Decent diversity — but could include more varied sources."
            else:
                return "Low diversity — many citations come from a narrow set of sources."

        elif 'relevance' in comp:
            score = scores.get('citation_relevance', 0)

            if score > 85:
                return "Citations strongly support your claims — excellent relevance."
            elif score > 60:
                return "Most citations are relevant, but a few don’t fully support the arguments."
            else:
                return "Several citations don’t clearly support your claims — refine selection for stronger alignment."

        return "No adaptive insight available."


    def _get_action_needed(self, component: str, score: float, metrics: Dict) -> str:
        """Provide specific actions based on component and score."""
        if 'coverage' in component.lower() or 'completeness' in component.lower():
            gaps = metrics.get('citation_gaps', 0)
            if score < 60:
                return f"Add the {min(gaps, 10)} highest-priority papers we found"
            else:
                return f"Add {min(gaps, 5)} additional key papers"
        
        elif 'recency' in component.lower():
            if score < 60:
                return "Add 8-10 papers from last 2 years"
            else:
                return "Add 3-5 recent papers from 2023-2024"
        
        elif 'relevance' in component.lower():
            if score < 60:
                return "Replace 5 weakest citations with more targeted ones"
            else:
                return "Replace 2-3 weak citations with stronger alternatives"
        
        elif 'diversity' in component.lower():
            return "Add citations from 2-3 different venues"
        
        elif 'self' in component.lower():
            return "Keep doing what you're doing"
        
        return "Review and improve as needed"

    def _format_issue(self, issue: Dict, severity: str) -> List:
        """Format each issue in a user-friendly way."""
        story = []
        
        # Issue title with emoji
        if severity == "critical":
            emoji = "❌"
        elif severity == "moderate":
            emoji = "⚠️"
        else:
            emoji = "✅"
        
        title = f"<b>{emoji} {issue['name']}</b>"
        story.append(Paragraph(title, self.styles['body']))
        
        # Show actual data
        if issue['data']:
            story.append(Paragraph(f"• {issue['data']}", self.styles['body']))
        
        # Show action needed
        if issue['action']:
            story.append(Paragraph(f"• <b>Action:</b> {issue['action']}", self.styles['body']))
        
        story.append(Spacer(1, 0.1*inch))
        return story

    def _create_citation_quality_section(self, json_data: Dict) -> List:
        """Create the citation quality assessment section without chart."""
        story = []
        
        story.append(Paragraph("2. Citation Quality Assessment", self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        score_report = json_data.get('literary_score_report', {}).get('score_report', {})
        
        # Overall status
        overall_score = score_report.get('overall_score', 0)
        
        if overall_score >= 80:
            status = "✅ Excellent"
            status_color = "green"
        elif overall_score >= 60:
            status = "⚠️ Needs Improvement"
            status_color = "orange"
        else:
            status = "❌ Critical Issues"
            status_color = "red"
        
        score_text = f"""
        <b>Overall Status: {status}</b><br/>
        <b>Citation Quality Score: {overall_score:.1f}/100</b>
        """
        story.append(Paragraph(score_text, self.styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        # Remove the chart completely - replace with diagnosis
        story.extend(self._create_component_diagnosis(json_data))
        
        return story

    def _create_quick_wins_section(self, json_data: Dict) -> List:
        """Add a quick wins section at the end."""
        story = []
        
        story.append(Paragraph("🎯 Quick Win Recommendations", self.styles['subsection_header']))
        
        quick_wins = """
        1. <b>This week:</b> Add 3 recent papers from 2023-2024<br/>
        2. <b>Next week:</b> Replace your 2 weakest citations<br/>
        3. <b>When polishing:</b> Add 2-3 foundational papers reviewers expect<br/><br/>
        <b>Estimated time to fix critical issues:</b> 4-6 hours<br/>
        <b>Expected improvement:</b> 15-20 point score increase
        """
        
        story.append(Paragraph(quick_wins, self.styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        return story
    def _create_claims_impact_analysis(self, json_data: Dict) -> List:
        """Analyze what the claims reveal about the paper's contribution."""
        story = []
        
        story.append(Paragraph("🎯 What Your Claims Reveal About Your Research", self.styles['subsection_header']))
        
        claims = json_data.get('claims', [])
        if not claims:
            story.append(Paragraph("No claims data available for analysis.", self.styles['body']))
            return story
        
        # Analyze claim patterns
        high_confidence_claims = [c for c in claims if c.get('confidence', 0) > 0.8]
        low_confidence_claims = [c for c in claims if c.get('confidence', 0) < 0.5]
        novel_claims = [c for c in claims if c.get('novelty_score', 0) > 0.7]
        
        # Overall claim assessment
        avg_confidence = sum(c.get('confidence', 0) for c in claims) / len(claims)
        avg_novelty = sum(c.get('novelty_score', 0) for c in claims) / len(claims)
        
        analysis_text = f"""
        <b>Your Research Contribution Profile:</b>
        <br/><br/>
        <b>Confidence Level:</b> {avg_confidence:.1%} average confidence
        <br/>
        • {len(high_confidence_claims)} high-confidence claims (strong evidence backing)
        <br/>
        • {len(low_confidence_claims)} low-confidence claims (need stronger support)
        <br/><br/>
        <b>Novelty Level:</b> {avg_novelty:.1%} average novelty
        <br/>
        • {len(novel_claims)} novel claims (new contributions to the field)
        <br/><br/>
        """
        
        if avg_confidence > 0.7 and avg_novelty > 0.6:
            contribution_assessment = """
            <b>Assessment:</b> You're making STRONG, NOVEL contributions with good evidence support. 
            This suggests high-impact research that advances the field significantly.
            """
        elif avg_confidence > 0.7:
            contribution_assessment = """
            <b>Assessment:</b> You're making WELL-SUPPORTED claims but they may be incremental. 
            Consider highlighting what makes your approach or findings unique.
            """
        elif avg_novelty > 0.6:
            contribution_assessment = """
            <b>Assessment:</b> You're making NOVEL claims but need stronger evidence. 
            Focus on gathering more robust support for your innovative ideas.
            """
        else:
            contribution_assessment = """
            <b>Assessment:</b> Your claims need both stronger evidence and clearer articulation of novelty. 
            Consider what unique contribution your research makes to the field.
            """
        
        story.append(Paragraph(analysis_text + contribution_assessment, self.styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        return story

    def _create_strategic_roadmap(self, json_data: Dict) -> List:
        """Create a strategic roadmap for improvement based on analysis."""
        story = []
        
        story.append(Paragraph("🗺️ Your Strategic Improvement Roadmap", self.styles['section_header']))
        
        # Priority-based roadmap
        gaps = json_data.get('categorized_gaps', [])
        high_priority_gaps = [g for g in gaps if g.get('importance') == 'high']
        
        component_scores = json_data.get('literary_score_report', {}).get('score_report', {}).get('component_scores', {})
        weak_components = [(k, v) for k, v in component_scores.items() if v < 60]
        
        # Phase 1: Immediate Actions (Week 1-2)
        story.append(Paragraph("Phase 1: Immediate Actions (Week 1-2)", self.styles['subsection_header']))
        
        immediate_actions = []
        
        # Add top 3 critical gaps
        for gap in high_priority_gaps[:3]:
            title = gap.get('title', 'Unknown')
            section = gap.get('relevant_section', 'Unknown')
            immediate_actions.append(f"Add citation: '{title}' to {section} section")
        
        # Address weakest component
        if weak_components:
            weakest = min(weak_components, key=lambda x: x[1])
            component_name = weakest[0].replace('_', ' ')
            if 'coverage' in component_name:
                immediate_actions.append(f"Expand literature search for {component_name} - aim for 5-10 additional relevant papers")
            elif 'quality' in component_name:
                immediate_actions.append(f"Replace lower-quality sources with higher-impact references")
        
        for action in immediate_actions:
            story.append(Paragraph(f"• {action}", self.styles['body']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Phase 2: Strategic Improvements (Week 3-4)
        story.append(Paragraph("Phase 2: Strategic Improvements (Week 3-4)", self.styles['subsection_header']))
        
        strategic_actions = [
            "Conduct systematic review of remaining medium-priority gaps",
            "Strengthen claim-citation alignment for low-confidence claims",
            "Review and update theoretical framework based on new citations",
            "Ensure balanced coverage across all research dimensions"
        ]
        
        for action in strategic_actions:
            story.append(Paragraph(f"• {action}", self.styles['body']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Phase 3: Final Optimization (Week 5-6)
        story.append(Paragraph("Phase 3: Final Optimization (Week 5-6)", self.styles['subsection_header']))
        
        final_actions = [
            "Integrate all new citations seamlessly into arguments",
            "Perform final citation quality check",
            "Ensure all claims have adequate supporting evidence",
            "Prepare manuscript for submission"
        ]
        
        for action in final_actions:
            story.append(Paragraph(f"• {action}", self.styles['body']))
        
        return story

    def _create_impact_prediction(self, json_data: Dict) -> List:
        """Predict the potential impact of improvements."""
        story = []
        
        story.append(Paragraph("🚀 Predicted Impact of Improvements", self.styles['subsection_header']))
        
        current_score = json_data.get('literary_score_report', {}).get('score_report', {}).get('overall_score', 0)
        high_priority_gaps = len([g for g in json_data.get('categorized_gaps', []) if g.get('importance') == 'high'])
        
        # Calculate potential improvement
        potential_improvement = min(15, high_priority_gaps * 3)  # Up to 15 points improvement
        projected_score = min(100, current_score + potential_improvement)
        
        prediction_text = f"""
        <b>Current Position:</b> {current_score:.1f}/100
        <br/>
        <b>Potential After Improvements:</b> {projected_score:.1f}/100
        <br/>
        <b>Expected Improvement:</b> +{potential_improvement:.1f} points
        <br/><br/>
        <b>What This Means for Your Paper:</b>
        <br/>
        """
        
        if projected_score >= 85:
            impact_text = """
            • <b>Journal Tier:</b> Eligible for top-tier journals in your field
            <br/>
            • <b>Review Process:</b> Likely to receive positive reviewer feedback on literature quality
            <br/>
            • <b>Citation Potential:</b> Higher likelihood of being cited by future researchers
            <br/>
            • <b>Academic Impact:</b> Positioned for significant scholarly influence
            """
        elif projected_score >= 70:
            impact_text = """
            • <b>Journal Tier:</b> Strong candidate for respected journals
            <br/>
            • <b>Review Process:</b> Literature review will likely satisfy reviewer expectations
            <br/>
            • <b>Citation Potential:</b> Good foundation for future citations
            <br/>
            • <b>Academic Impact:</b> Solid contribution to the field
            """
        else:
            impact_text = """
            • <b>Journal Tier:</b> Suitable for specialized or emerging journals
            <br/>
            • <b>Review Process:</b> May still receive requests for literature improvements
            <br/>
            • <b>Citation Potential:</b> Moderate potential for future citations
            <br/>
            • <b>Academic Impact:</b> Incremental contribution to the field
            """
        
        story.append(Paragraph(prediction_text + impact_text, self.styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        return story

    def _create_claims_distribution_chart(self, claims_data: List[Dict]) -> Drawing:
        """Create a pie chart showing distribution of claim types."""
        drawing = Drawing(400, 300)
        
        # Count claim types
        claim_types = {}
        for claim in claims_data:
            claim_type = claim.get('claim_type', 'unknown')
            claim_types[claim_type] = claim_types.get(claim_type, 0) + 1
        
        if not claim_types:
            return drawing
        
        # Create pie chart
        pie = Pie()
        pie.x = 100
        pie.y = 50
        pie.width = 200
        pie.height = 200
        
        pie.data = list(claim_types.values())
        pie.labels = list(claim_types.keys())
        
        # Color scheme
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success']]
        pie.slices.strokeColor = white
        pie.slices.strokeWidth = 2
        
        for i, color in enumerate(colors[:len(claim_types)]):
            pie.slices[i].fillColor = color
        
        drawing.add(pie)
        
        # Add title
        title = String(200, 270, 'Claims Distribution by Type', textAnchor='middle')
        title.fontSize = 12
        title.fillColor = self.colors['primary']
        drawing.add(title)
        
        return drawing
    
    def generate_report(self, json_data: Dict, output_path: str = None, return_bytes: bool = False) -> Optional[bytes]:
        """
        Generate a comprehensive PDF report from the analysis data.
        
        Args:
            json_data: The analysis data dictionary
            output_path: Optional path to save the PDF file
            return_bytes: Whether to return PDF as bytes
            
        Returns:
            PDF bytes if return_bytes is True, otherwise None
        """
        try:
            # Set up document
            if output_path:
                doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=72, leftMargin=72,
                                      topMargin=72, bottomMargin=72)
            else:
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                                      topMargin=72, bottomMargin=72)
            
            # Build content
            story = []
            
            # Title Page
            story.extend(self._create_title_page(json_data))
            story.append(PageBreak())
            
            # Executive Summary
            story.extend(self._create_executive_summary(json_data))
            story.append(PageBreak())
            
            # Section 1: Claim Analysis Deep Dive
            story.extend(self._create_claims_analysis_section(json_data))
            story.append(PageBreak())
            
            # Section 2: Citation Quality Assessment
            story.extend(self._create_citation_quality_section(json_data))
            story.append(PageBreak())
            
            # Section 3: Research Gaps and Missing References
            story.extend(self._create_research_gaps_section(json_data))
            story.append(PageBreak())
            
            # Section 4: Strategic Action Plan
            # Add quick wins section
            story.extend(self._create_quick_wins_section(json_data))
            # Build PDF
            doc.build(story)
            
            if return_bytes and not output_path:
                buffer.seek(0)
                return buffer.getvalue()
            
            print(f"Report generated successfully: {output_path}")
            return None
            
        except Exception as e:
            print(f"Error generating report: {e}")
            raise
    
    def _create_title_page(self, json_data: Dict) -> List:
        """Create the title page of the report."""
        story = []
        
        # Main title
        title = json_data.get('paper_json', {}).get('metadata', {}).get('title', 'Research Paper Analysis')
        story.append(Paragraph(title, self.styles['title']))
        story.append(Spacer(1, 0.5*inch))
        
        # Authors
        authors = json_data.get('paper_json', {}).get('metadata', {}).get('authors', [])
        if authors:
            authors_text = "Authors: " + ", ".join(authors)
            story.append(Paragraph(authors_text, self.styles['subtitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Generated by CitationEdge
        story.append(Paragraph("Generated by CitationEdge", self.styles['subtitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Generation date
        date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"Report Generated: {date_str}", self.styles['body']))
        
        return story
    
    def _create_executive_summary(self, json_data: Dict) -> List:
        """Create the executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Overall assessment
        analysis = json_data.get('literary_score_report', {}).get('analysis', {})
        overall_score = json_data.get('literary_score_report', {}).get('score_report', {}).get('overall_score', 0)
        
        summary_text = f"""
        This comprehensive analysis evaluates the research paper's citation quality, claim substantiation, 
        and identifies key areas for improvement. The paper received an overall citation quality score of 
        <b>{overall_score:.1f}/100</b>, indicating {'excellent' if overall_score >= 80 else 'good' if overall_score >= 60 else 'needs improvement'} 
        citation practices.
        """
        story.append(Paragraph(summary_text, self.styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key findings
        story.append(Paragraph("Key Findings", self.styles['subsection_header']))
        story.extend(self._create_insight_interpretation(json_data))

        # Strengths - Display only first element
        strengths = analysis.get('strengths', [])
        if strengths:
            story.append(Paragraph("<b>Strengths:</b>", self.styles['body']))
            story.append(Paragraph(f"• {strengths[0]}", self.styles['body']))
        
        # Areas for improvement - Display only first element
        weaknesses = analysis.get('weaknesses', [])
        if weaknesses:
            story.append(Paragraph("<b>Areas for Improvement:</b>", self.styles['body']))
            story.append(Paragraph(f"• {weaknesses[0]}", self.styles['body']))
        
        # Recommendations - Display only first element
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("<b>Key Recommendation:</b>", self.styles['body']))
            story.append(Paragraph(f"• {recommendations[0]}", self.styles['body']))
        
        return story

    def _create_claims_analysis_section(self, json_data: Dict) -> List:
        """Create the claims analysis section."""
        story = []
        
        story.append(Paragraph("1. Claim Analysis Deep Dive", self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        claims = json_data.get('claims', [])
        claim_arguments = json_data.get('claim_analysis', {}).get('claim_arguments', [])
        if not claims:
            story.append(Paragraph("No claims data available for analysis.", self.styles['body']))
            return story
        
        # Add claims distribution chart
        claims_chart = self._create_claims_distribution_chart(claims)
        story.append(claims_chart)
        story.append(Spacer(1, 0.2*inch))
        story.extend(self._create_claims_impact_analysis(json_data))
        
        # Group claims by type
        claims_by_type = {}
        for claim in claims:
            claim_type = claim.get('claim_type', 'unknown')
            if claim_type not in claims_by_type:
                claims_by_type[claim_type] = []
            claims_by_type[claim_type].append(claim)
        
        # Display claims by category
        for claim_type, type_claims in claims_by_type.items():
            story.append(Paragraph(f"{claim_type.title()} Claims", self.styles['subsection_header']))
            
            # Fixed: iterate through type_claims instead of claims
            for i, claim in enumerate(type_claims, 1):
                claim_text = claim.get('text', 'No claim text available')
                confidence = claim.get('confidence', 0)
                novelty_score = claim.get('novelty_score', 0)
                claim_type = claim.get('claim_type', 'unknown')
                category = claim.get('category', 'N/A')
                key_terms = claim.get('key_terms', [])
                
                # Find corresponding claim argument for AI evaluation
                ai_evaluation = None
                for arg in claim_arguments:
                    if arg.get('claim') == claim_text:
                        ai_evaluation = arg.get('ai_evaluation', {})
                        break
                
                story.append(Paragraph(f"Claim {i}:", self.styles['subsection_header']))
                
                # Create enhanced claim data
                claim_data = [
                    ['Claim Text', claim_text],
                    ['Confidence Score', f"{confidence:.2f}"],
                    ['Novelty Score', f"{novelty_score:.2f}"],
                    ['Claim Type', claim_type.title()],
                    ['Category', category],
                    ['Key Terms', ', '.join(key_terms) if key_terms else 'N/A'],
                    ['Section', claim.get('section', 'N/A')]
                ]
                
                # Add AI evaluation data if available
                if ai_evaluation:
                    reasoning = ai_evaluation.get('reasoning', 'N/A')
                    strengths = ai_evaluation.get('strengths', [])
                    weaknesses = ai_evaluation.get('weaknesses', [])
                    
                    claim_data.extend([
                        ['Claim Reasoning', reasoning[:300] + '...' if len(reasoning) > 300 else reasoning],
                        ['Strengths', '; '.join(strengths) if strengths else 'N/A'],
                        ['Weaknesses', '; '.join(weaknesses) if weaknesses else 'N/A']
                    ])

                # Wrap long text content in Paragraph objects for proper text wrapping
                wrapped_claim_data = []
                for row in claim_data:
                    label = row[0]
                    content = row[1]
                    # Wrap content in Paragraph for automatic text wrapping
                    wrapped_content = Paragraph(str(content), self.styles['body'])
                    wrapped_claim_data.append([label, wrapped_content])
                
                claim_table = Table(wrapped_claim_data, colWidths=[1.5*inch, 4*inch])
                claim_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), self.colors['light_grey']),
                    ('TEXTCOLOR', (0, 0), (-1, -1), self.colors['text']),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, self.colors['dark_grey']),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))

                story.append(claim_table)
                story.append(Spacer(1, 0.1*inch))
        
        return story

    def _create_research_gaps_section(self, json_data: Dict) -> List:
        """Create the research gaps and missing references section."""
        story = []
        
        story.append(Paragraph("3. Research Gaps and Missing References", self.styles['section_header']))
        story.append(Spacer(1, 0.2*inch))
        
        # Gap analysis summary
        categorized_gaps = json_data.get('categorized_gaps', [])
        
        summary_text = f"""
        Our analysis identified <b>{len(categorized_gaps)} potential citation gaps</b> that could strengthen 
        the paper's literature foundation. These gaps represent important works that are relevant to your 
        research but are not currently cited in your manuscript.
        """
        story.append(Paragraph(summary_text, self.styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        if not categorized_gaps:
            story.append(Paragraph("No significant citation gaps identified.", self.styles['body']))
            return story
        
        # Categorize by priority
        high_priority = [gap for gap in categorized_gaps if gap.get('importance') == 'high']
        medium_priority = [gap for gap in categorized_gaps if gap.get('importance') == 'medium']
        low_priority = [gap for gap in categorized_gaps if gap.get('importance') == 'low']
        
        # Sort by relevance score within each priority
        high_priority.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        medium_priority.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        low_priority.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Display gaps by priority
        for priority, gaps, color in [
            ('High Priority', high_priority, self.colors['success']),
            ('Medium Priority', medium_priority, self.colors['accent']),
            ('Low Priority', low_priority, self.colors['secondary'])
        ]:
            if gaps:
                story.append(Paragraph(f"{priority} Gaps ({len(gaps)})", self.styles['subsection_header']))
                
                for gap in gaps:
                    title = gap.get('title', 'Unknown Title')
                    authors = gap.get('authors', [])
                    year = gap.get('year', 'Unknown')
                    relevance_score = gap.get('relevance_score', 0)
                    relevant_section = gap.get('relevant_section', 'Unknown')
                    relevance_explanation = gap['explanation']['relevance_explanation']
                    context = gap.get('context_suggestions', 'Unknown')
                    usage_suggestion = gap['explanation']['usage_suggestion']
                    
                    # Create gap entry
                    gap_text = f"""
                    <b>{title}</b> ({year})<br/>
                    <i>Authors:</i> {', '.join(authors) if authors else 'Unknown'}<br/>
                    <i>Relevance Score:</i> {relevance_score:.2f}<br/>
                    <i>Section to be included in:</i> {relevant_section}<br/>
                    <i></i> <br/>
                    <i>Why is it relevant?:</i> {relevance_explanation}<br/>
                    <i></i> <br/>
                    <i>How to use it?:</i> {usage_suggestion}<br/>
                    <i></i> <br/>
                    <i>Context:</i> {context}<br/>
                    """
                    
                    story.append(Paragraph(gap_text, self.styles['body']))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(HRFlowable(width="100%", thickness=1, color=self.colors['light_grey']))
                    story.append(Spacer(1, 0.1*inch))
        story.extend(self._create_impact_prediction(json_data))
        story.extend(self._create_strategic_roadmap(json_data))
        return story

def create_citation_report(json_data: Dict, output_path: str = None, 
                          groq_api_key: str = None, return_bytes: bool = False) -> Optional[bytes]:
    """
    Convenience function to create a CitationEdge PDF report.
    
    Args:
        json_data: The analysis data dictionary
        output_path: Optional path to save the PDF file
        groq_api_key: Optional Groq API key for AI recommendations
        return_bytes: Whether to return PDF as bytes
        
    Returns:
        PDF bytes if return_bytes is True, otherwise None
    """
    generator = CitationEdgeReportGenerator(groq_api_key=groq_api_key)
    return generator.generate_report(json_data, output_path=output_path, return_bytes=return_bytes)


def validate_json_structure(json_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate the structure of the input JSON data.
    
    Args:
        json_data: The JSON data to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ['paper_json', 'claims', 'literary_score_report', 'categorized_gaps']
    for key in required_keys:
        if key not in json_data:
            errors.append(f"Missing required key: {key}")
    
    # Validate paper_json structure
    if 'paper_json' in json_data:
        paper_json = json_data['paper_json']
        if 'metadata' not in paper_json:
            errors.append("Missing 'metadata' in paper_json")
        else:
            metadata = paper_json['metadata']
            if 'title' not in metadata:
                errors.append("Missing 'title' in paper_json.metadata")
    
    # Validate claims structure
    if 'claims' in json_data:
        claims = json_data['claims']
        if not isinstance(claims, list):
            errors.append("'claims' should be a list")
        else:
            for i, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    errors.append(f"Claim {i} should be a dictionary")
                elif 'text' not in claim:
                    errors.append(f"Missing 'text' in claim {i}")
    
    # Validate literary_score_report structure
    if 'literary_score_report' in json_data:
        score_report = json_data['literary_score_report']
        if 'score_report' not in score_report:
            errors.append("Missing 'score_report' in literary_score_report")
    
    # Validate categorized_gaps structure
    if 'categorized_gaps' in json_data:
        gaps = json_data['categorized_gaps']
        if not isinstance(gaps, list):
            errors.append("'categorized_gaps' should be a list")
    
    return len(errors) == 0, errors


def customize_report_colors(primary: str = "#2E86AB", secondary: str = "#A23B72", 
                           accent: str = "#F18F01", success: str = "#C73E1D") -> Dict:
    """
    Create a custom color scheme for reports.
    
    Args:
        primary: Primary color (hex)
        secondary: Secondary color (hex)
        accent: Accent color (hex)
        success: Success/highlight color (hex)
        
    Returns:
        Color scheme dictionary
    """
    return {
        'primary': HexColor(primary),
        'secondary': HexColor(secondary),
        'accent': HexColor(accent),
        'success': HexColor(success),
        'light_grey': HexColor('#F5F5F5'),
        'dark_grey': HexColor('#4A4A4A'),
        'text': HexColor('#333333')
    }
"""
with open("D:\citationedge - Copy\Consistentpeer.json", 'r', encoding="utf-8") as f:
    json_data = json.load(f)

create_citation_report(
    json_data=json_data,
    output_path="citation_analysis_report.pdf"
)"""

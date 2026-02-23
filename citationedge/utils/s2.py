def parse_search_results(raw_results):
    """
    Parse the raw search results from Semantic Scholar into a structured format
    """
    papers = []
    
    paper_blocks = raw_results.strip().split("\n\n\n")
    
    for block in paper_blocks:
        if not block.strip():
            continue
            
        paper_info = {}
        lines = block.strip().split("\n")
        
        for line in lines:
            if not line.strip():
                continue
                
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "Published year":
                    paper_info["year"] = value
                elif key == "Title":
                    paper_info["title"] = value
                elif key == "Authors":
                    paper_info["authors"] = [author.strip() for author in value.split(",")]
                elif key == "Abstract":
                    paper_info["abstract"] = value
        
        if paper_info:
            papers.append(paper_info)
    return papers

from datetime import datetime

def get_current_year():
    """Get current year for calculations."""
    return datetime.now().year

def calculate_average_citation_age(references, paper_year):
    """Calculate average age of citations with error handling."""
    years = [ref.get("year") for ref in references if ref.get("year", 0) > 0]
    
    if not years:
        print("Warning: No valid citation years found in references")
        return 0.0

    avg_year = sum(years) / len(years)
    avg_age = get_current_year() - avg_year

    if avg_age < 0:
        print(f"Warning: Negative citation age. Paper year: {paper_year}, Avg citation year: {avg_year}")
        return 0.0

    print(f"Average citation age: {avg_age}")
    return avg_age

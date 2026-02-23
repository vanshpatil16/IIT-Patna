from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from citationedge.api import paper_analysis, claim_extractor, citation_gap, analyze_argumentation, literary_scorer, main_pipeline


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.include_router(paper_analysis.router)
app.include_router(claim_extractor.router)
app.include_router(citation_gap.router)
app.include_router(analyze_argumentation.router)
app.include_router(literary_scorer.router)
app.include_router(main_pipeline.router)
@app.get("/")
def root():
    return {"message": "Welcome to citationedge API!"} 

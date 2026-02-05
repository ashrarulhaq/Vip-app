from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import json
import re
import os

# Import core libraries
import pypdf
from io import BytesIO
import requests

# Job scraping
from jobspy import scrape_jobs

# AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

app = FastAPI(title="JobHuntAI API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class JobSearchRequest(BaseModel):
    job_title: str
    location: str
    experience_level: Optional[str] = None
    is_remote: bool = False
    results_wanted: int = 10

class AnalyzeJobRequest(BaseModel):
    job: dict
    resume_text: str
    provider: str = "gemini"  # "gemini" or "openrouter"
    api_key: str
    model: Optional[str] = "openai/gpt-4o-mini"

class ResumeTextRequest(BaseModel):
    text: str

# ============================================================================
# RESUME PARSING
# ============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        pdf_reader = pypdf.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse a PDF resume."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    return {"success": True, "text": text, "length": len(text)}

@app.post("/api/upload-resume-text")
async def upload_resume_text(request: ResumeTextRequest):
    """Accept resume as plain text."""
    if not request.text or len(request.text) < 50:
        raise HTTPException(status_code=400, detail="Resume text is too short")
    
    return {"success": True, "text": request.text, "length": len(request.text)}

# ============================================================================
# JOB SEARCH
# ============================================================================

@app.post("/api/search-jobs")
async def search_jobs(request: JobSearchRequest):
    """Search for jobs using python-jobspy."""
    try:
        jobs_df = scrape_jobs(
            site_name=["indeed", "linkedin", "glassdoor"],
            search_term=request.job_title,
            location=request.location,
            results_wanted=request.results_wanted,
            hours_old=72,
            is_remote=request.is_remote,
            country_indeed='India'
        )
        
        if jobs_df is None or jobs_df.empty:
            return {"success": True, "jobs": [], "count": 0}
        
        # Convert to list of dicts
        jobs = []
        for _, row in jobs_df.iterrows():
            job = {
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "location": str(row.get("location", "")),
                "description": str(row.get("description", ""))[:2000],
                "job_url": str(row.get("job_url", "")),
                "date_posted": str(row.get("date_posted", "")),
                "salary": str(row.get("min_amount", "")) + " - " + str(row.get("max_amount", "")) if row.get("min_amount") else "",
                "site": str(row.get("site", ""))
            }
            jobs.append(job)
        
        return {"success": True, "jobs": jobs, "count": len(jobs)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")

# ============================================================================
# AI ANALYSIS
# ============================================================================

def search_hr_contacts(company_name: str, job_title: str) -> str:
    """Search for HR contacts using DuckDuckGo."""
    if not DDGS_AVAILABLE:
        return ""
    
    try:
        query = f"{company_name} HR recruiter email LinkedIn {job_title}"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        return "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
    except:
        return ""

def call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini API."""
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=500, detail="Gemini not available")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    """Call OpenRouter API."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"OpenRouter error: {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

def parse_ai_response(response_text: str) -> dict:
    """Parse JSON from AI response."""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except:
        return {}

@app.post("/api/analyze-job")
async def analyze_job(request: AnalyzeJobRequest):
    """Analyze a job with AI."""
    job = request.job
    resume_text = request.resume_text
    
    # Get HR search results
    hr_search = search_hr_contacts(job.get("company", ""), job.get("title", ""))
    
    prompt = f"""You are an expert job matching AI. Analyze how well this resume matches the job.

JOB DETAILS:
Title: {job.get('title', '')}
Company: {job.get('company', '')}
Location: {job.get('location', '')}
Description: {job.get('description', '')}

RESUME:
{resume_text[:3000]}

HR SEARCH RESULTS:
{hr_search}

Return a JSON object with:
{{
    "match_score": <0-100 integer>,
    "reasoning": "<2-3 sentence explanation>",
    "missing_keywords": ["<skill1>", "<skill2>"],
    "hr_email": "<email if found, else null>",
    "hr_linkedin": "<linkedin URL if found, else null>",
    "cover_letter": "<professional cover letter in Indian business format>",
    "cold_email": "<short cold email to HR in Indian business format>"
}}

IMPORTANT: Return ONLY valid JSON, no markdown.
"""
    
    try:
        if request.provider == "gemini":
            response_text = call_gemini(prompt, request.api_key)
        else:
            response_text = call_openrouter(prompt, request.api_key, request.model or "openai/gpt-4o-mini")
        
        analysis = parse_ai_response(response_text)
        
        if not analysis:
            return {"success": False, "error": "Failed to parse AI response", "raw": response_text}
        
        return {"success": True, "analysis": analysis}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "gemini": GEMINI_AVAILABLE, "ddgs": DDGS_AVAILABLE}

# Serve frontend (for local dev)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

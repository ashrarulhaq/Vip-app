"""
JobHuntAI - Free Edition
A personal recruiter agent that scrapes jobs, matches them to your resume, 
and generates tailored application materials using AI.
"""

import streamlit as st
import pandas as pd
import json
import re
import requests
from io import BytesIO

# Import job scraping library
from jobspy import scrape_jobs

# Import PDF parsing
from pypdf import PdfReader

# Import Google Gemini AI
import google.generativeai as genai

# Import DuckDuckGo Search
from duckduckgo_search import DDGS

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_resume_text(uploaded_file) -> str:
    """Extract text from uploaded PDF resume."""
    try:
        pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def configure_gemini(api_key: str) -> bool:
    """Configure Google Gemini with the provided API key."""
    try:
        genai.configure(api_key=api_key)
        st.session_state.gemini_configured = True
        st.session_state.api_key = api_key
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False

def test_gemini_api_key() -> bool:
    """Test if the Gemini API key is valid and find a working model."""
    try:
        # Optimization: Fetch available models first to avoid blind guessing
        try:
            available_models = [m.name.replace('models/', '') for m in genai.list_models()]
        except Exception as e:
            # If list_models fails, it's likely an API key issue or region block
            st.error(f"❌ Connection Error: Unable to list models. {str(e)}")
            return False

        # Priorities
        priority_models = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-1.5-flash-001',
            'gemini-1.5-flash-002',
            'gemini-2.0-flash-exp',
            'gemini-pro'
        ]

        # Find the first available model that matches our priorities
        working_model = None
        for model in priority_models:
            if model in available_models:
                working_model = model
                break
        
        # If no priority model found, just pick the first available generation model
        if not working_model:
            for model in available_models:
                if 'generateContent' in genai.get_model(f"models/{model}").supported_generation_methods:
                    working_model = model
                    break

        if working_model:
            # Verify it actually processes prompts
            try:
                model = genai.GenerativeModel(working_model)
                response = model.generate_content("Hi")
                if response:
                    st.session_state.gemini_model = working_model
                    st.success(f"✅ Connected to fast model: {working_model}")
                    return True
            except Exception as e:
                st.error(f"❌ Model {working_model} found but failed to generate: {str(e)}")
                return False
        
        st.error(f"❌ No suitable Gemini models found. Available: {', '.join(available_models)}")
        return False
            
    except Exception as e:
        st.error(f"❌ API Key Error: {str(e)}")
        return False

def find_hr_contact(company: str, location: str) -> str:
    """Search for HR contact and LinkedIn using DuckDuckGo."""
    try:
        queries = [
            f"{company} {location} HR email",
            f"{company} {location} HR manager LinkedIn",
            f"{company} recruiter LinkedIn {location}",
            f"{company} talent acquisition LinkedIn",
            f"{company} careers contact email"
        ]
        
        results_text = ""
        with DDGS() as ddgs:
            # Try the first few queries, get top results
            # We combine results from email and linkedin queries
            for query in queries[:3]: # limit queries to avoid rate limits
                 results = list(ddgs.text(query, max_results=2))
                 for r in results:
                    results_text += f"Title: {r['title']}\nSnippet: {r['body']}\n\n"
                
        return results_text
    except Exception as e:
        return f"Search failed: {str(e)}"

def analyze_job_with_gemini(resume_text: str, job_data: dict) -> dict:
    """Analyze a single job against the resume using Gemini AI."""
    
    # Helper function to safely get string values (handles NaN from pandas)
    def safe_str(value, default='N/A', max_len=None):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        result = str(value)
        if max_len:
            result = result[:max_len]
        return result
    
    company = safe_str(job_data.get('company'))
    location = safe_str(job_data.get('location'))
    
    # External search for HR context
    external_hr_context = find_hr_contact(company, location)
    
    # Prepare job description
    job_description = f"""
    Title: {safe_str(job_data.get('title'))}
    Company: {company}
    Location: {location}
    Description: {safe_str(job_data.get('description'), max_len=3000)}
    Job Type: {safe_str(job_data.get('job_type'))}
    """
    
    prompt = f"""You are an expert career coach and ATS (Applicant Tracking System) analyzer.

RESUME:
{resume_text[:5000]}

JOB POSTING:
{job_description}

EXTERNAL SEARCH RESULTS FOR HR CONTACT:
{external_hr_context}

TASK:
1. Analyze how well this candidate's resume matches the job posting. Be critical and honest.
2. FIND THE HR EMAIL: 
   - First, check the JOB POSTING.
   - If not found there, check the EXTERNAL SEARCH RESULTS.
   - If found, extract it and specify the source ("Description" or "External Search").

Provide your analysis in the following JSON format (and ONLY JSON, no other text):
{{
    "match_score": <integer 0-100>,
    "reasoning": "<one sentence explaining the match/mismatch>",
    "missing_keywords": ["<keyword1>", "<keyword2>"],
    "hr_email": "<extracted email address or null>",
    "email_source": "<'Description', 'External Search', or null>",
    "cover_letter": "<professional cover letter under 200 words that connects specific resume achievements to job requirements>",
    "cold_email": "<short, punchy email to hiring manager under 100 words>"
}}

Remember:
- Be critical. Don't give high scores just for keyword matches.
- Look for actual experience alignment.
- The cover letter should NOT repeat the resume - it should connect achievements to job requirements.
- The cold email should be conversational and direct."""

    try:
        # Use the configured model or fallback
        model_name = st.session_state.get('gemini_model', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {
                "match_score": 0,
                "reasoning": "Error parsing AI response",
                "missing_keywords": [],
                "hr_email": None,
                "email_source": None,
                "cover_letter": "Error generating cover letter",
                "cold_email": "Error generating email"
            }
            
    except json.JSONDecodeError as e:
        return {
            "match_score": 0,
            "reasoning": f"JSON parsing error: {str(e)}",
            "missing_keywords": [],
            "hr_email": None,
            "email_source": None,
            "cover_letter": "Error generating cover letter",
            "cold_email": "Error generating email"
        }
    except Exception as e:
        return {
            "match_score": 0,
            "reasoning": f"AI error: {str(e)}",
            "missing_keywords": [],
            "hr_email": None,
            "email_source": None,
            "cover_letter": "Error generating cover letter",
            "cold_email": "Error generating email"
        }

# ... (get_score_badge remains)

def export_to_csv(analyzed_jobs: list) -> bytes:
    """Export analyzed jobs to CSV format."""
    export_data = []
    for job in analyzed_jobs:
        analysis = job.get('analysis', {})
        export_data.append({
            'Title': job.get('title', ''),
            'Company': job.get('company', ''),
            'Location': job.get('location', ''),
            'Match Score': analysis.get('match_score', 0),
            'Reasoning': analysis.get('reasoning', ''),
            'HR Email': analysis.get('hr_email', '') or 'N/A',
            'Email Source': analysis.get('email_source', '') or 'N/A',
            'Missing Keywords': ', '.join(analysis.get('missing_keywords', [])),
            'Job URL': job.get('job_url', ''),
            'Cover Letter': analysis.get('cover_letter', ''),
            'Cold Email': analysis.get('cold_email', '')
        })
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')






# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="JobHuntAI - Free Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Job cards */
    .job-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Score badges */
    .score-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .score-low {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
    
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
    
if 'analyzed_jobs' not in st.session_state:
    st.session_state.analyzed_jobs = []
    
if 'gemini_configured' not in st.session_state:
    st.session_state.gemini_configured = False

if 'offset' not in st.session_state:
    st.session_state.offset = 0

if 'job_history' not in st.session_state:
    st.session_state.job_history = []

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_resume_text(uploaded_file) -> str:
    """Extract text from uploaded PDF resume."""
    try:
        pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return None



# In session state initialization (add this if not present, or I'll just rely on the new logic setting it)
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = "gemini-1.5-flash" 

def configure_gemini(api_key: str) -> bool:
    """Configure Google Gemini with the provided API key."""
    try:
        genai.configure(api_key=api_key)
        st.session_state.gemini_configured = True
        st.session_state.api_key = api_key
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False


def test_gemini_api_key() -> bool:
    """Test if the Gemini API key is valid and find a working model."""
    try:
        # Optimization: Fetch available models first to avoid blind guessing
        try:
            available_models = [m.name.replace('models/', '') for m in genai.list_models()]
        except Exception as e:
            # If list_models fails, it's likely an API key issue or region block
            st.error(f"❌ Connection Error: Unable to list models. {str(e)}")
            return False

        # Priorities
        priority_models = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-1.5-flash-001',
            'gemini-1.5-flash-002',
            'gemini-2.0-flash-exp',
            'gemini-pro'
        ]

        # Find the first available model that matches our priorities
        working_model = None
        for model in priority_models:
            if model in available_models:
                working_model = model
                break
        
        # If no priority model found, just pick the first available generation model
        if not working_model:
            for model in available_models:
                if 'generateContent' in genai.get_model(f"models/{model}").supported_generation_methods:
                    working_model = model
                    break

        if working_model:
            # Verify it actually processes prompts
            try:
                model = genai.GenerativeModel(working_model)
                response = model.generate_content("Hi")
                if response:
                    st.session_state.gemini_model = working_model
                    st.success(f"✅ Connected to fast model: {working_model}")
                    return True
            except Exception as e:
                st.error(f"❌ Model {working_model} found but failed to generate: {str(e)}")
                return False
        
        st.error(f"❌ No suitable Gemini models found. Available: {', '.join(available_models)}")
        return False
            
    except Exception as e:
        st.error(f"❌ API Key Error: {str(e)}")
        return False


def scrape_job_listings(role: str, location: str, sites: list, num_results: int, 
                       experience: str = None, job_type: str = None, 
                       is_remote: bool = False, is_hybrid: bool = False,
                       offset: int = 0) -> pd.DataFrame:
    """Scrape jobs from selected sites using python-jobspy with filters."""
    try:
        # Map site names to jobspy format
        site_mapping = {
            "LinkedIn": "linkedin",
            "Indeed": "indeed", 
            "Glassdoor": "glassdoor",
            "ZipRecruiter": "zip_recruiter",
            "Naukri": "naukri"
        }
        
        selected_sites = [site_mapping[s] for s in sites if s in site_mapping]
        
        if not selected_sites:
            st.warning("Please select at least one job site.")
            return None
        
        with st.spinner(f"🔍 Scraping jobs from {', '.join(sites)} (Offset: {offset})..."):
            # Construct search term with experience level if provided
            search_query = f"{experience} {role}" if experience and experience != "Any" else role
            
            jobs = scrape_jobs(
                site_name=selected_sites,
                search_term=search_query,
                location=location,
                results_wanted=num_results,
                hours_old=48,  # Jobs from last 2 days (User Request)
                country_indeed='USA',
                is_remote=is_remote,
                offset=offset,
            )
            
        if jobs is not None and not jobs.empty:
            initial_count = len(jobs)
            
            # Post-processing filters
            if job_type:
                # Normalize string for comparison (remove special chars)
                clean_type = job_type.lower().replace('-', '').replace(' ', '')
                
                if 'job_type' in jobs.columns:
                    # Create normalized column for filtering
                    jobs['type_clean'] = jobs['job_type'].astype(str).str.lower().str.replace('-', '').str.replace(' ', '')
                    filtered_jobs = jobs[jobs['type_clean'].str.contains(clean_type, na=False)]
                    
                    if filtered_jobs.empty:
                        st.warning(f"⚠️ Found {initial_count} jobs, but none matched type '{job_type}'. Showing all jobs found.")
                    else:
                        jobs = filtered_jobs
            
            # Extract emails from description immediately
            if 'description' in jobs.columns:
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                
                def find_email(text):
                    if not isinstance(text, str): return None
                    matches = re.findall(email_pattern, text)
                    if not matches: return None
                    # Filter out common false positives if necessary (e.g. example.com)
                    valid_matches = [m for m in matches if 'example.com' not in m and 'email.com' not in m]
                    return valid_matches[0] if valid_matches else None
                
                jobs['hr_contact'] = jobs['description'].apply(find_email)
            
            st.success(f"✅ Found {len(jobs)} jobs!")
            return jobs
        else:
            st.warning("No jobs found with these criteria. Try relaxing the filters or checking your location.")
            return None
            
    except Exception as e:
        st.error(f"Error scraping jobs: {str(e)}")
        st.info("💡 Tip: Some sites may block requests. Try selecting different sites or waiting a few minutes.")
        return None


def analyze_job_with_gemini(resume_text: str, job_data: dict) -> dict:
    """Analyze a single job against the resume using Gemini AI."""
    
    # Helper function to safely get string values (handles NaN from pandas)
    def safe_str(value, default='N/A', max_len=None):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        result = str(value)
        if max_len:
            result = result[:max_len]
        return result
    
    company = safe_str(job_data.get('company'))
    location = safe_str(job_data.get('location'))
    
    # External search for HR context
    external_hr_context = find_hr_contact(company, location)
    
    # Prepare job description
    job_description = f"""
    Title: {safe_str(job_data.get('title'))}
    Company: {company}
    Location: {location}
    Description: {safe_str(job_data.get('description'), max_len=3000)}
    Job Type: {safe_str(job_data.get('job_type'))}
    """
    
    prompt = f"""You are an expert career coach and ATS (Applicant Tracking System) analyzer.

RESUME:
{resume_text[:5000]}

JOB POSTING:
{job_description}

EXTERNAL SEARCH RESULTS FOR HR CONTACT:
{external_hr_context}

TASK:
1. Analyze how well this candidate's resume matches the job posting. Be critical and honest.
2. FIND THE HR EMAIL AND LINKEDIN: 
   - Check both JOB POSTING and EXTERNAL SEARCH RESULTS.
   - Look for HR Manager/Recruiter Name, Email, and LinkedIn Profile URL.
   - If found, extract them.

Provide your analysis in the following JSON format (and ONLY JSON, no other text):
{{
    "match_score": <integer 0-100>,
    "reasoning": "<one sentence explaining the match/mismatch>",
    "missing_keywords": ["<keyword1>", "<keyword2>"],
    "hr_email": "<extracted email address or null>",
    "hr_linkedin": "<extracted linkedin url or null>",
    "email_source": "<'Description', 'External Search', or null>",
    "cover_letter": "<Formal Indian-style cover letter. Subject line required. Salutation: 'Respected Sir/Madam' or 'Dear Hiring Team'. Structure: Intro -> Skills/Exp -> Why this company -> Request for Interview. Tone: Respectful & Professional.>",
    "cold_email": "<Formal cold email for Indian context. Subject: Application for [Role] - [Name]. Salutation: 'Respected Sir/Madam'. Concise (100 words), highlighting key fit, requesting discussion.>"
}}

Remember:
- Use INDIAN BUSINESS FORMAT for correspondence.
- Tone: Formal, respectful, and professional (e.g., 'Respected Sir/Madam').
- Cover Letter MUST have a Subject line at the top.
- Be critical in matching (don't inflate scores)."""

    try:
        response_text = ""
        
        if st.session_state.get('provider') == "OpenRouter":
            # OpenRouter API Call
            api_key = st.session_state.get('api_key')
            model_name = st.session_state.get('openrouter_model', 'openai/gpt-4o-mini')
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501", # Optional, for including your app on openrouter.ai rankings.
            }
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            
            if response.status_code != 200:
                error_data = response.text
                if response.status_code == 429:
                    error_data = "Rate Limit (429). This model is currently busy/overloaded. Try a different model (e.g. google/gemini-2.0-flash-001) or wait."
                raise Exception(f"OpenRouter: {error_data}")
                
            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                response_text = response_json['choices'][0]['message']['content']
            else:
                raise Exception("OpenRouter returned no content")
                
        else:
            # Google Gemini Call
            model_name = st.session_state.get('gemini_model', 'gemini-1.5-flash')
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            response_text = response.text
        
        # Extract JSON from response
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            with st.expander("🔴 Analysis Error (No JSON found)"):
                st.error("The AI did not return a valid JSON object.")
                st.text("Raw Response:")
                st.code(response_text)
                
            return {
                "match_score": 0,
                "reasoning": "Error parsing AI response (No JSON found)",
                "missing_keywords": [],
                "hr_email": None,
                "email_source": None,
                "hr_linkedin": None,
                "cover_letter": "Error generating cover letter - No JSON",
                "cold_email": "Error generating email - No JSON"
            }
            
    except json.JSONDecodeError as e:
        with st.expander("🔴 JSON Error (Click to debug)"):
            st.error(f"JSON Parsing failed: {str(e)}")
            st.text("Raw Response content:")
            st.code(response_text)
            
        return {
            "match_score": 0,
            "reasoning": "JSON parsing error. See debug details above.",
            "missing_keywords": [],
            "hr_email": None,
            "email_source": None,
            "hr_linkedin": None,
            "cover_letter": "Error generating cover letter",
            "cold_email": "Error generating email"
        }
    except Exception as e:
        with st.expander("🔴 AI Error (Click to debug)"):
            st.error(f"Analysis failed: {str(e)}")
            if 'response_text' in locals():
                st.text("Raw Response content:")
                st.code(response_text)
                
        return {
            "match_score": 0,
            "reasoning": f"AI error: {str(e)}",
            "missing_keywords": [],
            "hr_email": None,
            "email_source": None,
            "hr_linkedin": None,
            "cover_letter": "Error generating cover letter",
            "cold_email": "Error generating email"
        }


def get_score_badge(score: int) -> str:
    """Return HTML badge based on score."""
    if score >= 80:
        return f'<span class="score-high">🎯 {score}% Match</span>'
    elif score >= 50:
        return f'<span class="score-medium">⚡ {score}% Match</span>'
    else:
        return f'<span class="score-low">📊 {score}% Match</span>'


def export_to_csv(analyzed_jobs: list) -> bytes:
    """Export analyzed jobs to CSV format."""
    export_data = []
    for job in analyzed_jobs:
        analysis = job.get('analysis', {})
        export_data.append({
            'Title': job.get('title', ''),
            'Company': job.get('company', ''),
            'Location': job.get('location', ''),
            'Match Score': analysis.get('match_score', 0),
            'Reasoning': analysis.get('reasoning', ''),
            'HR Email': analysis.get('hr_email', '') or 'N/A',
            'HR LinkedIn': analysis.get('hr_linkedin', '') or 'N/A',
            'Missing Keywords': ', '.join(analysis.get('missing_keywords', [])),
            'Job URL': job.get('job_url', ''),
            'Cover Letter': analysis.get('cover_letter', ''),
            'Cold Email': analysis.get('cold_email', '')
        })
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/find-matching-job.png", width=80)
    st.title("⚙️ Configuration")
    
    # API Key Section
    # API Key Section
    st.subheader("🔑 AI Configuration")
    
    provider = st.selectbox("Select AI Provider", ["Google Gemini", "OpenRouter"], index=0)
    st.session_state.provider = provider
    
    if provider == "Google Gemini":
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your free API key from https://aistudio.google.com/"
        )
        
        if api_key:
            if configure_gemini(api_key):
                st.success("✅ Gemini configured!")
                if st.button("🧪 Test API Key"):
                    with st.spinner("Testing API key..."):
                        if test_gemini_api_key():
                            st.success("✅ API key is valid and working!")
                        else:
                            st.error("❌ API key test failed. Please check your key.")
                            st.info("💡 Make sure you copied the full key from Google AI Studio")
    
    else: # OpenRouter
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your key from https://openrouter.ai/"
        )
        model = st.text_input("OpenRouter Model", value="openai/gpt-4o-mini", help="e.g. anthropic/claude-3-haiku")
        st.session_state.openrouter_model = model
        
        if api_key:
            st.session_state.gemini_configured = True # Reuse flag for general AI readiness
            st.session_state.api_key = api_key
            st.success("✅ OpenRouter Key set!")
    
    st.divider()
    
    # Resume Upload Section
    st.subheader("📄 Resume")
    uploaded_resume = st.file_uploader(
        "Upload your resume (PDF)",
        type=['pdf'],
        help="Upload your resume to enable job matching"
    )
    
    if uploaded_resume:
        if st.button("📖 Parse Resume"):
            resume_text = extract_resume_text(uploaded_resume)
            if resume_text:
                st.session_state.resume_text = resume_text
                st.success(f"✅ Resume parsed! ({len(resume_text)} characters)")
    
    if st.session_state.resume_text:
        with st.expander("👀 Preview Resume Text"):
            st.text(st.session_state.resume_text[:500] + "...")
    
    st.divider()
    
    # Job Search Section
    st.subheader("🔍 Job Search")
    
    job_role = st.text_input(
        "Job Role/Title",
        value="Software Engineer",
        placeholder="e.g., Product Manager, Data Scientist"
    )
    
    job_location = st.text_input(
        "Location",
        value="Remote",
        placeholder="e.g., New York, San Francisco, Remote"
    )
    
    num_results = st.slider(
        "Number of Results",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    # Filters
    st.markdown("### 🎯 Filters")
    col1, col2 = st.columns(2)
    with col1:
        is_remote = st.checkbox("🏠 Remote", value=False)
    with col2:
        is_hybrid = st.checkbox("🏢 Hybrid", value=False)
        
    experience_level = st.selectbox(
        "Experience Level",
        options=["Any", "Internship", "Entry Level", "Mid Level", "Senior", "Executive"],
        index=0
    )


    
    # (JobSpy doesn't natively support easy experience filtering, so we trust query or post-filter if we had the data)
    # st.info("Note: Experience filters are applied to the search query")

    job_sites = st.multiselect(
        "Job Sites",
        options=["LinkedIn", "Indeed", "Glassdoor", "ZipRecruiter", "Naukri"],
        default=["LinkedIn", "Indeed"],
        help="Select which job boards to search"
    )
    
    search_button = st.button("🚀 Search Jobs", use_container_width=True, type="primary")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 JobHuntAI - Free Edition</h1>
    <p>Your AI-powered personal recruiter • Scrape jobs • Match your resume • Generate applications</p>
</div>
""", unsafe_allow_html=True)

# Status indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    gemini_status = "✅ Ready" if st.session_state.gemini_configured else "❌ Not configured"
    st.metric("Gemini AI", gemini_status)

with col2:
    resume_status = "✅ Loaded" if st.session_state.resume_text else "❌ Not uploaded"
    st.metric("Resume", resume_status)

with col3:
    jobs_count = len(st.session_state.jobs_df) if st.session_state.jobs_df is not None else 0
    st.metric("Jobs Found", jobs_count)

with col4:
    analyzed_count = len(st.session_state.analyzed_jobs)
    st.metric("Jobs Analyzed", analyzed_count)

st.divider()

# Handle job search
if search_button:
    if not job_sites:
        st.error("Please select at least one job site.")
    else:
        # Reset offset for new search
        st.session_state.offset = 0
        
        st.session_state.jobs_df = scrape_job_listings(
            role=job_role,
            location=job_location,
            sites=job_sites,
            num_results=num_results,
            experience=experience_level,
            job_type=None,
            is_remote=is_remote,
            is_hybrid=is_hybrid,
            offset=0
        )
        st.session_state.analyzed_jobs = []  # Reset analyzed jobs

# Display scraped jobs
if st.session_state.jobs_df is not None and len(st.session_state.jobs_df) > 0:
    
    st.subheader("📋 Scraped Jobs")
    
    # History Expander
    if st.session_state.get('job_history'):
        with st.expander("📚 View Archived Jobs (History)", expanded=False):
            for i, batch in enumerate(reversed(st.session_state.job_history)):
                batch_num = len(st.session_state.job_history) - i
                st.markdown(f"### 🕰️ Batch {batch_num} ({len(batch['jobs'])} jobs)")
                st.dataframe(batch['jobs'], use_container_width=True, hide_index=True)
                if batch.get('analysis'):
                    st.success(f"Contains {len(batch['analysis'])} analyzed matches")
                st.divider()
    
    # Display dataframe with selected columns
    display_columns = ['title', 'company', 'hr_contact', 'location', 'site', 'job_url', 'job_type']
    available_columns = [col for col in display_columns if col in st.session_state.jobs_df.columns]
    
    st.dataframe(
        st.session_state.jobs_df[available_columns],
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Load More Jobs (Batching)
    if st.button("🔄 Load More Jobs (Archive Current & Fetch New)", help="Move current jobs to history and fetch next batch"):
        # 1. Archive Current
        current_batch = {
            'jobs': st.session_state.jobs_df,
            'analysis': st.session_state.analyzed_jobs
        }
        st.session_state.job_history.append(current_batch)
        
        # 2. Fetch New
        new_offset = st.session_state.offset + num_results
        
        with st.spinner(f"Fetching next batch (offset: {new_offset})..."):
            new_jobs = scrape_job_listings(
                role=job_role,
                location=job_location,
                sites=job_sites,
                num_results=num_results,
                experience=experience_level,
                job_type=None,
                is_remote=is_remote,
                is_hybrid=is_hybrid,
                offset=new_offset
            )
            
        if new_jobs is not None and not new_jobs.empty:
            # Check global deduplication against history if desired, but user asked for "new set of jobs". 
            # We'll just set it as active.
            st.session_state.jobs_df = new_jobs
            st.session_state.analyzed_jobs = [] # Clear analysis for new batch
            st.session_state.offset = new_offset
            st.success(f"✅ Loaded {len(new_jobs)} new jobs! Previous batch archived.")
            st.rerun()
        else:
            st.warning("⚠️ No more unique jobs found in this batch.")

    st.divider()
    
    # AI Analysis Section
    st.subheader("🤖 AI Analysis")
    
    if not st.session_state.gemini_configured:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar to enable AI analysis.")
    elif not st.session_state.resume_text:
        st.warning("⚠️ Please upload and parse your resume in the sidebar to enable AI analysis.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            total_jobs = len(st.session_state.jobs_df)
            num_to_analyze = st.number_input(
                "Jobs to analyze",
                min_value=1,
                max_value=total_jobs,
                value=total_jobs # Default to ALL jobs
            )
        
        with col2:
            st.info(f"💡 Ready to analyze {num_to_analyze} jobs (1 API call per job).")
        
        if st.button("🎯 Analyze & Match Jobs", use_container_width=True, type="primary"):
            st.session_state.analyzed_jobs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            jobs_to_analyze = st.session_state.jobs_df.head(num_to_analyze).to_dict('records')
            
            for idx, job in enumerate(jobs_to_analyze):
                status_text.text(f"Analyzing job {idx + 1}/{num_to_analyze}: {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}")
                
                analysis = analyze_job_with_gemini(st.session_state.resume_text, job)
                
                analyzed_job = {
                    **job,
                    'analysis': analysis
                }
                st.session_state.analyzed_jobs.append(analyzed_job)
                
                progress_bar.progress((idx + 1) / num_to_analyze)
            
            status_text.text("✅ Analysis complete!")
            st.rerun()

# Display analyzed jobs
if st.session_state.analyzed_jobs:
    st.divider()
    st.subheader("🎯 Analysis Results")
    
    # Sort by match score
    sorted_jobs = sorted(
        st.session_state.analyzed_jobs, 
        key=lambda x: x.get('analysis', {}).get('match_score', 0),
        reverse=True
    )
    
    # Export button
    csv_data = export_to_csv(sorted_jobs)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="jobhuntai_results.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.divider()
    
    # Display each analyzed job as a card
    for idx, job in enumerate(sorted_jobs):
        analysis = job.get('analysis', {})
        score = analysis.get('match_score', 0)
        
        # Determine card color based on score
        if score >= 80:
            border_color = "#38ef7d"
        elif score >= 50:
            border_color = "#f5576c"
        else:
            border_color = "#eb3349"
        
        with st.container():
            # Job header
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {job.get('title', 'Unknown Title')}")
                st.markdown(f"**🏢 {job.get('company', 'Unknown')}** • 📍 {job.get('location', 'Unknown')}")
                
                # Show extracted HR Email/LinkedIn immediately if found
                hr_email = analysis.get('hr_email')
                hr_linkedin = analysis.get('hr_linkedin')
                email_source = analysis.get('email_source')
                
                if hr_email and hr_email != "None":
                    source_badge = f" ({email_source})" if email_source else ""
                    st.markdown(f"📧 **HR Email:** `{hr_email}` {source_badge}")
                
                if hr_linkedin and hr_linkedin != "None":
                    st.markdown(f"🔗 **HR LinkedIn:** [{hr_linkedin}]({hr_linkedin})")
            
            with col2:
                st.markdown(get_score_badge(score), unsafe_allow_html=True)
            
            # Reasoning
            st.markdown(f"**Why:** {analysis.get('reasoning', 'N/A')}")
            
            # Missing keywords
            missing = analysis.get('missing_keywords', [])
            if missing:
                st.markdown(f"**Missing Skills:** {', '.join(missing)}")
            
            # Job URL
            job_url = job.get('job_url', '')
            if job_url:
                st.markdown(f"[🔗 View Job Posting]({job_url})")
            
            # Expandable assets section
            with st.expander("📝 Generated Application Assets"):
                tab1, tab2 = st.tabs(["Cover Letter", "Cold Email"])
                
                with tab1:
                    cover_letter = analysis.get('cover_letter', 'Not generated')
                    st.text_area(
                        "Cover Letter",
                        value=cover_letter,
                        height=250,
                        key=f"cover_{idx}_{job.get('title', '')[:10]}"
                    )
                    st.button(
                        "📋 Copy Cover Letter",
                        key=f"copy_cover_{idx}_{job.get('title', '')[:10]}",
                        help="Click to copy to clipboard"
                    )
                
                with tab2:
                    cold_email = analysis.get('cold_email', 'Not generated')
                    st.text_area(
                        "Cold Email",
                        value=cold_email,
                        height=150,
                        key=f"email_{idx}_{job.get('title', '')[:10]}"
                    )
                    st.button(
                        "📋 Copy Email",
                        key=f"copy_email_{idx}_{job.get('title', '')[:10]}",
                        help="Click to copy to clipboard"
                    )
            
            st.divider()



# Empty state
if st.session_state.jobs_df is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
        <h2>👋 Welcome to JobHuntAI!</h2>
        <p style="font-size: 1.2rem; color: #666;">Get started in 3 easy steps:</p>
        <ol style="text-align: left; max-width: 400px; margin: 0 auto; font-size: 1.1rem;">
            <li>🔑 Enter your <b>Gemini API key</b> in the sidebar</li>
            <li>📄 Upload your <b>resume</b> (PDF)</li>
            <li>🔍 Set your job preferences and click <b>Search Jobs</b></li>
        </ol>
        <p style="margin-top: 2rem; color: #888;">
            <a href="https://aistudio.google.com/" target="_blank">
                🆓 Get your free Gemini API key here
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #888; margin-top: 2rem;">
    <p>Made with ❤️ using Streamlit, python-jobspy & Google Gemini</p>
    <p style="font-size: 0.8rem;">100% Free • No Paid APIs • Open Source</p>
</div>
""", unsafe_allow_html=True)

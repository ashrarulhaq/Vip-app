"""
JobHuntAI - Free Edition
A personal recruiter agent that scrapes jobs, matches them to your resume, 
and generates tailored application materials using AI.
"""

import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO

# Import job scraping library
from jobspy import scrape_jobs

# Import PDF parsing
from pypdf import PdfReader

# Import Google Gemini AI
import google.generativeai as genai


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
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False


def scrape_job_listings(role: str, location: str, sites: list, num_results: int) -> pd.DataFrame:
    """Scrape jobs from selected sites using python-jobspy."""
    try:
        # Map site names to jobspy format
        site_mapping = {
            "LinkedIn": "linkedin",
            "Indeed": "indeed", 
            "Glassdoor": "glassdoor",
            "ZipRecruiter": "zip_recruiter"
        }
        
        selected_sites = [site_mapping[s] for s in sites if s in site_mapping]
        
        if not selected_sites:
            st.warning("Please select at least one job site.")
            return None
        
        with st.spinner(f"🔍 Scraping jobs from {', '.join(sites)}..."):
            jobs = scrape_jobs(
                site_name=selected_sites,
                search_term=role,
                location=location,
                results_wanted=num_results,
                hours_old=72,  # Jobs from last 3 days
                country_indeed='USA'  # Default to USA for Indeed
            )
            
        if jobs is not None and len(jobs) > 0:
            st.success(f"✅ Found {len(jobs)} jobs!")
            return jobs
        else:
            st.warning("No jobs found. Try different search terms or locations.")
            return None
            
    except Exception as e:
        st.error(f"Error scraping jobs: {str(e)}")
        st.info("💡 Tip: Some sites may block requests. Try selecting different sites or waiting a few minutes.")
        return None


def analyze_job_with_gemini(resume_text: str, job_data: dict) -> dict:
    """Analyze a single job against the resume using Gemini AI."""
    
    # Prepare job description
    job_description = f"""
    Title: {job_data.get('title', 'N/A')}
    Company: {job_data.get('company', 'N/A')}
    Location: {job_data.get('location', 'N/A')}
    Description: {job_data.get('description', 'N/A')[:3000]}
    """
    
    prompt = f"""You are an expert career coach and ATS (Applicant Tracking System) analyzer.

RESUME:
{resume_text[:5000]}

JOB POSTING:
{job_description}

TASK:
Analyze how well this candidate's resume matches the job posting. Be critical and honest.

Provide your analysis in the following JSON format (and ONLY JSON, no other text):
{{
    "match_score": <integer 0-100>,
    "reasoning": "<one sentence explaining the match/mismatch>",
    "missing_keywords": ["<keyword1>", "<keyword2>"],
    "cover_letter": "<professional cover letter under 200 words that connects specific resume achievements to job requirements>",
    "cold_email": "<short, punchy email to hiring manager under 100 words>"
}}

Remember:
- Be critical. Don't give high scores just for keyword matches.
- Look for actual experience alignment.
- The cover letter should NOT repeat the resume - it should connect achievements to job requirements.
- The cold email should be conversational and direct."""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
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
                "cover_letter": "Error generating cover letter",
                "cold_email": "Error generating email"
            }
            
    except json.JSONDecodeError as e:
        return {
            "match_score": 0,
            "reasoning": f"JSON parsing error: {str(e)}",
            "missing_keywords": [],
            "cover_letter": "Error generating cover letter",
            "cold_email": "Error generating email"
        }
    except Exception as e:
        return {
            "match_score": 0,
            "reasoning": f"AI error: {str(e)}",
            "missing_keywords": [],
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
        export_data.append({
            'Title': job.get('title', ''),
            'Company': job.get('company', ''),
            'Location': job.get('location', ''),
            'Match Score': job.get('analysis', {}).get('match_score', 0),
            'Reasoning': job.get('analysis', {}).get('reasoning', ''),
            'Missing Keywords': ', '.join(job.get('analysis', {}).get('missing_keywords', [])),
            'Job URL': job.get('job_url', ''),
            'Cover Letter': job.get('analysis', {}).get('cover_letter', ''),
            'Cold Email': job.get('analysis', {}).get('cold_email', '')
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
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your free API key from https://aistudio.google.com/"
    )
    
    if api_key:
        if configure_gemini(api_key):
            st.success("✅ Gemini configured!")
    
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
    
    job_sites = st.multiselect(
        "Job Sites",
        options=["LinkedIn", "Indeed", "Glassdoor", "ZipRecruiter"],
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
        st.session_state.jobs_df = scrape_job_listings(
            role=job_role,
            location=job_location,
            sites=job_sites,
            num_results=num_results
        )
        st.session_state.analyzed_jobs = []  # Reset analyzed jobs

# Display scraped jobs
if st.session_state.jobs_df is not None and len(st.session_state.jobs_df) > 0:
    
    st.subheader("📋 Scraped Jobs")
    
    # Display dataframe with selected columns
    display_columns = ['title', 'company', 'location', 'site', 'job_url']
    available_columns = [col for col in display_columns if col in st.session_state.jobs_df.columns]
    
    st.dataframe(
        st.session_state.jobs_df[available_columns],
        use_container_width=True,
        hide_index=True
    )
    
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
            num_to_analyze = st.number_input(
                "Jobs to analyze",
                min_value=1,
                max_value=min(10, len(st.session_state.jobs_df)),
                value=min(5, len(st.session_state.jobs_df))
            )
        
        with col2:
            st.info(f"💡 Analyzing {num_to_analyze} jobs will make {num_to_analyze} API calls to Gemini.")
        
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
    for job in sorted_jobs:
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
                        key=f"cover_{job.get('title', '')}_{job.get('company', '')}"
                    )
                    st.button(
                        "📋 Copy Cover Letter",
                        key=f"copy_cover_{job.get('title', '')}_{job.get('company', '')}",
                        help="Click to copy to clipboard"
                    )
                
                with tab2:
                    cold_email = analysis.get('cold_email', 'Not generated')
                    st.text_area(
                        "Cold Email",
                        value=cold_email,
                        height=150,
                        key=f"email_{job.get('title', '')}_{job.get('company', '')}"
                    )
                    st.button(
                        "📋 Copy Email",
                        key=f"copy_email_{job.get('title', '')}_{job.get('company', '')}",
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

# JobHuntAI - Free Edition 🎯

A free, AI-powered job hunting assistant that scrapes jobs from multiple job boards, matches them to your resume, and generates tailored application materials.

## Features

- 🔍 **Multi-Site Scraping** - LinkedIn, Indeed, Glassdoor, ZipRecruiter
- 📄 **Resume Parsing** - Upload PDF and extract skills/experience
- 🤖 **AI Matching** - Smart job-resume match scoring using Google Gemini
- 📝 **Asset Generation** - Tailored cover letters and cold emails
- 📥 **CSV Export** - Download results for tracking

## Quick Start

### 1. Get API Key
Get your free Gemini API key at [Google AI Studio](https://aistudio.google.com/)

### 2. Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Use the App
1. Enter your Gemini API key
2. Upload your resume (PDF)
3. Set job search parameters
4. Click "Search Jobs" → "Analyze & Match Jobs"

## Cost

**$0.00** - Uses only free/open-source tools:
- `python-jobspy` - Free job scraping
- `google-generativeai` - Gemini Flash free tier
- `streamlit` - Open source framework

## Tech Stack

- Python 3.10+
- Streamlit
- python-jobspy
- Google Gemini 1.5 Flash
- pypdf
- pandas

## License

MIT

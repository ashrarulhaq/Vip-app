// State Management
const state = {
    apiKey: '',
    provider: 'gemini', // 'gemini' or 'openrouter'
    openRouterModel: 'google/gemini-2.0-flash-exp:free',
    resumeText: '',
    jobs: []
};

// API Base URL (auto-detects if local or production)
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000/api'
    : '/api';

// DOM Elements
const dom = {
    provider: document.getElementById('provider'),
    apiKey: document.getElementById('api-key'),
    modelGroup: document.getElementById('model-group'),
    model: document.getElementById('model'),
    apiStatus: document.getElementById('api-status'),

    resumeMethod: document.getElementsByName('resume-method'),
    uploadSection: document.getElementById('upload-section'),
    pasteSection: document.getElementById('paste-section'),
    resumeFile: document.getElementById('resume-file'),
    parseResumeBtn: document.getElementById('parse-resume-btn'),
    resumeText: document.getElementById('resume-text'),
    saveResumeBtn: document.getElementById('save-resume-btn'),
    resumeStatus: document.getElementById('resume-status'),

    jobTitle: document.getElementById('job-title'),
    location: document.getElementById('location'),
    resultsCount: document.getElementById('results-count'),
    resultsCountValue: document.getElementById('results-count-value'),
    isRemote: document.getElementById('is-remote'),
    searchBtn: document.getElementById('search-btn'),
    analyzeAllBtn: document.getElementById('analyze-all-btn'),

    welcomeScreen: document.getElementById('welcome-screen'),
    jobsContainer: document.getElementById('jobs-container'),
    jobsTableBody: document.getElementById('jobs-table-body'),
    loading: document.getElementById('loading'),
    loadingText: document.getElementById('loading-text'),

    statAi: document.getElementById('stat-ai'),
    statResume: document.getElementById('stat-resume'),
    statJobs: document.getElementById('stat-jobs'),
    statScore: document.getElementById('stat-score')
};

// ============================================================================
// INITIALIZATION & EVENTS
// ============================================================================

function init() {
    // Provider toggle
    dom.provider.addEventListener('change', (e) => {
        state.provider = e.target.value;
        dom.modelGroup.style.display = state.provider === 'openrouter' ? 'block' : 'none';
        updateStats();
    });

    // API Key input
    dom.apiKey.addEventListener('input', (e) => {
        state.apiKey = e.target.value;
        updateStats();
    });

    // Resume method toggle
    dom.resumeMethod.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'upload') {
                dom.uploadSection.style.display = 'block';
                dom.pasteSection.style.display = 'none';
            } else {
                dom.uploadSection.style.display = 'none';
                dom.pasteSection.style.display = 'block';
            }
        });
    });

    // Parse PDF
    dom.parseResumeBtn.addEventListener('click', async () => {
        const file = dom.resumeFile.files[0];
        if (!file) return alert('Please select a PDF file first.');

        showLoading('Parsing resume...');
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/upload-resume`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            state.resumeText = data.text;
            showStatus(dom.resumeStatus, '✅ Resume parsed successfully!', 'success');
            updateStats();
        } catch (error) {
            showStatus(dom.resumeStatus, '❌ Failed to parse resume.', 'error');
            console.error(error);
        } finally {
            hideLoading();
        }
    });

    // Save pasted resume
    dom.saveResumeBtn.addEventListener('click', () => {
        const text = dom.resumeText.value.trim();
        if (text.length < 50) return alert('Resume text is too short.');

        state.resumeText = text;
        showStatus(dom.resumeStatus, '✅ Resume saved successfully!', 'success');
        updateStats();
    });

    // Results slider
    dom.resultsCount.addEventListener('input', (e) => {
        dom.resultsCountValue.textContent = e.target.value;
    });

    // Job Search
    dom.searchBtn.addEventListener('click', searchJobs);

    // Analyze All
    if (dom.analyzeAllBtn) {
        dom.analyzeAllBtn.addEventListener('click', analyzeAllJobs);
    }
}

// ============================================================================
// CORE FUNCTIONS
// ============================================================================

async function searchJobs() {
    if (!dom.jobTitle.value || !dom.location.value) {
        return alert('Please enter both Job Title and Location.');
    }

    showLoading('Scraping jobs... This may take 30-60 seconds.');
    dom.welcomeScreen.style.display = 'none';
    dom.jobsContainer.style.display = 'none';

    const payload = {
        job_title: dom.jobTitle.value,
        location: dom.location.value,
        results_wanted: parseInt(dom.resultsCount.value),
        is_remote: dom.isRemote.checked
    };

    try {
        const response = await fetch(`${API_BASE_URL}/search-jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            state.jobs = data.jobs;
            renderJobsTable(state.jobs);
            dom.jobsContainer.style.display = 'block';
            updateStats();
        } else {
            alert('Job search failed. Please try again.');
        }

    } catch (error) {
        alert('Error searching jobs: ' + error.message);
    } finally {
        hideLoading();
    }
}

function renderJobsTable(jobs) {
    const tbody = document.getElementById('jobs-table-body');
    tbody.innerHTML = '';

    if (jobs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 2rem;">No jobs found. Try different keywords.</td></tr>';
        return;
    }

    jobs.forEach((job, index) => {
        // Main Row
        const tr = document.createElement('tr');
        tr.id = `job-row-${index}`;
        tr.innerHTML = `
            <td>
                <span class="job-title">${job.title}</span>
                <a href="${job.job_url}" target="_blank" style="font-size:0.8rem; color:#60a5fa;">View Job 🔗</a>
            </td>
            <td>
                <span class="job-company">${job.company}</span>
                <div class="job-date" style="font-size:0.75rem">${job.location}</div>
            </td>
            <td>
                <span class="job-date">${job.date_posted || 'Recently'}</span>
            </td>
            <td id="hr-cell-${index}" class="hr-info">
                <span style="opacity:0.5; font-size: 0.8rem;">Click Analyze</span>
            </td>
            <td id="match-cell-${index}">
                <span style="opacity:0.5">-</span>
            </td>
            <td>
                <div class="action-buttons">
                    <button class="btn btn-primary" onclick="analyzeJob(${index})">🤖 Analyze</button>
                    <!-- Apply uses job_url directly -->
                </div>
            </td>
        `;
        tbody.appendChild(tr);

        // Details Row (Hidden by default)
        const trDetails = document.createElement('tr');
        trDetails.id = `details-row-${index}`;
        trDetails.style.display = 'none';
        trDetails.innerHTML = `
            <td colspan="6" style="padding: 0; border: none;">
                <div id="details-content-${index}" class="details-content" style="padding: 1.5rem; background: rgba(0,0,0,0.2); border-bottom: 1px solid var(--border-color);">
                    <!-- Analysis content goes here -->
                </div>
            </td>
        `;
        tbody.appendChild(trDetails);
    });
}

function toggleDetails(index) {
    const row = document.getElementById(`details-row-${index}`);
    if (row.style.display === 'none') {
        row.style.display = 'table-row';
    } else {
        row.style.display = 'none';
    }
}

async function analyzeAllJobs() {
    if (!state.resumeText) return alert('Please upload/paste your resume first.');
    if (!state.apiKey) return alert('Please enter your API Key first.');

    const btn = dom.analyzeAllBtn;
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing All...';

    for (let i = 0; i < state.jobs.length; i++) {
        try {
            await analyzeJob(i, true); // true = silent
        } catch (e) {
            console.error(`Failed to analyze job ${i}`, e);
        }
    }

    btn.textContent = '✅ Analysis Complete';
    setTimeout(() => {
        btn.disabled = false;
        btn.textContent = originalText;
    }, 3000);
}

function calculateOverallScore() {
    const badges = document.querySelectorAll('.score-badge');
    let sum = 0;
    let count = 0;

    badges.forEach(b => {
        const val = parseInt(b.textContent);
        if (!isNaN(val)) {
            sum += val;
            count++;
        }
    });

    if (count > 0 && dom.statScore) {
        const avg = Math.round(sum / count);
        dom.statScore.textContent = `${avg}%`;

        if (avg >= 75) dom.statScore.style.color = 'var(--success)';
        else if (avg >= 50) dom.statScore.style.color = 'var(--warning)';
        else dom.statScore.style.color = 'var(--error)';
    }
}

window.analyzeJob = async function (index, silent = false) {
    if (!state.resumeText) {
        if (!silent) alert('Please upload/paste your resume first.');
        return;
    }
    if (!state.apiKey) {
        if (!silent) alert('Please enter your API Key first.');
        return;
    }

    const job = state.jobs[index];
    const btn = document.querySelector(`button[onclick="analyzeJob(${index})"]`);

    if (btn) {
        btn.textContent = '⏳ ...';
        btn.disabled = true;
    }

    try {
        const payload = {
            job: job,
            resume_text: state.resumeText,
            provider: state.provider,
            api_key: state.apiKey,
            model: state.provider === 'openrouter' ? dom.model.value : undefined
        };

        const response = await fetch(`${API_BASE_URL}/analyze-job`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            updateRowWithAnalysis(index, data.analysis);
            // Open details row automatically
            const detailsRow = document.getElementById(`details-row-${index}`);
            if (detailsRow) detailsRow.style.display = 'table-row';

            calculateOverallScore();
            return data.analysis.match_score;
        } else {
            if (!silent) alert('Analysis failed: ' + (data.error || 'Unknown error'));
        }

    } catch (error) {
        if (!silent) alert('Analysis error: ' + error.message);
    } finally {
        if (btn) {
            btn.textContent = '✅ Done';
            // Change button to "View" if analysis exists? 
            // Better yet, just leave it as Done. Clicking again re-analyzes.
            btn.disabled = false;
        }
    }
};

function updateRowWithAnalysis(index, analysis) {
    // 1. Update Main Row Cells
    const hrCell = document.getElementById(`hr-cell-${index}`);
    let hrHtml = '';
    if (analysis.hr_email) hrHtml += `<div>📧 <a href="mailto:${analysis.hr_email}">${analysis.hr_email}</a></div>`;
    if (analysis.hr_linkedin) hrHtml += `<div>🔗 <a href="${analysis.hr_linkedin}" target="_blank">Link</a></div>`;
    if (!analysis.hr_email && !analysis.hr_linkedin) hrHtml = `<span style="opacity:0.5">-</span>`;
    hrCell.innerHTML = hrHtml;

    const matchCell = document.getElementById(`match-cell-${index}`);
    matchCell.innerHTML = `<span class="score-badge ${getScoreClass(analysis.match_score)}">${analysis.match_score}%</span>`;

    // 2. Populate Details Row
    const detailsContent = document.getElementById(`details-content-${index}`);
    detailsContent.innerHTML = `
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <h4 style="color:var(--accent-primary); margin-bottom:0.5rem;">🧠 Why: The candidate...</h4>
                <p style="font-size:0.9rem; color:var(--text-secondary); margin-bottom:1rem;">${analysis.reasoning}</p>
                
                <h4 style="color:var(--error); margin-bottom:0.5rem;">⚠️ Missing Skills</h4>
                <div class="keywords-list" style="margin-bottom:1rem;">
                    ${analysis.missing_keywords.map(k => `<span class="keyword-tag">${k}</span>`).join('')}
                </div>
            </div>
            
            <div style="flex: 1; min-width: 300px;">
                <h4 style="color:var(--accent-blue); margin-bottom:0.5rem;">📝 Generated Application Assets</h4>
                <div class="tabs" style="margin-bottom:0.5rem;">
                    <button class="tab-btn active" onclick="switchTab(${index}, 'cover')">Cover Letter</button>
                    <button class="tab-btn" onclick="switchTab(${index}, 'email')">Cold Email</button>
                    <button class="btn btn-secondary" style="margin-left:auto; padding:0.2rem 0.5rem; font-size:0.8rem;" onclick="copyToClipboard('data-cover-${index}')">📋 Copy</button>
                </div>
                
                <div id="tab-content-${index}" class="tab-content" style="padding:0;">
                    <textarea readonly style="width:100%; height:200px; padding:1rem; border:1px solid var(--border-color); border-radius:var(--radius); background:var(--bg-primary);">${analysis.cover_letter}</textarea>
                </div>
                
                <!-- Hidden storage -->
                <div id="data-cover-${index}" style="display:none">${analysis.cover_letter}</div>
                <div id="data-email-${index}" style="display:none">${analysis.cold_email}</div>
            </div>
        </div>
    `;
}

window.copyToClipboard = function (elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        alert('Copied to clipboard!');
    });
};

window.switchTab = function (index, type) {
    const contentBox = document.querySelector(`#tab-content-${index} textarea`);
    const dataBox = document.getElementById(`data-${type}-${index}`);
    const btns = document.querySelectorAll(`#details-content-${index} .tab-btn`);
    const copyBtn = document.querySelector(`#details-content-${index} .btn-secondary`); // The copy button

    contentBox.value = dataBox.textContent;

    btns.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Update copy target
    copyBtn.setAttribute('onclick', `copyToClipboard('data-${type}-${index}')`);
};

// ============================================================================
// UTILITIES
// ============================================================================

function updateStats() {
    dom.statAi.textContent = state.apiKey ? '✅ Configured' : '❌ Not configured';
    dom.statResume.textContent = state.resumeText ? '✅ Uploaded' : '❌ Not uploaded';
    dom.statJobs.textContent = state.jobs.length;
}

function showStatus(element, message, type) {
    element.textContent = message;
    element.className = `status-badge ${type}`;
    element.style.display = 'block';
    setTimeout(() => { element.style.display = 'none'; }, 5000);
}

function showLoading(text) {
    dom.loadingText.textContent = text;
    dom.loading.style.display = 'flex';
}

function hideLoading() {
    dom.loading.style.display = 'none';
}

function getScoreClass(score) {
    if (score >= 75) return 'score-high';
    if (score >= 50) return 'score-medium';
    return 'score-low';
}

// Start App
document.addEventListener('DOMContentLoaded', init);

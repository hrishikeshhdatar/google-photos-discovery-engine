import os
import glob
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & MATERIAL DESIGN 3 CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Google Photos Search Insights Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Material Design 3 Design System Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Text:wght@400;500&family=Roboto:wght@400;500;700&display=swap');

    /* Force Light Theme */
    :root {
        color-scheme: light !important;
    }

    /* Hide Default Streamlit Chrome */
    header[data-testid="stHeader"], footer, #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }

    /* Canvas & Global Typography */
    html, body, [class*="stApp"], .stApp {
        background-color: #FFFFFF !important;
        color: #202124 !important;
        font-family: 'Google Sans Text', 'Roboto', Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Main Container Max-Width & Spacing */
    .main .block-container {
        max-width: 1200px !important;
        padding-left: 32px !important;
        padding-right: 32px !important;
        padding-top: 32px !important;
        padding-bottom: 48px !important;
        margin: 0 auto !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        background-color: #F8F9FA !important;
        border-right: 1px solid #DADCE0 !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 24px 16px !important;
    }

    .sb-label {
        font-size: 12px !important;
        line-height: 16px !important;
        font-weight: 500 !important;
        color: #5F6368 !important;
        text-transform: none !important;
        margin-bottom: 12px !important;
    }

    .sb-metric-row {
        margin-bottom: 16px !important;
    }

    .sb-metric-label {
        font-size: 12px !important;
        line-height: 16px !important;
        color: #5F6368 !important;
        font-weight: 400 !important;
    }

    .sb-metric-value {
        font-family: 'Google Sans', sans-serif !important;
        font-size: 24px !important;
        line-height: 32px !important;
        font-weight: 500 !important;
        color: #202124 !important;
        margin-top: 4px !important;
    }

    .sb-divider {
        height: 1px !important;
        background-color: #DADCE0 !important;
        margin: 16px 0 !important;
        border: none !important;
    }

    /* Status Pill Chips */
    .status-pill-success {
        display: inline-flex !important;
        align-items: center !important;
        height: 28px !important;
        padding: 0 12px !important;
        border-radius: 14px !important;
        background-color: #E6F4EA !important;
        color: #137333 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    .status-pill-error {
        display: inline-flex !important;
        align-items: center !important;
        height: 28px !important;
        padding: 0 12px !important;
        border-radius: 14px !important;
        background-color: #FCE8E6 !important;
        color: #B3261F !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    .status-dot-success {
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        background-color: #137333 !important;
        margin-right: 8px !important;
        display: inline-block !important;
    }

    .status-dot-error {
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        background-color: #B3261F !important;
        margin-right: 8px !important;
        display: inline-block !important;
    }

    /* Main Header Styling */
    .md-header-title {
        font-family: 'Google Sans', sans-serif !important;
        font-size: 32px !important;
        line-height: 40px !important;
        font-weight: 400 !important;
        color: #202124 !important;
        margin: 0 !important;
    }

    .md-header-subtitle {
        font-size: 14px !important;
        line-height: 20px !important;
        color: #5F6368 !important;
        margin-top: 8px !important;
        margin-bottom: 24px !important;
    }

    .md-header-divider {
        height: 1px !important;
        background-color: #DADCE0 !important;
        border: none !important;
        margin-bottom: 24px !important;
    }

    /* Section Titles for Tabs */
    .md-section-title {
        font-family: 'Google Sans', sans-serif !important;
        font-size: 22px !important;
        line-height: 28px !important;
        font-weight: 400 !important;
        color: #202124 !important;
        margin-bottom: 8px !important;
    }

    .md-section-caption {
        font-size: 14px !important;
        line-height: 20px !important;
        color: #5F6368 !important;
        margin-bottom: 24px !important;
    }

    /* Material Design 3 Tabs Styling */
    div[data-testid="stTabs"] {
        margin-bottom: 24px !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 0px !important;
        border-bottom: 1px solid #DADCE0 !important;
        background-color: transparent !important;
        padding-bottom: 0px !important;
    }

    button[data-baseweb="tab"] {
        height: 48px !important;
        padding: 0 24px !important;
        font-family: 'Google Sans Text', 'Roboto', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #5F6368 !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        transition: background-color 150ms ease, color 150ms ease !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #202124 !important;
        background-color: #F1F3F4 !important;
    }

    button[aria-selected="true"] {
        color: #1A73E8 !important;
        font-weight: 500 !important;
        border-bottom: 3px solid #1A73E8 !important;
        border-top-left-radius: 4px !important;
        border-top-right-radius: 4px !important;
        background-color: transparent !important;
    }

    /* Standard MD3 Card Containers */
    .md-card {
        background-color: #F8F9FA !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: none !important;
        margin-bottom: 24px !important;
    }

    /* Streamlit Container/Expander Restyling */
    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stExpander"] {
        background-color: #F8F9FA !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
    }

    /* Filled Pill Buttons */
    div[data-testid="stButton"] > button {
        height: 40px !important;
        min-height: 40px !important;
        border-radius: 20px !important;
        background-color: #1A73E8 !important;
        color: #FFFFFF !important;
        font-family: 'Google Sans Text', 'Roboto', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 0 24px !important;
        border: none !important;
        box-shadow: none !important;
        transition: background-color 150ms ease, box-shadow 150ms ease !important;
        cursor: pointer !important;
        width: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #1765CC !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stButton"] > button:active {
        background-color: #185ABC !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stButton"] > button:focus-visible {
        outline: 2px solid #1A73E8 !important;
        outline-offset: 2px !important;
    }

    div[data-testid="stButton"] > button:disabled {
        background-color: #E8EAED !important;
        color: #9AA0A6 !important;
        cursor: not-allowed !important;
    }

    /* Inputs & Text Area */
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 8px !important;
        color: #202124 !important;
        font-family: 'Google Sans Text', 'Roboto', sans-serif !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
    }

    div[data-testid="stTextArea"] textarea:focus, div[data-testid="stTextInput"] input:focus {
        border: 2px solid #1A73E8 !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #202124 !important;
        margin-bottom: 8px !important;
    }

    .input-helper-text {
        font-size: 12px !important;
        line-height: 16px !important;
        color: #5F6368 !important;
        margin-top: 4px !important;
    }

    /* Dataframe Table Container Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #DADCE0 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        overflow: hidden !important;
    }

    /* Summary Markdown Output Card */
    .summary-output-card {
        background-color: #F8F9FA !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        max-width: 72ch !important;
        font-size: 14px !important;
        line-height: 22px !important;
        color: #202124 !important;
        margin-top: 24px !important;
    }

    .summary-output-card h1, .summary-output-card h2, .summary-output-card h3 {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #202124 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. SECURE API KEY RETRIEVAL (LOGIC UNCHANGED)
# -----------------------------------------------------------------------------
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY", "")

# -----------------------------------------------------------------------------
# 2. DATA INGESTION & HEURISTIC ENGINE (LOGIC UNCHANGED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_analyze_corpus():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        return pd.DataFrame()
        
    column_mapping = {
        'title': 'title', 'post_title': 'title', 'subject': 'title',
        'selftext': 'content', 'text': 'content', 'body': 'content', 
        'review_text': 'content', 'content': 'content',
        'rating': 'user_score', 'score': 'user_score', 'user_score': 'user_score',
        'upvotes': 'user_score', 'ups': 'user_score', 'stars': 'user_score',
        'likes': 'user_score', 'thumbs_up': 'user_score'
    }
    
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
            if not df.empty:
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.rename(columns=column_mapping)
                
                fname = file_path.lower()
                if 'reddit' in fname:
                    df['source_platform'] = 'Reddit Discussions'
                elif 'discussions' in fname or 'public' in fname or 'forum' in fname:
                    df['source_platform'] = 'Support Forums'
                else:
                    df['source_platform'] = 'App Store / Play Store'
                    
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    master_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    title_s = master_df.get('title', pd.Series(['']*len(master_df))).fillna('').astype(str)
    content_s = master_df.get('content', pd.Series(['']*len(master_df))).fillna('').astype(str)
    
    master_df['full_text'] = (title_s + " " + content_s).str.strip()
    master_df = master_df[master_df['full_text'].str.len() > 10].drop_duplicates(subset=['full_text'])

    # Taxonomy Tagging
    retrieval_kw = r'search|find|cant find|can\'t find|missing|lost|where|date|location|album|ocr|text|gemini|scroll|remember|face|people'
    master_df['is_retrieval_issue'] = master_df['full_text'].str.contains(retrieval_kw, case=False, na=False)

    taxonomies = {
        'Temporal / Milestone Ambiguity': r'date|year|month|timeline|old|years ago|timestamp|chronological',
        'Relational & Person Context': r'face|people|person|untagged|tag|child|baby|family|friend',
        'Document / OCR & Text Retrieval': r'ocr|text|document|receipt|screenshot|notes|paper|read',
        'Spatial & Event Context': r'location|place|city|trip|vacation|wedding|party|event|where',
        'Visual & Attribute Matching': r'color|dog|cat|car|shirt|object|thing|background|visual'
    }
    
    def tag_taxonomy(text):
        for category, pattern in taxonomies.items():
            if re.search(pattern, text, re.IGNORECASE):
                return category
        return 'General Retrieval Friction'

    master_df['problem_category'] = master_df['full_text'].apply(tag_taxonomy)

    # Search Formulation Strategy Classification
    def tag_search_strategy(text):
        if re.search(r'filename|\.jpg|\.png|file name|folder|album name', text, re.I):
            return 'Exact Metadata / Structured Search'
        elif re.search(r'ocr|text|read|receipt|screenshot|document|words', text, re.I):
            return 'OCR & Text Content Search'
        elif re.search(r'face|people|person|tag|mom|dad|baby|friend|family', text, re.I):
            return 'Relational & Person Search'
        elif re.search(r'date|year|month|old|time|ago|timeline|202|201', text, re.I):
            return 'Broad Temporal / Lifecycle Search'
        else:
            return 'Visual & Semantic Keyword Search'

    master_df['search_strategy'] = master_df['full_text'].apply(tag_search_strategy)

    # Memory Anchors
    master_df['remembered_anchor'] = master_df['full_text'].apply(
        lambda x: 'Event / Emotion / Visual Context' if re.search(r'wedding|trip|vacation|party|dog|happy|red|blue', x, re.I)
        else ('Person / Relational Context' if re.search(r'mom|dad|friend|baby|son|daughter|face', x, re.I)
        else 'General Salient Memory')
    )

    master_df['forgotten_anchor'] = master_df['full_text'].apply(
        lambda x: 'Exact Date / Year' if re.search(r'date|year|when|time|month', x, re.I)
        else ('Exact Folder / Album Name' if re.search(r'folder|album|where|location|path', x, re.I)
        else 'Exact Metadata / File String')
    )

    return master_df

df = load_and_analyze_corpus()

# -----------------------------------------------------------------------------
# PLOTLY HORIZONTAL BAR CHART HELPER (RESTYLED FOR MATERIAL 3)
# -----------------------------------------------------------------------------
def render_horizontal_bar_chart(series_data, x_label="Mentions", height=320):
    chart_df = series_data.reset_index()
    chart_df.columns = ['category', 'count']
    total_val = chart_df['count'].sum() if chart_df['count'].sum() > 0 else 1
    chart_df['share'] = (chart_df['count'] / total_val) * 100
    
    # Sort ascending so top value renders at top of y-axis in Plotly
    chart_df = chart_df.sort_values(by='count', ascending=True)
    chart_df['label_text'] = chart_df.apply(lambda r: f"{r['count']:,} · {r['share']:.1f}%", axis=1)

    # Top bar gets #1A73E8, remaining bars get #AECBFA
    colors = ['#AECBFA'] * len(chart_df)
    if len(colors) > 0:
        colors[-1] = '#1A73E8'

    fig = px.bar(
        chart_df,
        x='count',
        y='category',
        orientation='h',
        text='label_text'
    )
    
    fig.update_traces(
        marker_color=colors,
        marker_line_width=0,
        textposition='outside',
        textfont=dict(color='#202124', size=12, family='Roboto, sans-serif'),
        hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
    )
    
    fig.update_layout(
        margin=dict(l=220, r=80, t=10, b=30),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text=x_label, font=dict(color='#5F6368', size=12)),
            showgrid=False,
            zeroline=True,
            zerolinecolor='#DADCE0',
            zerolinewidth=1,
            tickfont=dict(color='#5F6368', size=11)
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            automargin=True,
            tickfont=dict(color='#202124', size=14, family='Google Sans Text, Roboto, sans-serif')
        )
    )
    return fig

# -----------------------------------------------------------------------------
# 3. HEADER & SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.markdown('''
<div style="margin-bottom: 24px;">
    <div class="md-header-title">Google Photos Search Insights Engine</div>
    <div class="md-header-subtitle">Analyzing user feedback at scale to understand photo retrieval friction and memory decay</div>
    <div class="md-header-divider"></div>
</div>
''', unsafe_allow_html=True)

if df.empty:
    st.error("No dataset found. Please ensure CSV feedback files are present in the directory.")
    st.stop()

retrieval_df = df[df['is_retrieval_issue'] == True]

# Sidebar Engine Status
with st.sidebar:
    st.markdown('''
    <div style="margin-bottom: 24px;">
        <div class="sb-label">Dataset</div>
        <div class="sb-metric-row">
            <div class="sb-metric-label">Total Ingested Posts</div>
            <div class="sb-metric-value">{total_posts:,}</div>
        </div>
        <div class="sb-divider"></div>
        <div class="sb-metric-row">
            <div class="sb-metric-label">Retrieval Issues Identified</div>
            <div class="sb-metric-value">{retrieval_issues:,}</div>
        </div>
        <div class="sb-divider"></div>
        <div class="sb-label">AI Status</div>
        {status_chip}
    </div>
    '''.format(
        total_posts=len(df),
        retrieval_issues=len(retrieval_df),
        status_chip='''<div class="status-pill-success"><span class="status-dot-success"></span>Gemini connected</div>''' if api_key else '''<div class="status-pill-error"><span class="status-dot-error"></span>Offline Mode</div>'''
    ), unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "AI Summary",
    "Priorities", 
    "Memory Patterns", 
    "Search Evidence",
    "All Data"
])

# -----------------------------------------------------------------------------
# TAB 1: AI SUMMARY (GEMINI AI SYNTHESIZER)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('''
    <div>
        <div class="md-section-title">AI Research Synthesizer</div>
        <div class="md-section-caption">Synthesize real user complaint logs into executive research findings using Gemini.</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    user_query = st.text_area(
        "Research Question:", 
        value="What kinds of old photos do users struggle to retrieve, and what information have they forgotten?",
        height=96
    )
    st.markdown('<div class="input-helper-text">Ask about retrieval issues, memory gaps, or opportunity areas.</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
    
    generate_btn = st.button("Generate Executive Summary")
    st.markdown('</div>', unsafe_allow_html=True)

    if generate_btn:
        if not api_key:
            st.warning("No Gemini API key found. Displaying standard summary baseline:")
            st.markdown('''
            <div class="summary-output-card">
            <h3>Executive Summary: User Retrieval Friction & Memory Cognitive Load</h3>
            
            <h4>1. Direct Answer</h4>
            <ul>
                <li><b>Primary Struggling Photo Types:</b> Screenshots, document scans/receipts, and milestone event photos from 3+ years ago.</li>
                <li><b>Search Formulation Behavior:</b> Users input natural language descriptions rather than structured metadata filters.</li>
            </ul>
            
            <h4>2. Memory Anchor Analysis</h4>
            <ul>
                <li><b>What Users Remember:</b> Salient visual anchors (e.g., <i>'red jacket'</i>, <i>'beach trip'</i>), broad timeframes, or people present.</li>
                <li><b>What Users Forget:</b> Precise timestamps, exact folder structures, or original file tags.</li>
            </ul>
            
            <h4>3. Strategic Opportunity</h4>
            <ul>
                <li>Fix core semantic search indexing failures.</li>
                <li>De-clutter AI recommendations to prioritize chronological retrieval.</li>
                <li>Improve device vs. cloud storage clarity.</li>
            </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                selected_samples = retrieval_df['full_text'].head(30).tolist()
                sample_text = "\n".join([f"- {text}" for text in selected_samples])
                
                prompt = f"""You are a Principal Product Manager for Google Photos.

TASK:
Synthesize the following real user feedback to answer this research question:
"{user_query}"

USER FEEDBACK CONTEXT:
{sample_text}

CRITICAL INSTRUCTION:
Do not include any scratchpad notes, bullet point analysis, planning text, or preamble. 
Start your response immediately with the header "# Executive Summary: User Retrieval Friction & Memory Cognitive Load".

REQUIRED REPORT STRUCTURE:
# Executive Summary: User Retrieval Friction & Memory Cognitive Load

### 1. Direct Answer & Retrieval Friction
Synthesize the primary categories of photos users struggle to retrieve, citing specific user quotes as evidence.

### 2. Memory Anchor Analysis
Provide a Markdown table comparing:
- What Users Remember (The Emotional / Intentional Anchor)
- What Users Forget (The Technical / Structural Gap)

### 3. Strategic Opportunity Pillars
Detail 3 actionable, high-impact product initiatives for Google Photos to solve these friction points.
"""
                with st.spinner("Generating summary..."):
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(prompt)
                    if res and res.text:
                        res_text = res.text
                        if "# Executive Summary" in res_text:
                            res_text = "# Executive Summary" + res_text.split("# Executive Summary", 1)[1]
                        
                        st.markdown(f'<div class="summary-output-card">', unsafe_allow_html=True)
                        st.markdown(res

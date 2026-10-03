import os
import glob
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Google Photos Search Insights Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (Inter font, sleek cards, refined tabs, clean metrics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8FAFC;
    }

    /* Backgrounds */
    .stApp {
        background-color: #0F172A;
    }
    
    /* Card Container Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 8px;
    }
    
    div[data-testid="stForm"] {
        border: 1px solid #334155;
        border-radius: 8px;
        background-color: #1E293B;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        padding: 10px 18px !important;
        border-radius: 6px 6px 0 0 !important;
        background-color: transparent !important;
        border: none !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background-color: rgba(255,255,255,0.03) !important;
    }
    button[aria-selected="true"] {
        color: #3B82F6 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    /* Dataframe Table Headers */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        margin-bottom: 12px;
    }

    /* Primary Buttons */
    div.stButton > button {
        background-color: #3B82F6;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.875rem;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        background-color: #2563EB;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Header typography */
    .app-header {
        margin-bottom: 24px;
        border-bottom: 1px solid #334155;
        padding-bottom: 16px;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.875rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .section-caption {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 12px;
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
                
                if 'reddit' in file_path.lower():
                    df['source_platform'] = 'Reddit Discussions'
                elif 'discussions' in file_path.lower() or 'public' in file_path.lower():
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
# PLOTLY HORIZONTAL BAR CHART HELPER (FIXED LABEL MARGINS & AUTO-MARGIN)
# -----------------------------------------------------------------------------
def render_horizontal_bar_chart(series_data, x_label="Mentions", height=320):
    chart_df = series_data.reset_index()
    chart_df.columns = ['category', 'count']
    total_val = chart_df['count'].sum()
    chart_df['share'] = (chart_df['count'] / total_val) * 100
    
    # Sort ascending so highest value is rendered at top of y-axis
    chart_df = chart_df.sort_values(by='count', ascending=True)

    fig = px.bar(
        chart_df,
        x='count',
        y='category',
        orientation='h',
        text='count',
        custom_data=['share']
    )
    
    fig.update_traces(
        marker_color='#3B82F6',
        marker_line_width=0,
        textposition='outside',
        texttemplate='%{x:,}',
        textfont=dict(color='#F8FAFC', size=12, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Count: %{x:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>'
    )
    
    fig.update_layout(
        margin=dict(l=220, r=50, t=10, b=30),  # Expanded left margin to prevent truncation
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text=x_label, font=dict(color='#94A3B8', size=12)),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.06)',
            zeroline=False,
            tickfont=dict(color='#94A3B8', size=11)
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            automargin=True,  # Automatically calculates padding for long y-axis titles
            tickfont=dict(color='#F8FAFC', size=12, family='Inter')
        )
    )
    return fig

# -----------------------------------------------------------------------------
# 3. HEADER & SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">Google Photos Search Insights Engine</div>
    <div class="app-subtitle">Analyzing user feedback at scale to understand photo retrieval friction and memory decay</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.error("No dataset found. Please ensure CSV feedback files are present in the directory.")
    st.stop()

# Sidebar Engine Status
retrieval_df = df[df['is_retrieval_issue'] == True]

with st.sidebar:
    st.markdown("<div class=\"section-title\">Dataset Metrics</div>", unsafe_allow_html=True)
    st.metric("Total Ingested Posts", f"{len(df):,}")
    st.metric("Retrieval Issues Identified", f"{len(retrieval_df):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-title\">AI Status</div>", unsafe_allow_html=True)
    if api_key:
        st.success("Gemini API Connected")
    else:
        st.warning("Offline Mode (Static Synthesis)")

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
    st.markdown("<div class=\"section-title\">AI Research Synthesizer</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-caption\">Query the full dataset to synthesize retrieval issues, memory gaps, and opportunity areas.</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        user_query = st.text_area(
            "Research Question:", 
            value="What kinds of old photos do users struggle to retrieve, and what information have they forgotten?",
            height=80
        )
        generate_btn = st.button("Generate Executive Summary")

    if generate_btn:
        if not api_key:
            st.warning("No Gemini API key found. Displaying standard summary baseline:")
            st.markdown("""
            ### Executive Summary: User Retrieval Friction & Memory Cognitive Load
            
            #### 1. Direct Answer
            * **Primary Struggling Photo Types:** Screenshots, document scans/receipts, and milestone event photos from 3+ years ago.
            * **Search Formulation Behavior:** Users input natural language descriptions rather than structured metadata filters.
            
            #### 2. Memory Anchor Analysis
            * **What Users Remember:** Salient visual anchors (e.g., *'red jacket'*, *'beach trip'*), broad timeframes, or people present.
            * **What Users Forget:** Precise timestamps, exact folder structures, or original file tags.
            
            #### 3. Strategic Opportunity
            * Fix core semantic search indexing failures.
            * De-clutter AI recommendations to prioritize chronological retrieval.
            * Improve device vs. cloud storage clarity.
            """)
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # Smart context retriever logic
                q_words = [w.lower() for w in user_query.split() if len(w) > 3]
                selected_samples = []
                
                for cat, group in retrieval_df.groupby('problem_category'):
                    if q_words:
                        pattern = '|'.join(q_words)
                        group['q_match'] = group['full_text'].str.contains(pattern, case=False, na=False)
                        sorted_group = group.sort_values(by=['q_match'], ascending=False)
                    else:
                        sorted_group = group
                    selected_samples.extend(sorted_group['full_text'].head(7).tolist())
                
                sample_text = "\n".join([f"- {text}" for text in selected_samples[:35]])
                
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
                with st.spinner("Synthesizing feedback across corpus..."):
                    candidate_models = []
                    try:
                        for m in genai.list_models():
                            if 'generateContent' in getattr(m, 'supported_generation_methods', []):
                                clean_name = m.name.replace('models/', '')
                                if 'gemma' not in clean_name.lower():
                                    candidate_models.append(clean_name)
                    except Exception:
                        pass

                    preferred_order = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
                    search_list = preferred_order + [m for m in candidate_models if m not in preferred_order]

                    res_text = None
                    used_model = None
                    last_error = None

                    for m_name in search_list:
                        try:
                            model = genai.GenerativeModel(m_name)
                            res = model.generate_content(prompt)
                            if res and hasattr(res, 'text') and res.text:
                                res_text = res.text
                                used_model = m_name
                                break
                        except Exception as e:
                            last_error = e
                            continue

                    if res_text:
                        if "# Executive Summary" in res_text:
                            res_text = "# Executive Summary" + res_text.split("# Executive Summary", 1)[1]
                        elif "Executive Summary:" in res_text:
                            res_text = "# Executive Summary:" + res_text.split("Executive Summary:", 1)[1]

                        st.caption(f"Synthesized via `{used_model}`")
                        with st.container(border=True):
                            st.markdown(res_text)
                    else:
                        st.error(f"Gemini API Error: {str(last_error)}")
            except Exception as e:
                st.error(f"Configuration Error: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: PRIORITIES (STACKED LAYOUT + SEARCH CATEGORY DEFINITIONS)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("<div class=\"section-title\">What's going wrong in search, and what to fix first</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-caption\">Comparing search friction volume against issue severity to prioritize fixes.</div>", unsafe_allow_html=True)
    
    category_counts = retrieval_df['problem_category'].value_counts()
    total_retrieval = len(retrieval_df)
    
    # Top KPI Cards
    top_issue = category_counts.index[0] if not category_counts.empty else "N/A"
    top_share = (category_counts.iloc[0] / total_retrieval * 100) if not category_counts.empty else 0
    
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Total Search Complaints", f"{total_retrieval:,}")
    with kpi2:
        st.metric("Top Search Friction Area", top_issue)
    with kpi3:
        st.metric("Top Area Share", f"{top_share:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Full-width Distribution Chart
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Search Issue Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-caption\">Total mentions per search friction category across user feedback.</div>", unsafe_allow_html=True)
        fig_cats = render_horizontal_bar_chart(category_counts, x_label="Mentions", height=320)
        st.plotly_chart(fig_cats, use_container_width=True)

    # 2. Search Issue Glossary Expander (DEFINITIONS ADDED)
    with st.expander("📖 Guide: What do these 6 Search Categories mean?", expanded=False):
        st.markdown("""
        * **Temporal / Milestone Ambiguity:** User searches that rely on approximate dates, timeframes, or life events (e.g., *"photos from 3 years ago"*, *"wedding 2019"*). Failure happens when timestamps are wrong or chronological indexing fails.
        * **General Retrieval Friction:** Broad, unclassified search failures where photos are missing, hidden, or unavailable despite normal scrolling and basic keywords.
        * **Relational & Person Context:** Queries targeting specific individuals, family members, or friends (e.g., *"baby photos"*, *"untagged faces"*). Failure happens when face-tagging or grouping breaks.
        * **Spatial & Event Context:** Searches based on locations, cities, or organized trips (e.g., *"trip to Japan"*, *"beach vacation"*). Failure happens when geotags or event clustering are inaccurate.
        * **Visual & Attribute Matching:** Keyword searches for specific objects, colors, or visual items (e.g., *"red shirt"*, *"dog"*, *"car"*). Failure occurs when computer vision model indexing misses key objects.
        * **Document / OCR & Text Retrieval:** Searches for embedded text inside screenshots, receipts, notes, or scanned documents. Failure occurs when Optical Character Recognition (OCR) fails to index image text.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Full-width Priority Table
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Fix-First Priority Score</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-caption\">Priority = volume share × frustration severity rating.</div>", unsafe_allow_html=True)
        
        opp_data = []
        for cat, group in retrieval_df.groupby('problem_category'):
            count = len(group)
            pct = (count / total_retrieval) * 100
            scores = pd.to_numeric(group['user_score'], errors='coerce').dropna()
            avg_score = scores.mean() if not scores.empty else 2.5
            
            friction_factor = 1.5
            if not scores.empty and avg_score <= 5.0:
                friction_factor = max(1.0, 5.0 - avg_score)
            
            raw_opp_score = pct * friction_factor
            opp_data.append({
                "Search Issue": cat,
                "Share": pct / 100.0,
                "Raw Score": raw_opp_score
            })
            
        opp_df = pd.DataFrame(opp_data)
        max_raw = opp_df['Raw Score'].max() if not opp_df.empty else 1
        opp_df['Priority'] = (opp_df['Raw Score'] / max_raw) * 100
        opp_df = opp_df.drop(columns=['Raw Score']).sort_values(by='Priority', ascending=False)
        
        st.dataframe(
            opp_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Search Issue": st.column_config.TextColumn("Search Issue", width="large"),
                "Share": st.column_config.ProgressColumn("Share of Issues", format="%.1f%%", min_value=0, max_value=1, width="medium"),
                "Priority": st.column_config.ProgressColumn("Fix Priority", format="%.0f / 100", min_value=0, max_value=100, width="medium")
            }
        )

    # 4. Side-by-Side Comparator
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Side-by-Side Issue Comparator</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-caption\">Compare user memory anchors and search behavior across two problem areas.</div>", unsafe_allow_html=True)
        
        cats = list(retrieval_df['problem_category'].unique())
        if len(cats) >= 2:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                cat_a = st.selectbox("Select Problem A:", options=cats, index=0)
            with comp_col2:
                cat_b = st.selectbox("Select Problem B:", options=cats, index=min(1, len(cats)-1))
                
            df_a = retrieval_df[retrieval_df['problem_category'] == cat_a]
            df_b = retrieval_df[retrieval_df['problem_category'] == cat_b]
            
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown(f"**{cat_a}**")
                    st.caption(f"Volume: {(len(df_a)/total_retrieval)*100:.1f}% ({len(df_a):,} posts)")
                    st.markdown(f"**Dominant Search:** {df_a['search_strategy'].mode()[0] if not df_a.empty else 'N/A'}")
                    st.markdown(f"**Main Memory Cue:** {df_a['remembered_anchor'].mode()[0] if not df_a.empty else 'N/A'}")
                    st.markdown(f"**Main Forgotten Detail:** {df_a['forgotten_anchor'].mode()[0] if not df_a.empty else 'N/A'}")
                    if not df_a.empty:
                        st.caption(f"\"...{df_a.iloc[0]['full_text'][:140]}...\"")
                    
            with c2:
                with st.container(border=True):
                    st.markdown(f"**{cat_b}**")
                    st.caption(f"Volume: {(len(df_b)/total_retrieval)*100:.1f}% ({len(df_b):,} posts)")
                    st.markdown(f"**Dominant Search:** {df_b['search_strategy'].mode()[0] if not df_b.empty else 'N/A'}")
                    st.markdown(f"**Main Memory Cue:** {df_b['remembered_anchor'].mode()[0] if not df_b.empty else 'N/A'}")
                    st.markdown(f"**Main Forgotten Detail:** {df_b['forgotten_anchor'].mode()[0] if not df_b.empty else 'N/A'}")
                    if not df_b.empty:
                        st.caption(f"\"...{df_b.iloc[0]['full_text'][:140]}...\"")

# -----------------------------------------------------------------------------
# TAB 3: MEMORY PATTERNS (STACKED VERTICAL TABLES — ZERO SCROLLING)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("<div class=\"section-title\">What people remember vs. what they forget</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-caption\">Mapping emotional and visual cues against lost technical metadata.</div>", unsafe_allow_html=True)
    
    # 1. Details People Remember (Full Width)
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Details people remember</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-caption\">Emotional, visual, and relational memory cues.</div>", unsafe_allow_html=True)
        
        rem_counts = retrieval_df['remembered_anchor'].value_counts().reset_index()
        rem_counts.columns = ['Memory Cue', 'Mentions']
        max_rem = int(rem_counts['Mentions'].max()) if not rem_counts.empty else 100
        
        st.dataframe(
            rem_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Memory Cue": st.column_config.TextColumn("Memory Cue", width="large"),
                "Mentions": st.column_config.ProgressColumn(
                    "Total Mentions", 
                    format="%d", 
                    min_value=0, 
                    max_value=max_rem,
                    width="medium"
                )
            }
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Details People Forget (Full Width)
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Details people forget</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-caption\">Technical metadata, exact dates, and folder structures.</div>", unsafe_allow_html=True)
        
        for_counts = retrieval_df['forgotten_anchor'].value_counts().reset_index()
        for_counts.columns = ['Forgotten Detail', 'Mentions']
        max_for = int(for_counts['Mentions'].max()) if not for_counts.empty else 100
        
        st.dataframe(
            for_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Forgotten Detail": st.column_config.TextColumn("Forgotten Detail", width="large"),
                "Mentions": st.column_config.ProgressColumn(
                    "Total Mentions", 
                    format="%d", 
                    min_value=0, 
                    max_value=max_for,
                    width="medium"
                )
            }
        )

# -----------------------------------------------------------------------------
# TAB 4: SEARCH EVIDENCE (EVIDENCE & QUERY FORMULATION)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("<div class=\"section-title\">How people search when memory fails</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-caption\">Analysis of search formulations and verbatim user feedback.</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Search Formulation Patterns</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-caption\">Frequency of search methods used by users attempting photo retrieval.</div>", unsafe_allow_html=True)
        
        strategy_counts = retrieval_df['search_strategy'].value_counts()
        fig_strat = render_horizontal_bar_chart(strategy_counts, x_label="Posts Using Strategy", height=280)
        st.plotly_chart(fig_strat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class=\"section-title\">Verbatim User Feedback</div>", unsafe_allow_html=True)
        
        selected_cat = st.selectbox("Filter by Search Issue Category:", options=retrieval_df['problem_category'].unique())
        filtered_ev = retrieval_df[retrieval_df['problem_category'] == selected_cat]
        total_count = len(filtered_ev)
        
        if total_count == 0:
            st.info("No entries found for this category.")
        else:
            display_limit = st.slider(
                "Display Limit:", 
                min_value=5, 
                max_value=max(5, total_count), 
                value=min(15, total_count), 
                step=5
            )
            
            st.caption(f"Showing **{min(display_limit, total_count)}** of **{total_count:,}** quotes")
            
            for idx, row in filtered_ev.head(display_limit).iterrows():
                score_val = row.get('user_score')
                source_platform = str(row.get('source_platform', 'Public Feedback'))
                
                # Format ratings/upvotes dynamically
                score_tag = ""
                if pd.notna(score_val):
                    try:
                        num_val = float(score_val)
                        if 'App Store' in source_platform or 'Play Store' in source_platform:
                            score_tag = f" • Rating: {num_val:.1f}/5.0"
                        elif 'Reddit' in source_platform:
                            score_tag = f" • Upvotes: {int(num_val)}"
                        else:
                            score_tag = f" • Score: {num_val:.1f}"
                    except (ValueError, TypeError):
                        score_tag = ""

                expander_title = f"{source_platform}{score_tag} | Strategy: {row.get('search_strategy', 'General')}"
                
                with st.expander(expander_title):
                    st.write(f"\"{row['full_text']}\"")

# -----------------------------------------------------------------------------
# TAB 5: ALL DATA (RAW CORPUS EXPLORER)
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("<div class=\"section-title\">Unified Feedback Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-caption\">Filter and explore all ingested user feedback across public forums and app reviews.</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        search_q = st.text_input("Filter posts by keyword:", placeholder="e.g. receipt, wedding, folder, face tag")
        
        disp = df.copy()
        if search_q:
            disp = disp[disp['full_text'].str.contains(search_q, case=False, na=False)]
            
        # Display layer column renaming
        disp_table = disp[['full_text', 'problem_category', 'search_strategy', 'source_platform']].copy()
        disp_table.columns = ['User Feedback Text', 'Search Issue Category', 'Search Strategy', 'Source Platform']
        
        st.dataframe(
            disp_table.head(200),
            use_container_width=True,
            hide_index=True,
            column_config={
                "User Feedback Text": st.column_config.TextColumn("User Feedback Text", width="large"),
                "Search Issue Category": st.column_config.TextColumn("Search Issue Category", width="medium"),
                "Search Strategy": st.column_config.TextColumn("Search Strategy", width="medium"),
                "Source Platform": st.column_config.TextColumn("Source Platform", width="small")
            }
        )

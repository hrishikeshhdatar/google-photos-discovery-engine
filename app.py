import os
import glob
import re
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Google Photos AI Discovery Engine",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. SECURE API KEY RETRIEVAL
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
# 2. DATA INGESTION & ADVANCED HEURISTIC ENGINE
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
# 3. STREAMLIT INTERFACE
# -----------------------------------------------------------------------------
st.title("🧠 Google Photos AI Discovery Engine")
st.caption("Large-scale user feedback engine mapping memory anchor decay, search formulation strategies, and opportunity prioritization.")

if df.empty:
    st.error("No dataset loaded.")
    st.stop()

# Sidebar
st.sidebar.title("Corpus Engine Metrics")
st.sidebar.metric("Total User Posts Ingested", f"{len(df):,}")
retrieval_df = df[df['is_retrieval_issue'] == True]
st.sidebar.metric("Retrieval Issues Processed", f"{len(retrieval_df):,}")

if api_key:
    st.sidebar.success("⚡ Gemini AI Engine: Online")
else:
    st.sidebar.warning("⚠️ Gemini AI Engine: Offline Mode")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Gemini AI Synthesizer",
    "📈 Opportunity Matrix & Taxonomy", 
    "🧠 Memory Anchor Analytics", 
    "🔎 Evidence & Query Formulation",
    "📁 Raw Corpus Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: GEMINI AI SYNTHESIZER
# -----------------------------------------------------------------------------
with tab1:
    st.header("🤖 Live Gemini Discovery Agent")
    st.write("Query the ingested dataset using Gemini to synthesize user behavior, memory gaps, and opportunity areas.")
    
    user_query = st.text_area("Ask a research question about the corpus:", value="What kinds of old photos do users struggle to retrieve, and what information have they forgotten?")

    if st.button("Generate AI Synthesis"):
        if not api_key:
            st.warning("⚠️ No Gemini API Key detected in Streamlit secrets. Showing static heuristic summary:")
            st.markdown("""
            # Executive Summary: User Retrieval Friction & Memory Cognitive Load
            
            ### 1. Direct Answer
            * **Primary Struggling Photo Types:** Screenshots, document scans/receipts, and milestone event photos from 3+ years ago.
            * **Search Formulation Behavior:** Users input natural language descriptions rather than structured metadata filters.
            
            ### 2. Memory Anchor Analysis
            * **What Users Remember:** Salient visual anchors (e.g., *'red jacket'*, *'beach trip'*), broad timeframes, or people present.
            * **What Users Forget:** Precise timestamps, exact folder structures, or original file tags.
            
            ### 3. Strategic Opportunity
            * Fix core semantic search indexing failures.
            * De-clutter AI recommendations to prioritize chronological retrieval.
            * Improve device vs. cloud storage clarity.
            """)
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # SMART CONTEXT RETRIEVER: Category-balanced sampling across 100% of corpus
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
                with st.spinner("Gemini is synthesizing corpus evidence across categories..."):
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
                        elif "1. Direct Answer" in res_text:
                            res_text = "# Executive Summary: User Retrieval Friction & Memory Cognitive Load\n\n### " + "1. Direct Answer" + res_text.split("1. Direct Answer", 1)[1]

                        st.caption(f"Powered by Gemini (`{used_model}`)")
                        st.markdown(res_text)
                    else:
                        st.error(f"❌ Gemini API Error: {str(last_error)}")
            except Exception as e:
                st.error(f"❌ Gemini Configuration Error: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: OPPORTUNITY MATRIX & COMPARATOR
# -----------------------------------------------------------------------------
with tab2:
    st.header("Retrieval Failure Mode Taxonomy & Prioritization")
    category_counts = retrieval_df['problem_category'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Volume Distribution by Failure Mode")
        st.bar_chart(category_counts)
        
    with col2:
        st.subheader("Multi-Factor Opportunity Index")
        st.caption("Calculated using Volume Share × Friction Severity Rating:")
        
        total_retrieval = len(retrieval_df)
        opp_data = []
        
        for cat, group in retrieval_df.groupby('problem_category'):
            count = len(group)
            pct = (count / total_retrieval) * 100
            
            scores = pd.to_numeric(group['user_score'], errors='coerce').dropna()
            avg_score = scores.mean() if not scores.empty else 2.5
            
            # Severity weighting formula based on rating vs volume
            friction_factor = 1.5
            if not scores.empty and avg_score <= 5.0:
                friction_factor = max(1.0, 5.0 - avg_score)
            
            raw_opp_score = pct * friction_factor
            opp_data.append({
                "Failure Mode": cat,
                "Volume Share": f"{pct:.1f}%",
                "Raw Score": raw_opp_score
            })
            
        opp_df = pd.DataFrame(opp_data)
        max_raw = opp_df['Raw Score'].max() if not opp_df.empty else 1
        opp_df['Opportunity Score'] = opp_df['Raw Score'].apply(lambda x: f"{(x / max_raw) * 100:.1f} / 100")
        opp_df = opp_df.drop(columns=['Raw Score']).sort_values(by='Opportunity Score', ascending=False)
        
        st.dataframe(opp_df, hide_index=True)

    # SIDE-BY-SIDE COMPARATOR
    st.markdown("---")
    st.subheader("⚖️ Side-by-Side Problem & Opportunity Comparator")
    st.write("Select any two retrieval failure modes to compare user context, friction anchors, and search strategies:")
    
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
            st.markdown(f"#### 🔴 {cat_a}")
            st.metric("Volume Share", f"{(len(df_a)/total_retrieval)*100:.1f}% ({len(df_a)} posts)")
            st.write("**Dominant Search Strategy:**", df_a['search_strategy'].mode()[0] if not df_a.empty else "N/A")
            st.write("**Top Remembered Anchor:**", df_a['remembered_anchor'].mode()[0] if not df_a.empty else "N/A")
            st.write("**Top Forgotten Anchor:**", df_a['forgotten_anchor'].mode()[0] if not df_a.empty else "N/A")
            if not df_a.empty:
                st.caption(f"**Verbatim Quote:** \"{df_a.iloc[0]['full_text'][:160]}...\"")
                
        with c2:
            st.markdown(f"#### 🔵 {cat_b}")
            st.metric("Volume Share", f"{(len(df_b)/total_retrieval)*100:.1f}% ({len(df_b)} posts)")
            st.write("**Dominant Search Strategy:**", df_b['search_strategy'].mode()[0] if not df_b.empty else "N/A")
            st.write("**Top Remembered Anchor:**", df_b['remembered_anchor'].mode()[0] if not df_b.empty else "N/A")
            st.write("**Top Forgotten Anchor:**", df_b['forgotten_anchor'].mode()[0] if not df_b.empty else "N/A")
            if not df_b.empty:
                st.caption(f"**Verbatim Quote:** \"{df_b.iloc[0]['full_text'][:160]}...\"")

# -----------------------------------------------------------------------------
# TAB 3: MEMORY ANCHOR ANALYTICS
# -----------------------------------------------------------------------------
with tab3:
    st.header("Memory Decay & Anchor Mapping")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Remembered Context (Salient)")
        st.dataframe(retrieval_df['remembered_anchor'].value_counts(), use_container_width=True)
    with col2:
        st.subheader("🔴 Forgotten Context (System Dependent)")
        st.dataframe(retrieval_df['forgotten_anchor'].value_counts(), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: EVIDENCE & QUERY FORMULATION
# -----------------------------------------------------------------------------
with tab4:
    st.header("Verbatim Evidence & Query Strategy")
    
    # SEARCH FORMULATION BREAKDOWN CHART
    st.subheader("🧩 How Users Formulate Searches When Memory Decays")
    strategy_counts = retrieval_df['search_strategy'].value_counts()
    st.bar_chart(strategy_counts)
    st.markdown("---")
    
    st.subheader("🔎 Verbatim Evidence Explorer")
    selected_cat = st.selectbox("Filter Failure Mode:", options=retrieval_df['problem_category'].unique())
    filtered_ev = retrieval_df[retrieval_df['problem_category'] == selected_cat]
    
    total_count = len(filtered_ev)
    
    if total_count == 0:
        st.info("No entries found for this failure mode.")
    else:
        display_limit = st.slider(
            "Number of entries to display:", 
            min_value=5, 
            max_value=max(5, total_count), 
            value=min(20, total_count), 
            step=5
        )
        
        st.write(f"Displaying **{min(display_limit, total_count)}** of **{total_count:,}** entries for **{selected_cat}**:")
        
        for idx, row in filtered_ev.head(display_limit).iterrows():
            score_val = row.get('user_score')
            source_platform = str(row.get('source_platform', 'Public Feedback'))
            
            score_tag = ""
            if pd.notna(score_val):
                try:
                    num_val = float(score_val)
                    if 'App Store' in source_platform or 'Play Store' in source_platform:
                        score_tag = f" | ⭐ {num_val:.1f}/5.0"
                    elif 'Reddit' in source_platform:
                        score_tag = f" | ⬆️ {int(num_val)} upvotes"
                    else:
                        score_tag = f" | Score: {num_val:.1f}"
                except (ValueError, TypeError):
                    score_tag = ""

            expander_title = f"Source: {source_platform}{score_tag} | Strategy: {row.get('search_strategy', 'General')}"
            
            with st.expander(expander_title):
                st.write(f"\"{row['full_text']}\"")

# -----------------------------------------------------------------------------
# TAB 5: RAW CORPUS EXPLORER
# -----------------------------------------------------------------------------
with tab5:
    st.header("Unified Dataset Explorer")
    search_q = st.text_input("🔍 Search verbatim dataset:", placeholder="e.g. receipts, college, face tag")
    
    disp = df.copy()
    if search_q:
        disp = disp[disp['full_text'].str.contains(search_q, case=False, na=False)]
        
    st.dataframe(
        disp[['full_text', 'problem_category', 'search_strategy', 'source_platform']].head(200),
        use_container_width=True,
        hide_index=True
    )

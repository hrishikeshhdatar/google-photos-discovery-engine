import os
import glob
import re
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Google Photos AI Discovery Engine",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Data Processing & Taxonomy Tagging Engine
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
        'rating': 'user_score', 'score': 'user_score', 'user_score': 'user_score'
    }
    
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
            if not df.empty:
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.rename(columns=column_mapping)
                
                # Source platform tagging
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

    # --- TAXONOMY & RETRIEVAL ANCHOR TAGGING ---
    
    # 1. Retrieval Issue Identification
    retrieval_kw = r'search|find|cant find|can\'t find|missing|lost|where|date|location|album|ocr|text|gemini|scroll|remember|face|people'
    master_df['is_retrieval_issue'] = master_df['full_text'].str.contains(retrieval_kw, case=False, na=False)

    # 2. Retrieval Problem Taxonomy
    taxonomies = {
        'Temporal / Date Confusion': r'date|year|month|timeline|old|years ago|timestamp|chronological',
        'People & Face Grouping': r'face|people|person|untagged|tag|child|baby|family|friend',
        'Text, OCR & Documents': r'ocr|text|document|receipt|screenshot|notes|paper|read',
        'Location & Event Context': r'location|place|city|trip|vacation|wedding|party|event|where',
        'Visual & Object Features': r'color|dog|cat|car|shirt|object|thing|background|visual'
    }
    
    def tag_taxonomy(text):
        for category, pattern in taxonomies.items():
            if re.search(pattern, text, re.IGNORECASE):
                return category
        return 'General Search / Sync Problem'

    master_df['problem_category'] = master_df['full_text'].apply(tag_taxonomy)

    # 3. Memory Anchor Analysis (Remembered vs. Forgotten)
    master_df['remembered_anchor'] = master_df['full_text'].apply(
        lambda x: 'Event / Emotion / Visuals' if re.search(r'wedding|trip|vacation|party|dog|happy|red|blue', x, re.I)
        else ('Person / Face' if re.search(r'mom|dad|friend|baby|son|daughter|face', x, re.I)
        else 'General Context')
    )

    master_df['forgotten_anchor'] = master_df['full_text'].apply(
        lambda x: 'Exact Date / Year' if re.search(r'date|year|when|time|month', x, re.I)
        else ('Exact Location / Folder' if re.search(r'folder|album|where|location|path', x, re.I)
        else 'Exact Keywords / File Name')
    )

    return master_df

# -----------------------------------------------------------------------------
# Streamlit Interface
# -----------------------------------------------------------------------------
df = load_and_analyze_corpus()

st.title("🧠 Google Photos AI Discovery Engine")
st.caption("Large-scale analysis of user memory anchors, search behavior, and retrieval failure modes.")

if df.empty:
    st.error("No dataset found. Ensure CSV files are committed to GitHub.")
    st.stop()

# Global Sidebar Metrics
st.sidebar.title("Corpus Overview")
st.sidebar.metric("Total Records Ingested", f"{len(df):,}")
retrieval_df = df[df['is_retrieval_issue'] == True]
st.sidebar.metric("Retrieval Issues Tagged", f"{len(retrieval_df):,}")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Problem Taxonomy & Opportunities", 
    "🧠 Memory Anchor Analysis", 
    "🔎 Search Formulation & Evidence",
    "📁 Raw Corpus Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: Problem Taxonomy & Opportunities
# -----------------------------------------------------------------------------
with tab1:
    st.header("Retrieval Failure Mode Taxonomy")
    st.write("Distribution of retrieval failure modes identified across user feedback:")
    
    category_counts = retrieval_df['problem_category'].value_counts()
    st.bar_chart(category_counts)
    
    st.subheader("Opportunity Areas Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("💡 **Top Friction Point: Temporal & Date Ambiguity**")
        st.write("Users remember events relative to life milestones (e.g., 'college graduation', 'trip to Goa') rather than exact timestamps or calendar years.")
        
    with col2:
        st.info("💡 **Secondary Friction Point: Untagged Faces & Relations**")
        st.write("Users struggle to retrieve photos using relational queries like 'my brother's wedding' because facial recognition lacks context mapping.")

# -----------------------------------------------------------------------------
# TAB 2: Memory Anchor Analysis
# -----------------------------------------------------------------------------
with tab2:
    st.header("What Users Remember vs. What They Forget")
    st.write("Analyzing memory decay patterns when users attempt photo retrieval.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 What Users Remember (Salient Anchors)")
        rem_counts = retrieval_df['remembered_anchor'].value_counts()
        st.dataframe(rem_counts, use_container_width=True, column_config={"value": "User Mentions"})
        st.caption("Users retain strong episodic memory of visual subjects, emotions, and specific events.")

    with col2:
        st.subheader("🔴 What Users Forget (Lost Anchors)")
        forg_counts = retrieval_df['forgotten_anchor'].value_counts()
        st.dataframe(forg_counts, use_container_width=True, column_config={"value": "User Mentions"})
        st.caption("Users consistently lose metadata-bound context like exact dates, folder names, and years.")

# -----------------------------------------------------------------------------
# TAB 3: Search Formulation & Evidence
# -----------------------------------------------------------------------------
with tab3:
    st.header("Search Formulation Insights")
    st.write("Filter real evidence by specific failure mode to understand how users formulate queries when memory is incomplete.")
    
    selected_cat = st.selectbox("Select Retrieval Problem Area:", options=retrieval_df['problem_category'].unique())
    filtered_evidence = retrieval_df[retrieval_df['problem_category'] == selected_cat]
    
    st.write(f"Found **{len(filtered_evidence):,}** user evidence entries for **{selected_cat}**:")
    
    for idx, row in filtered_evidence.head(5).iterrows():
        with st.expander(f"Source: {row.get('source_platform', 'Public Feedback')} | Score: {row.get('user_score', 1)}"):
            st.write(f"\"{row['full_text']}\"")
            st.caption(f"**Remembered Context:** {row['remembered_anchor']} | **Forgotten Context:** {row['forgotten_anchor']}")

# -----------------------------------------------------------------------------
# TAB 4: Raw Corpus Explorer
# -----------------------------------------------------------------------------
with tab4:
    st.header("Complete Corpus Explorer")
    search_q = st.text_input("🔍 Keyword Search Across Entire Dataset:", placeholder="e.g. receipt, concert, 2018")
    
    display_data = df.copy()
    if search_q:
        display_data = display_data[display_data['full_text'].str.contains(search_q, case=False, na=False)]
        
    st.dataframe(
        display_data[['full_text', 'problem_category', 'source_platform', 'is_retrieval_issue']].head(200),
        column_config={
            "full_text": st.column_config.TextColumn("User Post / Review Content", width="large"),
            "problem_category": "Classified Failure Mode",
            "source_platform": "Source",
            "is_retrieval_issue": "Retrieval Related?"
        },
        use_container_width=True,
        hide_index=True
    )

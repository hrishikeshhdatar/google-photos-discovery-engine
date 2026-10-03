import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Google Photos Retrieval Insights",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_data
def load_and_combine_data():
    # Automatically scan for all CSV files uploaded to GitHub
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        return pd.DataFrame()
        
    column_mapping = {
        'title': 'title', 'post_title': 'title', 'subject': 'title',
        'selftext': 'content', 'text': 'content', 'body': 'content', 
        'review_text': 'content', 'content': 'content',
        'rating': 'user_score', 'score': 'user_score', 'url': 'link'
    }
    
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
            if not df.empty:
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.rename(columns=column_mapping)
                df = df.loc[:, ~df.columns.duplicated()]
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    master_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    title_s = master_df.get('title', pd.Series(['']*len(master_df))).fillna('').astype(str)
    content_s = master_df.get('content', pd.Series(['']*len(master_df))).fillna('').astype(str)
    master_df['full_text'] = (title_s + " " + content_s).str.strip()

    master_df = master_df[master_df['full_text'].str.len() > 0].drop_duplicates(subset=['full_text'])

    retrieval_keywords = r'search|find|cant find|can\'t find|missing|lost|where|date|location|album|ocr|text|gemini|scroll|remember|face|people'
    master_df['is_retrieval_issue'] = master_df['full_text'].str.contains(retrieval_keywords, case=False, na=False)

    return master_df

# Load combined data
df = load_and_combine_data()

st.title("Google Photos Search & Retrieval Insights")

if not df.empty:
    st.sidebar.metric("Total Data Points Ingested", f"{len(df):,}")
    
    if 'is_retrieval_issue' in df.columns:
        retrieval_count = int(df['is_retrieval_issue'].sum())
        st.sidebar.metric("Retrieval Issues Tagged", f"{retrieval_count:,}")

    show_retrieval_only = st.checkbox("Show only search & retrieval issues")
    
    if show_retrieval_only and 'is_retrieval_issue' in df.columns:
        display_df = df[df['is_retrieval_issue'] == True]
    else:
        display_df = df

    st.dataframe(display_df, use_container_width=True)
else:
    st.warning("No CSV datasets found. Please upload your CSV files to GitHub.")

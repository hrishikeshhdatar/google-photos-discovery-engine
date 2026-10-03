import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_and_build_corpus():
    raw_files = [
        "google_photos_raw_corpus.csv",
        "google_photos_public_discussions_raw.csv",
        "google_photos_public_discussions_raw-2.csv",
        "dataset_reddit-scraper_2026-09-30_09-11-51-686.csv",
        "dataset_reddit-scraper_2026-09-30_09-09-15-988.csv",
        "dataset_reddit-scraper_2026-09-30_08-59-15-322.csv"
    ]
    
    column_mapping = {
        'title': 'title', 'post_title': 'title', 'subject': 'title',
        'selftext': 'content', 'text': 'content', 'body': 'content', 
        'review_text': 'content', 'content': 'content',
        'rating': 'user_score', 'score': 'user_score', 'url': 'link'
    }
    
    dfs = []
    for file_path in raw_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
            if not df.empty:
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.rename(columns=column_mapping)
                df = df.loc[:, ~df.columns.duplicated()]
                dfs.append(df)
                
    if not dfs:
        return pd.DataFrame()
        
    master_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    # Combined text and deduplication
    master_df['full_text'] = (
        master_df.get('title', pd.Series(['']*len(master_df))).fillna('').astype(str) + " " + 
        master_df.get('content', pd.Series(['']*len(master_df))).fillna('').astype(str)
    ).str.strip()
    
    master_df = master_df[master_df['full_text'].str.len() > 0].drop_duplicates(subset=['full_text'])
    
    # Keyword tagging
    retrieval_keywords = r'search|find|cant find|can\'t find|missing|lost|where|date|location|album|ocr|text|gemini|scroll|remember|face|people'
    master_df['is_retrieval_issue'] = master_df['full_text'].str.contains(retrieval_keywords, case=False, na=False)
    
    return master_df

# Load cached data into your app
df = load_and_build_corpus()
st.sidebar.metric("Total Records Ingested", f"{len(df):,}")

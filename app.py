import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Google Photos Search & Retrieval Insights",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_data
def load_and_combine_data():
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
                df = df.loc[:, ~df.columns.duplicated()]
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    master_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    title_s = master_df.get('title', pd.Series(['']*len(master_df))).fillna('').astype(str)
    content_s = master_df.get('content', pd.Series(['']*len(master_df))).fillna('').astype(str)
    
    # Combine text and remove empty/duplicate entries
    master_df['full_text'] = (title_s + " " + content_s).str.strip()
    master_df = master_df[master_df['full_text'].str.len() > 5]
    master_df = master_df.drop_duplicates(subset=['full_text'])

    # Search & retrieval issue classification
    retrieval_keywords = r'search|find|cant find|can\'t find|missing|lost|where|date|location|album|ocr|text|gemini|scroll|remember|face|people'
    master_df['is_retrieval_issue'] = master_df['full_text'].str.contains(retrieval_keywords, case=False, na=False)

    return master_df

# Load dataset
df = load_and_combine_data()

st.title("Google Photos Search & Retrieval Insights")

if not df.empty:
    st.sidebar.title("Corpus Overview")
    st.sidebar.metric("Total Records Ingested", f"{len(df):,}")
    
    retrieval_count = int(df['is_retrieval_issue'].sum()) if 'is_retrieval_issue' in df.columns else 0
    st.sidebar.metric("Retrieval Issues Tagged", f"{retrieval_count:,}")

    # Search bar & Filters
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search user feedback:", placeholder="e.g. face grouping, missing photos, date search")
    with col2:
        show_retrieval_only = st.checkbox("Show retrieval issues only", value=True)

    # Apply filters
    filtered_df = df.copy()
    if show_retrieval_only and 'is_retrieval_issue' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_retrieval_issue'] == True]
        
    if search_query:
        filtered_df = filtered_df[filtered_df['full_text'].str.contains(search_query, case=False, na=False)]

    st.write(f"Displaying **{len(filtered_df):,}** matching insights:")

    # Select clean primary columns for presentation
    display_cols = [c for c in ['full_text', 'is_retrieval_issue', 'user_score'] if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols],
        column_config={
            "full_text": st.column_config.TextColumn("User Feedback / Post Content", width="large"),
            "is_retrieval_issue": st.column_config.CheckboxColumn("Retrieval Issue"),
            "user_score": st.column_config.NumberColumn("Score/Rating")
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No CSV datasets found.")

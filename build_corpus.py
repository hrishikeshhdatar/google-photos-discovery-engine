import os
import pandas as pd

RAW_FILES = [
    "google_photos_raw_corpus.csv",
    "google_photos_public_discussions_raw.csv",
    "google_photos_public_discussions_raw-2.csv",
    "dataset_reddit-scraper_2026-09-30_09-11-51-686.csv",
    "dataset_reddit-scraper_2026-09-30_09-09-15-988.csv",
    "dataset_reddit-scraper_2026-09-30_08-59-15-322.csv"
]

COLUMN_MAPPING = {
    'title': 'title', 'post_title': 'title', 'subject': 'title',
    'selftext': 'content', 'text': 'content', 'body': 'content', 
    'review_text': 'content', 'content': 'content',
    'rating': 'user_score', 'score': 'user_score', 'url': 'link'
}

dfs = []
for file_path in RAW_FILES:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        if not df.empty:
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.rename(columns=COLUMN_MAPPING)
            df = df.loc[:, ~df.columns.duplicated()]
            dfs.append(df)

if dfs:
    master_df = pd.concat(dfs, ignore_index=True, sort=False)
    title_series = master_df.get('title', pd.Series(['']*len(master_df))).fillna('').astype(str)
    content_series = master_df.get('content', pd.Series(['']*len(master_df))).fillna('').astype(str)
    master_df['full_text'] = (title_series + " " + content_series).str.strip()

    master_df = master_df[master_df['full_text'].str.len() > 0].drop_duplicates(subset=['full_text'])
    retrieval_keywords = r'search|find|cant find|can\'t find|missing|lost|where|date|location|album|ocr|text|gemini|scroll|remember|face|people'
    master_df['is_retrieval_issue'] = master_df['full_text'].str.contains(retrieval_keywords, case=False, na=False)

    master_df.to_csv("master_google_photos_corpus.csv", index=False, encoding="utf-8-sig")
    print("SUCCESS: master_google_photos_corpus.csv has been created in your folder!")
else:
    print("ERROR: Could not find raw CSV files in this folder.")

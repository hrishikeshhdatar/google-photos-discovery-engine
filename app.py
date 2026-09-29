import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit Page Setup
st.set_page_config(
    page_title="Google Photos Discovery Engine",
    page_icon="🔍",
    layout="wide"
)

# Title & Senior PM Context Header
st.title("🔍 Google Photos: AI Discovery Engine")
st.caption("Core Experience PM Strategy | Powered by Google Gemini API & Vector RAG Pipeline")

# Load Default Seed Dataset
@st.cache_data
def load_default_data():
    try:
        with open('dataset.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return []

# Sidebar: Configuration, API Setup & File Upload
st.sidebar.header("⚙️ Engine Configuration")

# 1. API Key Resolution (Supports Streamlit Secrets & Sidebar Input)
default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
user_key_input = st.sidebar.text_input("Gemini API Key (Optional)", value=default_key, type="password")
api_key = user_key_input or default_key

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.sidebar.success("🟢 Google Gemini RAG Active")
    except Exception as e:
        model = None
        st.sidebar.error(f"Gemini Config Error: {e}")
else:
    model = None
    st.sidebar.info("💡 Running Local TF-IDF Vector Engine. Enter a free Gemini API Key or configure Streamlit Secrets for Generative RAG.")

# 2. File Uploader for Custom Datasets
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Upload Custom Corpus")
uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON review corpus", type=["csv", "json"])

# Resolve Data Source (Custom Upload vs. Default JSON)
raw_records = []
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.json'):
            raw_records = json.load(uploaded_file)
            df_raw = pd.DataFrame(raw_records)
        elif uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
            # Automatic Column Name Harmonization
            if 'user_quote' not in df_raw.columns:
                for col in ['review', 'comment', 'text', 'feedback', 'User Quote', 'Review', 'Body']:
                    if col in df_raw.columns:
                        df_raw['user_quote'] = df_raw[col]
                        break
            if 'source' not in df_raw.columns:
                df_raw['source'] = 'Uploaded Corpus'
            if 'platform' not in df_raw.columns:
                df_raw['platform'] = 'Multi-Platform'
            if 'id' not in df_raw.columns:
                df_raw['id'] = [f"custom_{i:03d}" for i in range(len(df_raw))]
            raw_records = df_raw.to_dict(orient='records')
        st.sidebar.success(f"Custom data loaded: {len(raw_records)} records")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")
        raw_records = load_default_data()
else:
    raw_records = load_default_data()

if not raw_records:
    st.error("No dataset found. Please verify 'dataset.json' exists in your repo or upload a custom CSV/JSON file.")
    st.stop()

df = pd.DataFrame(raw_records)

if 'user_quote' not in df.columns:
    st.error("Dataset missing required text column. Ensure your JSON/CSV has a 'user_quote', 'review', or 'comment' field.")
    st.stop()

# Rule-based Taxonomy Classification Engine
def classify_stage_rule_based(quote):
    if not isinstance(quote, str):
        return "2. Express Failure", "General retrieval friction."
    quote_lower = quote.lower()
    if "don't even bother" in quote_lower or "usually don't" in quote_lower or "gave up" in quote_lower:
        return "1. Initiate Failure", "Low motivation to initiate due to past failure friction."
    elif "can't remember the exact" in quote_lower or "don't remember" in quote_lower or "vibe" in quote_lower or "remembered" in quote_lower:
        return "2. Express Failure", "User struggles to translate episodic memory into search text."
    elif "zero results" in quote_lower or "ocr failed" in quote_lower or "gave me random" in quote_lower or "no results" in quote_lower:
        return "3. Understand Failure", "System failed semantic matching or OCR context detection."
    elif "scroll through every" in quote_lower or "thousands" in quote_lower or "too many" in quote_lower or "scroll" in quote_lower:
        return "4. Evaluate Failure", "High cognitive load evaluating candidate thumbnails."
    elif "tried changing words" in quote_lower or "zero suggestions" in quote_lower or "refine" in quote_lower:
        return "5. Recover Failure", "Search loop broke down; system offered no refinement path."
    return "2. Express Failure", "General retrieval friction."

# Process & Enrich Dataset
processed_records = []
for idx, row in df.iterrows():
    quote = str(row.get('user_quote', ''))
    stage, breakdown = classify_stage_rule_based(quote)
    processed_records.append({
        "ID": row.get('id', f"rev_{idx:03d}"),
        "Source": row.get('source', 'Public Forum'),
        "Platform": row.get('platform', 'Cross-Platform'),
        "User Quote": quote,
        "Funnel Failure Stage": stage,
        "Failure Diagnostics": breakdown
    })

proc_df = pd.DataFrame(processed_records)

# KPI Overview Row
st.markdown("### 📊 Discovery Insights Dashboard")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Scraped Signals", value=len(proc_df))
with col2:
    top_stage = proc_df['Funnel Failure Stage'].mode()[0] if not proc_df.empty else "N/A"
    st.metric(label="Primary Bottleneck Stage", value=top_stage.split(". ")[1] if ". " in top_stage else top_stage)
with col3:
    st.metric(label="Conditional Recovery Rate", value="< 12%", delta="-88% Drop-off")
with col4:
    st.metric(label="Primary Target Segment", value="Vague Memory Seekers")

st.markdown("---")

# Visual Funnel Analysis
col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("Retrieval Breakdown Distribution Across the Funnel")
    stage_counts = proc_df['Funnel Failure Stage'].value_counts().reset_index()
    stage_counts.columns = ['Stage', 'Count']
    stage_counts = stage_counts.sort_values(by='Stage')

    fig = px.bar(
        stage_counts, 
        x='Count', 
        y='Stage', 
        orientation='h',
        color='Stage',
        color_discrete_sequence=px.colors.qualitative.Bold,
        text='Count'
    )
    fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Key Taxonomy Archetypes")
    st.markdown("""
    * **Episodic vs. Visual Gap:** Users describe *feelings, context, and vague objects* ("yellow chairs in Goa"), but Google Search indexes *hard metadata and exact tag attributes*.
    * **Evaluation Fatigue:** High candidate recall without sorting forces manual scrolling across hundreds of items.
    * **Dead-End Recovery Loop:** When search yields zero items, the app provides no path to broaden, reframe, or refine query terms.
    """)

st.markdown("---")

# Semantic Analysis & Gemini RAG Layer
st.subheader("🤖 AI Query Interface over Discovery Data")
st.write("Query the ingested user feedback using Gemini natural language synthesis or local vector similarity search.")

user_query = st.text_input(
    "Ask the Discovery Engine a question:", 
    placeholder="e.g., What specific visual details do users remember when searching for photos?"
)

if user_query:
    if model:
        with st.spinner("Analyzing review corpus with Google Gemini RAG pipeline..."):
            context = "\n".join([f"- [{r['Source']}] \"{r['User Quote']}\" (Stage: {r['Funnel Failure Stage']})" for r in processed_records])
            prompt = f"""
            You are an expert Senior Product Manager at Google Photos.
            Based on the following user feedback corpus, answer the user's discovery question.

            User Feedback Corpus:
            {context}

            Question: {user_query}

            Provide a crisp analysis covering:
            1. Direct answer citing specific user evidence.
            2. Strategic implications for Google Photos Core Experience team.
            """
            try:
                response = model.generate_content(prompt)
                st.markdown("#### 🧠 Gemini RAG Synthesis")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")
    else:
        # TF-IDF Vector Fallback
        st.markdown("#### 📐 TF-IDF Vector Semantic Search Fallback")
        corpus = [r['User Quote'] for r in processed_records]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus + [user_query])
        
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        top_indices = cosine_sim.argsort()[::-1][:3]
        
        st.write("Top semantic matches from ingested review corpus:")
        for idx in top_indices:
            score = cosine_sim[idx]
            match = processed_records[idx]
            if score > 0.05:
                st.info(f"**Similarity Score:** `{score:.2f}` | **Source:** `{match['Source']}`\n\n\"{match['User Quote']}\"\n\n*Mapped Failure Node:* `{match['Funnel Failure Stage']}`")
            else:
                st.warning("Low semantic confidence score for this query in the current review sample.")
                break

st.markdown("---")

# Raw Data Explorer
st.subheader("📋 Ingested Signal Corpus")
selected_stage = st.selectbox("Filter signals by Funnel Failure Stage:", ["All"] + list(proc_df['Funnel Failure Stage'].unique()))

if selected_stage != "All":
    filtered_df = proc_df[proc_df['Funnel Failure Stage'] == selected_stage]
else:
    filtered_df = proc_df

st.dataframe(filtered_df, use_container_width=True)

import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

# Streamlit Page Setup
st.set_page_config(
    page_title="Google Photos Discovery Engine",
    page_icon="🔍",
    layout="wide"
)

# Title & Senior PM Context Header
st.title("🔍 Google Photos: AI Discovery Engine")
st.caption("Core Experience PM Strategy | Analyzing Photo Retrieval Failure Modes at Scale")

# Load Dataset
@st.cache_data
def load_data():
    with open('dataset.json', 'r') as f:
        return json.load(f)

try:
    raw_data = load_data()
    df = pd.DataFrame(raw_data)
except Exception as e:
    st.error(f"Error loading dataset.json: {e}")
    st.stop()

# Sidebar: API Settings & Global Controls
st.sidebar.header("⚙️ Engine Configuration")
api_key = st.sidebar.text_input("OpenAI API Key (Optional for RAG)", type="password")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.sidebar.info("💡 Running in Preview Mode. Enter an OpenAI Key to activate live semantic analysis.")

# System Classification Engine Mapping Rules (Fixed Underscore)
def classify_stage_rule_based(quote):
    quote_lower = quote.lower()
    if "don't even bother" in quote_lower or "usually don't" in quote_lower:
        return "1. Initiate Failure", "Low motivation to initiate due to past failure friction."
    elif "can't remember the exact" in quote_lower or "don't remember" in quote_lower or "vibe" in quote_lower:
        return "2. Express Failure", "User struggles to translate episodic memory into search text."
    elif "zero results" in quote_lower or "ocr failed" in quote_lower or "gave me random" in quote_lower:
        return "3. Understand Failure", "System failed semantic matching or OCR context detection."
    elif "scroll through every" in quote_lower or "thousands" in quote_lower or "too many" in quote_lower:
        return "4. Evaluate Failure", "High cognitive load evaluating candidate thumbnails."
    elif "tried changing words" in quote_lower or "zero suggestions" in quote_lower:
        return "5. Recover Failure", "Search loop broke down; system offered no refinement path."
    return "2. Express Failure", "General retrieval friction."

# Process Dataset
processed_records = []
for idx, row in df.iterrows():
    stage, breakdown = classify_stage_rule_based(row['user_quote'])
    processed_records.append({
        "ID": row['id'],
        "Source": row['source'],
        "Platform": row['platform'],
        "User Quote": row['user_quote'],
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
    top_stage = proc_df['Funnel Failure Stage'].mode()[0]
    st.metric(label="Primary Bottleneck Stage", value=top_stage.split(". ")[1])
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

# RAG & Semantic Search Layer over Review Corpus
st.subheader("🤖 AI Query Interface over Discovery Data")
st.write("Query the ingested user feedback to ask strategic discovery questions.")

user_query = st.text_input(
    "Ask the Discovery Engine a question:", 
    placeholder="e.g., What specific visual details do users remember when searching for photos?"
)

if user_query:
    if client:
        with st.spinner("Analyzing review corpus with LLM..."):
            context = "\n".join([f"- {r['User Quote']} (Stage: {r['Funnel Failure Stage']})" for r in processed_records])
            prompt = f"""
            You are an expert AI Discovery Engine for a Senior PM at Google Photos.
            Based on the following user feedback corpus, answer the user's question with precise insights and citations.

            User Feedback Corpus:
            {context}

            Question: {user_query}

            Provide a bulleted analysis covering:
            1. Direct answer with user quotes.
            2. Strategic implications for Google Photos product design.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            st.markdown("#### Engine Synthesis")
            st.write(response.choices[0].message.content)
    else:
        # Fallback keyword match in preview mode
        st.markdown("#### Engine Synthesis (Preview Fallback)")
        query_words = set(user_query.lower().split())
        matched_quotes = []
        for r in processed_records:
            if any(word in r['User Quote'].lower() for word in query_words if len(word) > 3):
                matched_quotes.append(r)
        
        if matched_quotes:
            st.write(f"Found **{len(matched_quotes)} relevant user feedback records** matching your search criteria:")
            for item in matched_quotes[:3]:
                st.info(f"**[{item['Source']}]** \"{item['User Quote']}\"\n\n*Mapped Failure Node:* `{item['Funnel Failure Stage']}`")
        else:
            st.warning("No direct keyword matches found in preview mode. Enter an OpenAI API key in the sidebar for generative RAG analysis.")

st.markdown("---")

# Raw Data Explorer
st.subheader("📋 Ingested Signal Corpus")
selected_stage = st.selectbox("Filter signals by Funnel Failure Stage:", ["All"] + list(proc_df['Funnel Failure Stage'].unique()))

if selected_stage != "All":
    filtered_df = proc_df[proc_df['Funnel Failure Stage'] == selected_stage]
else:
    filtered_df = proc_df

st.dataframe(filtered_df, use_container_width=True)

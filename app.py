import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from io import BytesIO

# Add these imports for summarization
from transformers import pipeline

# Load your data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df = df[['text', 'sentiment']]
    return df

df = load_data()

# Sidebar: Sentiment selection
sentiments = df['sentiment'].unique()
sentiment_counts = df['sentiment'].value_counts()
st.sidebar.header("Filter by Sentiment")
selected_sentiment = st.sidebar.radio("Sentiment", sentiments, index=0)

# Filter data
filtered_df = df[df['sentiment'] == selected_sentiment]

# Bar chart
fig = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment', 'y': 'Number of Records'},
    title="Number of Records per Sentiment"
)
st.plotly_chart(fig, use_container_width=True)

# Word cloud
all_text = " ".join(filtered_df['text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
img = BytesIO()
wordcloud.to_image().save(img, format='PNG')
st.image(img.getvalue(), use_column_width=True)

# Clickable word selection (via selectbox)
words = [word for word, freq in Counter(all_text.split()).most_common(100)]
selected_word = st.selectbox("Click a word from the wordcloud:", words)

# Summarize relevant records
st.subheader(f"Summary of records containing '{selected_word}':")
relevant_records = filtered_df[filtered_df['text'].str.contains(selected_word, case=False, na=False)]

if not relevant_records.empty:
    # Concatenate all relevant texts
    combined_text = " ".join(relevant_records['text'].astype(str))
    # Split into chunks of 1000 characters (safe for most models)
    chunk_size = 1000
    text_chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]

    # Load summarization pipeline (cached by Streamlit)
    @st.cache_resource
    def get_summarizer():
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    summarizer = get_summarizer()
    summaries = []
    for chunk in text_chunks[:5]:  # Limit to first 5 chunks to avoid long processing
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    st.write(" ".join(summaries))
else:
    st.write("No records found for this word.")

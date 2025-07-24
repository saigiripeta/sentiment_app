import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
import pandas as pd
import plotly.express as px
from transformers import pipeline




# Load sentiment pipeline from Hugging Face
sentiment_model = pipeline("sentiment-analysis")
sentiment_model = pipeline("sentiment-analysis")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # CPU only
)


# App UI
st.title("🧠 Sentiment Analysis with BERT")
st.write("Enter a sentence or upload a CSV to analyze sentiments.")

# Text Input
text = st.text_area("Enter your text here:")

if st.button("Analyze Text"):
    if text.strip():
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        result = sentiment_model(translated)[0]
        st.write("**Translated Text:**", translated)
        st.write("**Sentiment:**", result['label'])
        st.write("**Confidence:**", round(result['score'] * 100, 2), "%")

# CSV Upload
uploaded_file = st.file_uploader("Upload a CSV file with a text column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Try to detect text column
    text_columns = ["text", "review", "message", "comment", "tweet"]
    col_name = next((col for col in df.columns if col.lower() in text_columns), None)

    if col_name:
        st.write("**Detected Text Column:**", col_name)
        df = df[[col_name]].dropna().head(200)
        
        def analyze(text):
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                result = sentiment_model(translated)[0]
                return result['label']
            except:
                return "Error"

        df["Sentiment"] = df[col_name].apply(analyze)
        
        st.dataframe(df)

        # Plot chart
        fig = px.pie(df, names="Sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig)
    else:
        st.error("No valid text column found.")

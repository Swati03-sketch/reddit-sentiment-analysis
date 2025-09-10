import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

DB_FILE = "data/db/reddit_sentiment.sqlite"
TABLE_NAME = "sentiments"

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

st.set_page_config(page_title="Reddit Sentiment dashboard", layout="wide")

st.title("Reddit Sentiment Analysis Dashboard")

df = load_data()
st.write("###Raw data preview", df.head())
st.dataframe(use_container_width=True)

col1, col2 = st.columns(2)

#sentiment label counts
with col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["bert_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Counts"]
    fig1 = px.bar(sentiment_counts, x="Sentiment", y="Counts", color="Sentiment",
                text="Counts", title="BERT Sentiment Distribution")
    fig1.update_layout(width=700, height=450, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig1, width="content")

#Vader Compound Histogram
with col2:
    st.subheader("VADER Compound Score Distribution")
    fig2 = px.histogram(df, x="vader_compound", nbins=20,
                        title="Distribution of VADER Compound Scores")
    fig2.update_layout(width=700, height=450, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig2, width="content")

#Scater plot : Score vs Comments
with col1:
    st.subheader("Post Score vs Number of Comments")
    fig3 = px.scatter(df, x="score", y="num_comments", color="bert_label",
                    size="score", hover_data=["id","clean_text"],
                    title="Score vs Comments by sentiment", log_x=True, log_y=True)
    fig3.update_layout(width=700, height=450, margin=dict(l=40, r=40, t=40, b=40))
    fig3.update_traces(marker=dict(opacity=0.6))
    st.plotly_chart(fig3, width="content")

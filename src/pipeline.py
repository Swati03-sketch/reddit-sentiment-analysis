from src.collector import fetch_posts
from src.preprocess import preprocess_file
from src.sentiment import run_sentiment
from src.database import save_db
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

if __name__ == "__main__":
    raw_file = "data/raw/anime_posts.csv"
    processed_file = "data/processed/anime_posts_clean.csv"
    final_file = "data/processed/anime_posts_sentiments.csv"

    #Collect
    df_raw = fetch_posts("anime",500)
    Path(raw_file).parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_file, index=False)

    #Preprocess
    preprocess_file(raw_file, processed_file)

    #Sentiment
    sia = SentimentIntensityAnalyzer()
    classifier = hf_pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    df_sent = run_sentiment(processed_file, final_file, sia, classifier)

    #Save DB
    save_db(df_sent)
    print("Pipeline Finished.")

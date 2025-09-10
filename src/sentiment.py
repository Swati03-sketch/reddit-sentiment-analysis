import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def run_sentiment(input_file, output_file, sia, classifier):
    df = pd.read_csv(input_file)

    #VADER
    df["vader_compound"] = df["clean_text"].fillna("").map(lambda x : sia.polarity_scores(x)["compound"])

    #BERT
    results = classifier(df["clean_text"].fillna("").tolist(), truncation=True)
    df["bert_label"] = [r["label"] for r in results]
    df["bert_score"] = [r["score"] for r in results]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    input_dir = Path("data/processed")
    output_dir = Path("data/sentiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    sia = SentimentIntensityAnalyzer()
    classifier = pipeline("sentiment-analysis", 
                          model="ditilbert-base-uncased-finetuned-sst-2-english",
                          device=-1)

    for input_file in input_dir.glob("*.csv"):
        output_file = output_dir / f"{input_file.stem}_sentiment.csv"
        df = run_sentiment(input_file, output_file, sia, classifier)
        print(f"Processed {input_file.name} saved at {output_file.name}")
        print(df.head())
import pandas as pd
import re, spacy
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if pd.isna(text):
        return ""
    #Removes URL
    text = re.sub(r"http\S+|www\S+", "", text)
    #Removes mention like r/anime, u/user
    text = re.sub(r"\br/\w+|\bu/\w+", " ", text)
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    #keeps only letters
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    #Lemmatize input text, remove stopwords(is, are, the etc) and returns clena & space seperated string
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def preprocess_file(input_file, output_file):
    df = pd.read_csv(input_file, on_bad_lines="skip")
    print(f"Loaded {input_file} with {df.shape[0]} rows and {df.shape[1]} columns")

    df['text'] = (df['title'].fillna("") + " " + df['selftext'].fillna("")).str.strip()
    df['clean_text'] = df['text'].map(clean_text).map(lemmatize_text)
    # Drop rows where clean_text is empty
    df = df[df['clean_text'].str.strip() != ""]
    
    keep_cols = ['id','clean_text','score','num_comments']
    df = df[keep_cols]

    scaler = MinMaxScaler()
    df[['score','num_comments']] = scaler.fit_transform(df[['score','num_comments']])
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(df.head(5))
    return df

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    for file in raw_dir.glob("*.csv"):
        output_file = processed_dir / f"{file.stem}_clean.csv"
        preprocess_file(file, output_file)
        print(f"Processed {file.name} saved at {output_file.name}")
    """
    input_file = Path("data/raw/anime_posts.csv")
    output_file = Path("data/processed/anime_posts_clean.csv")
    preprocess_file(input_file, output_file)
    print(f"Processed {input_file.name} saved at {output_file.name}")
    """
import praw
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_posts(subreddit_name, limit=500):
    print(f"Fetching {limit} posts from r/{subreddit_name}...")
    data = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=limit):
        data.append({
            "id" : post.id,
            "title" : post.title,
            "selftext" : post.selftext,
            "created_utc" : post.created_utc,
            "score" : post.score,
            "num_comments" : post.num_comments,
            "permalink" : f"https://reddit.com{post.permalink}"
        }) 
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset='id').reset_index(drop=True)
    df = df.head(limit)
    print(f"Fetched {len(df)} posts form r/{subreddit_name}")
    return df
    
def save_to_csv(df, subreddit_name, limit=500):
    raw_data_path = Path(__file__).parent.parent/"data"/"raw"
    raw_data_path.mkdir(parents=True, exist_ok=True)
    file_path = raw_data_path / f"{subreddit_name}_posts.csv"
    
    """if file_path.exists():
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df,df], ignore_index=True)
        df = df.drop_duplicates(subset='id').reset_index(drop=True)
    """
    #df = df.tail(limit)
    df = df.sort_values("created_utc").tail(limit).reset_index(drop=True)

    df.to_csv(file_path, index=False)
    print(f"Saved data to {subreddit_name} posts to {file_path}")

if __name__ == "__main__":
    subreddits = ["technology","anime","news","cricket","travel","personalfinance"]
    posts_limit = 500

    for sub in subreddits:
        df = fetch_posts(sub, limit=posts_limit)
        save_to_csv(df,sub,posts_limit)
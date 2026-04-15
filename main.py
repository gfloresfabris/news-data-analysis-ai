"""
News Data Analysis & AI Summarization

This project extracts real-time news data from NewsAPI, cleans and processes
the data using pandas, applies AI-based summarization and topic classification
with OpenAI, and visualizes topic distribution with matplotlib.
"""

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def analyze_article(client: OpenAI, text: str) -> tuple[str, str]:
    """Summarize an article and classify it into a topic."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Summarize this article in 1 sentence AND classify it into ONE word:
(politics, crime, finance, tech, other)

Return EXACTLY like this:
summary: ...
topic: ...

Article:
{text}
"""
                }
            ]
        )

        output = response.choices[0].message.content.strip()

        summary = "N/A"
        topic = "other"

        for line in output.splitlines():
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("topic:"):
                topic = line.split(":", 1)[1].strip().lower()

        return summary, topic

    except Exception as e:
        print(f"Error analyzing article: {e}")
        return "Error generating summary", "other"


def main() -> None:
    """Run the news analysis pipeline."""
    if not NEWS_API_KEY:
        raise ValueError("Missing NEWS_API_KEY in .env file")

    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Fetch news data
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": 10,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "articles" not in data:
        print("Error from NewsAPI:", data)
        raise SystemExit

    # Create DataFrame
    df = pd.DataFrame(data["articles"])

    # Keep only needed columns
    df = df[["title", "description", "content"]].copy()
    df.dropna(inplace=True)

    # Limit rows while testing
    df = df.head(5)

    # Analyze articles
    results = df["content"].apply(lambda text: analyze_article(client, text))
    df["summary"] = results.apply(lambda x: x[0])
    df["topic"] = results.apply(lambda x: x[1])

    # Save dataset
    df.to_csv("news_analysis.csv", index=False)

    # Plot topic counts
    topic_counts = df["topic"].value_counts()
    topic_counts.plot(kind="bar", title="News Topics")
    plt.tight_layout()
    plt.savefig("news_topics.png")
    plt.show()

    # Print results
    print("\nTop Topics:\n")
    print(topic_counts)

    print("\nSample Results:\n")
    print(df[["title", "topic", "summary"]].head())


if __name__ == "__main__":
    main()
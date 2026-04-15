import requests
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get news data
url = "https://newsapi.org/v2/top-headlines"
params = {
    "apiKey": NEWS_API_KEY,
    "language": "en",
    "pageSize": 10
}

response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()

if "articles" not in data:
    print("Error from NewsAPI:", data)
    raise SystemExit

articles = data["articles"]

# Create DataFrame
df = pd.DataFrame(articles)

# Keep only needed columns
df = df[["title", "description", "content"]].copy()
df.dropna(inplace=True)

# Limit rows while testing
df = df.head(5)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_article(text):
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

# Apply analysis
results = df["content"].apply(analyze_article)
df["summary"] = results.apply(lambda x: x[0])
df["topic"] = results.apply(lambda x: x[1])

# Save output
df.to_csv("news_analysis.csv", index=False)

# Plot
df["topic"].value_counts().plot(kind="bar", title="News Topics")
plt.tight_layout()
plt.savefig("news_topics.png")
plt.show()

# Preview
print("\nTop Topics:\n")
print(df["topic"].value_counts())

print("\nSample Results:\n")
print(df[["title", "topic", "summary"]].head())
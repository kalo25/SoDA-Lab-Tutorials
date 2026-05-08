#################################################
#API Pulls for final project (news outlets)
#################################################
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup 

#########setting up API keys
import os
from dotenv import load_dotenv, dotenv_values
# loading variables from .env file
load_dotenv("C:/Users/karra/Desktop/Coding_work/keys.env") 
# accessing and printing value
print(os.getenv("currentsapi"))
key1 = os.environ["currentsapi"]

print(os.getenv("freenewsapi"))
key2 = os.environ["freenewsapi"]


####SECTION 1: FOR THE 2026 IRAN WAR

url = "https://api.currentsapi.services/v2/search" #free API service

url2= "https://api.freenewsapi.io/v1/news" #free API service #2
url3 = "https://api.freenewsapi.io/v1/details" #extracting news text

start = datetime(2026, 2, 28)
end = datetime(2026, 4, 27)

all_articles = []


keywords_list = ["Iran", "Strait of Hormuz","sanctions", "nuclear stockpile",
    "airstrikes", "Israel", "naval blockade", "Khamenei", "Operation Epic Fury",
    "Islamabad Talks", "nuclear program", "Iran War", "oil resources",
    "regime change", "ceasefire", "Persian Gulf"]


for keyword in keywords_list:
    print(f"\nSearching keyword: {keyword}")

    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(days=14), end)

        cursor = None  

        while True:
            params = {
                "keyword": keyword,
                "language": "en",
                "start": current_start.strftime("%Y-%m-%dT00:00:00Z"),
                "end": current_end.strftime("%Y-%m-%dT23:59:59Z"),
                "page_size": 50,
                "api_key": key1
            }


            try:
                res = requests.get(url2, params=params, timeout=10)

                if res.status_code == 429:
                    print("Rate limited — sleeping...")
                    time.sleep(5)
                    continue

                res.raise_for_status()
                data = res.json()

                articles = data.get("data", [])
                meta = data.get("meta", {})

                if not articles:
                    break

                all_articles.extend(articles)

                print(
                    f"{keyword} | {current_start.date()}–{current_end.date()} "
                    f"| fetched {len(articles)}"
                )
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print("Request failed:", e)
                time.sleep(3)
                continue

        current_start = current_end + timedelta(days=1)



#convert to df
df = pd.DataFrame(all_articles)
df = df.drop_duplicates(subset="url")

#combine text fields into one searchable string
df["text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("")
)

#filter for articles that mention Iran
df_filt = df[df["text"].str.contains(r"\bIran\b", case=False, na=False)]

df["published"].value_counts().sort_index()

#building column names + filtering out news sources
cols = ["title", "author", "published", "url", "description"]
allowed = "wsj|washingtonexaminer|realclearpolitics|dailywire|" \
"theepochtimes|drudgereport|foxnews|renewedright|thefp|nypost|reuters" \
"|thehill|forbes|nationalreview|westernjournal|theamericanconservative|" \
"bloomberg|breitbart|thedispatch|justthenews|cbn|ijr|newsweek|san"
df_filt = df[df_filt["url"].str.contains(allowed, na=False)]

df_filt = df_filt[[c for c in cols if c in df.columns]]
#change to df_filt when needed

print(df.to_string(index=False))

print("Total articles collected:", len(df_filt))
print(df_filt["url"].head(10))

from tabulate import tabulate

df = df.sort_values("published", ascending=False)
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

#filtered to Iran-only and selected news sources only
df_filt.to_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_filt.csv', index=False)
#filtered to Iran only, all news sources possible
df.to_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons.csv', index=False)


######################Extracting the article body text from URL's
df = pd.read_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons.csv')

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from urllib.parse import urljoin
import time
from concurrent.futures import ThreadPoolExecutor
from newspaper import Article
import trafilatura


def extract_full_text(url):
    # Try newspaper3k first
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text and len(article.text) > 200:
            return article.text
    except:
        pass

    #Fallback: trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        if text and len(text) > 200:
            return text
    except:
        pass

    #Last fallback: BeautifulSoup
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text if len(text) > 200 else None
    except:
        return None

with ThreadPoolExecutor(max_workers=10) as executor:
    df["full_text"] = list(executor.map(extract_full_text, df["url"]))
    time.sleep(2)


df['full_text'] = df['full_text'].str.replace(r"\s+", " ", regex=True)

# drop failed extractions
df = df[df['full_text'].notna()]
print(df.to_string(index=False))

df.to_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_full.csv', index=False)

#correct but now I have to drop the observations that got blocked by cloudflare/security
df = pd.read_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_full.csv')

BLOCK_PATTERNS = [
    "access denied", "you have been blocked", "request blocked",
    "unusual traffic", "verify you are human", "captcha",
    "cloudflare", "akamai", "perimeterx",
    "bot detection", "security check",
    "enable javascript and cookies", "JavaScript", "click the button",
    "you're not a robot"
]

pattern = "|".join(BLOCK_PATTERNS)

df_clean = df[
    df["full_text"].notna() &
    (df["full_text"].str.len() > 200) &
    (~df["full_text"].str.lower().str.contains(pattern, na=False))
]d

df["is_blocked"] = df["full_text"].str.lower().str.contains(pattern, na=False)
df_clean = df[~df["is_blocked"]]


print("Before:", len(df))
print("After:", len(df_clean))


df_clean.to_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_full_clean.csv', index=False)

#more cleaning
def fix_encoding(text):
    try:
        return text.encode("latin1").decode("utf-8")
    except:
        return text
    
df_clean["full_text"] = df_clean["full_text"].apply(lambda x: fix_encoding(x) if isinstance(x, str) else x)

import re
def clean_text(text):
    if not isinstance(text, str):
        return text

    replacements = {
        "â€œ": '"', "â€\x9d": '"',
        "â€˜": "'", "â€™": "'",
        "â€“": "-", "â€”": "-",
        "…": "...",
        "\xa0": " "  # non-breaking space
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # remove leftover HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

df_clean["full_text"] = df_clean["full_text"].apply(clean_text)

df_clean.to_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_full_clean.csv', index=False)

##################################################################
###adding conservative vs liberal labels
OG_df = pd.read_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_full_labelled.csv')
from urllib.parse import urlparse

def extract_domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except:
        return None

OG_df["domain"] = OG_df["url"].apply(extract_domain)

def clean_domain(domain):
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain

OG_df["domain_clean"] = OG_df["domain"].apply(clean_domain)

#labelling
bias = {
    "foxnews.com": "conservative",
    "breitbart.com": "conservative",
    "wsj.com": "centrist",
    "nytimes.com": "liberal",
    "washingtonpost.com": "liberal",
    "cnn.com": "liberal",
    "msnbc.com": "liberal",
    "washingtonexaminer.com": "conservative",
    "dailywire.com": "conservative",
    "realclearpolitics.com": "conservative",
    "apnews.com": "liberal",
    "thehill.com": "centrist",
    "theamericanconservative": "conservative",
    "theepochtimes.com": "conservative",
    "bloomberg.com": "centrist",
    "ijr.com": "conservative",
    "drudgereport.com": "conservative",
    "thefp.com": "conservative",
    "nypost.com": "conservative",
    "reuters.com": "centrist",
    "vox.com": "liberal",
    "latimes.com": "liberal",
    "justthenews.com": "conservative",
    "cbn.com": "conservative",
    "newsweek.com": "centrist",
    "westernjournal.com": "conservative",
    "renewedright.com": "conservative",
    "san.com": "conservative",
    "npr.org": "liberal",
    "cnbc.com": "liberal",
    "theintercept.com": "liberal",
    "theatlantic.com": "liberal",
    "abcnews.com": "liberal",
    "cbsnews.com": "liberal",
    "politico.com": "liberal",
    "foreignpolicy.com": "centrist",
    "nbcnews.com": "liberal",
    "hotair.com": "conservative",
    "newsbusters.org": "conservative",
    "chicagotribune.com": "conservative",
    "theguardian.com": "liberal",
    "pbs.org": "liberal",
    "washingtontimes.com": "conservative",
    "truthout.org": "liberal",
    "zerohedge.com": "conservative",
    "usatoday.com": "centrist",
    "huffpost.com": "liberal",
    "forbes.com": "centrist",
    "nasdaq.com": "other",
    "nakedcapitalism.com": "liberal",
    "deadline.com": "liberal",
    "newsday.com": "centrist",
    "timesunion.com": "liberal",
    "fortune.com": "centrist",
    "thefederalist.com": "conservative",
    "yahoo.com": "liberal",

}


OG_df["label"] = OG_df["domain_clean"].map(bias)
OG_df["label"] = OG_df["domain"].map(bias)
OG_df["label"] = OG_df["label"].fillna("foreign")

foreign_domains = OG_df.loc[OG_df["label"] == "foreign"].value_counts()
print(foreign_domains.head(20))

OG_df.head()

OG_df.to_csv('C:/Users/karra/Desktop/Coding_work/soda_501/final_project/iran_cons_full_labelled.csv', index=False)

##################################################################

#######PART 2: 2001 WAR ON TERROR
#API's don't go this far back. have to use a historical dataset. GDELT Project.
#nothing worked. scrap 2001 timeframe. use freenewsapi
import random

#start = datetime(2001, 9, 14)
#end = datetime(2001, 11, 15)

start = datetime(2026, 2, 28)
end = datetime(2026, 4, 27)

#keywords_list = [
#    "AUMF", "Operation Enduring Freedom", "Operation Anaconda",
#    "Al Qaeda", "terrorism", "terrorists",
#    "Patriot Act", "George W Bush", "extraordinary rendition",
#    "human rights", "black sites", "September 11",
#    "9/11", "missile strikes"
#]

#queries = (
#    "terrorism OR \"Al Qaeda\" OR 9/11 OR Bush",
#    "\"Patriot Act\" OR Taliban OR \"Operation Enduring Freedom\"",
#     "terrorists OR \"black sites\" OR \"September 11\"",
#    "\"September 11\" OR \"missile strikes\" OR \"extraordinary rendition\""
#)

queries = (
    "Iran AND Israel OR Iran AND sanctions OR Khamenei OR airstrikes OR \"Iran War\"",
    "ceasefire OR \"oil resources\" OR \"Persian Gulf\" OR \"naval blockade\"",
    "\"Strait of Hormuz\" OR \"regime change\" OR \"Operation Epic Fury\"",
    "\"Islamabad Talks\" OR \"Iran AND \"military intervention\""
    )
   
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

#------------------------
#bypass rate limit
#------------------------
def fetch_with_retry(url, params, max_retries=5):
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params=params, timeout=20)

            if res.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited. Sleeping {wait}s...")
                time.sleep(wait)
                continue

            res.raise_for_status()

            if not res.text.strip():
                print("Empty response — retrying...")
                time.sleep(3)
                continue

            return res.json()

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

    return None



#for keyword in keywords_list:
    print(f"\nSearching keyword: {keyword}")

for query in queries:

    current_start = start

    while current_start < end:

        current_end = min(current_start + timedelta(days=15), end)

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": current_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": current_end.strftime("%Y%m%d%H%M%S"),
            "maxrecords": 100
        }

        data = fetch_with_retry(GDELT_URL, params)

        if data and "articles" in data:
            articles = data["articles"]

            if articles:
                all_articles.extend(articles)
                print(f"Fetched {len(articles)} articles ({current_start.date()} → {current_end.date()})")

        time.sleep(random.uniform(2, 5))

     
        current_start = current_end + timedelta(days=1)

df_gdelt = pd.DataFrame(all_articles)
df_gdelt = df_gdelt.drop_duplicates(subset="url")

print("Total articles collected": len(df_gdelt))

df_gdelt["text"] = (
    df_gdelt["title"].fillna("") + " " +
    df_gdelt["date"].fillna("")
)

#extracting full article text
from newspaper import Article

def extract_text(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return a.text
    except:
        return None

df_gdelt["full_text"] = df_gdelt["url"].apply(extract_text)
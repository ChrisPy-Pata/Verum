# required libraries/packages
import re
import streamlit as st
import requests
import os
import sys
import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from langchain.agents import tool
from langchain.tools import tool
from langchain_core.documents import Document

from keybert import KeyBERT
import spacy

from dotenv import load_dotenv
import pandas as pd
import httpx
from langchain_groq import ChatGroq

# used for storing the results for each agent step
display_results = []
news_articles = []

# set the event loop policy for Windows to avoid issues with asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# load environment variables
load_dotenv(".env")
# you can switch to a different model by changing the MODEL_NAME variable
MODEL_NAME = "llama3-70b-8192"
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
BRAVEAPI_KEY = os.environ.get("BRAVE_API")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NEWS_PAGE_SIZE = 10
FALLACY_CSV = "fallacies.csv"
fallacies_df = pd.read_csv(FALLACY_CSV)
FALLACIES_STR = fallacies_df.to_string(index=False)

# load spaCy model with error handling

nlp = spacy.load("en_core_web_sm")

url_pattern = r"https?://(?:www\.)?\S+"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# NOTE: In some functions, I used asyncio.run instead of await to avoid making the whole function async too.

# utility functions

# use Brave Search to find news articles related to the user's input keywords.
async def brave_search_news(api_key, user_input):
    """Use Brave Search to find news articles related to the user's input keywords."""
    query = user_input
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/news/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "x-subscription-token": api_key
                },
                params={
                    "q": query + "News from Philippines",
                    "search_lang": "en",
                    "ui_lang": "en-US",
                    "country": "PH",
                    "safesearch": "off",
                    "count": "5",
                    "spellcheck": "true",
                    "freshness": "pm",
                    "extra_snippets": "false"
                }
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        st.error(f"Brave API request failed: {e}")
        return []
    return data.get("results", [])

# extract keywords and named entities from text using KeyBERT and spaCy
def extract_keywords_and_entities(text, num_keywords=6):
    """Extract keywords and named entities from text using KeyBERT and spaCy."""
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    keywords = [kw[0] for kw in kw_model.extract_keywords(
        text, keyphrase_ngram_range=(2, 3), stop_words='english', top_n=num_keywords)]
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "GPE", "PERSON", "EVENT")]
    combined = list(dict.fromkeys(keywords + entities))
    return combined

# convert news articles to LangChain Document objects
def news_to_docs(news_articles):
    """Convert a list of news articles to LangChain Document objects."""
    docs = []
    for article in news_articles[:5]:
        text = f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}\nContent: {article.get('content', '')}\nSource: {article.get('source', {}).get('name', '')}\nURL: {article.get('url', '')}\npublishedAt: {article.get('publishedAt', '')}"
        if len(text) > 800:
            text = text[:800] + "..."
        docs.append(Document(page_content=text, metadata={"source": "newsapi", "url": article.get("url", "")}))
    return docs

# fetch news articles from NewsAPI based on the query or keywords
def fetch_news_articles(api_key, query):
    """Fetch news articles from the NewsAPI based on the query or keywords."""
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"q={query}&country=ph&pageSize={NEWS_PAGE_SIZE}&apiKey={api_key}"
    )
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "ok":
        return []
    return data.get("articles", [])

# filter news articles for relevance to the user input using llm model
def filter_relevant_articles(user_input: str, docs: list) -> list:
    """Filter news articles for relevance to the user input using LLM."""
    if not docs:
        return []
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=700,
    )
    prompt = (
        f"User's question/claim: {user_input}\n\n"
        "Below is a list of news articles. For each one, reply YES if it is relevant to answering or fact-checking the user's question, or NO if not."
        "Format your answer as a numbered list with YES or NO and a very short reason if YES.\n"
    )
    for idx, doc in enumerate(docs, 1):
        # limit to 500 characters and remove newlines
        short_text = doc.page_content[:500].replace('\n', ' ')
        prompt += f"{idx}. {short_text}\n\n"
    response = llm.invoke(prompt)
    # parse the response to extract relevant documents
    response_text = response.content if hasattr(response, "content") else str(response)
    relevant_docs = []
    lines = response_text.splitlines()
    # seperate documents by line and check if they contain the index number and "yes"
    for idx, line in enumerate(lines, 1):
        line_clean = line.lower().replace(")", ".").replace(":", ".").strip()
        num_str = f"{idx}."
        if num_str in line_clean and "yes" in line_clean.split(num_str, 1)[1]:
            relevant_docs.append(docs[idx-1])
    return relevant_docs or docs

# tools functions

# scrape the content of a news article from the given URL
@tool
async def news_scrape(url: str) -> str:
    """Scrape the content of a news article from the given URL. Returns the content as a string."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        return result
    
    
# scrape the content of a link and summarize it
@tool
def summarize_link_content(content: str) -> str:
    """Summarize the scraped content to get the main claim or topic."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=300,
    )
    prompt = (
        "Summarize this content in 2-3 sentences:\n"
        "Content: {content}\n\nSummary:"
    )
    return llm.invoke(prompt.format(content=content))

# fetch news articles based on keywords extracted from the link summary and/or user input
@tool
async def get_news(keywords: str) -> str:
    """Fetch PHILIPPINES news articles from NewsAPI or Brave Search based on extracted keywords."""
    news_articles = fetch_news_articles(NEWSAPI_KEY, keywords)
    if not news_articles:
        news_articles = await brave_search_news(BRAVEAPI_KEY, keywords)
    docs = news_to_docs(news_articles)
    docs_keywords = filter_relevant_articles(keywords, docs)
    
    url_news = []
    page_content = []
    
    # for each article, get page content and url
    for doc in docs_keywords:
        page_content.append(doc.page_content)
        if doc.metadata and "url" in doc.metadata:
            url_news.append(doc.metadata["url"])
    
    # store each url in display_result for ai model to access
    if url_news:
        display_results.append({"news_sources": url_news})
    
    return "\n\n".join(page_content)

# summarize the fetched news articles
@tool
def summarize_news(content: str) -> str:
    """Summarize all fetched news articles together."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=400,
    )
    prompt = (
        "You are a communication expert. Summarize the news articles in 3-4 clear sentences.\n"
        "Articles: {content}\n\nSummary:"
    )
    return llm.invoke(prompt.format(content=content))

# analyze the summary for logical fallacies and ethical issues
@tool
def analyze_summary(summary: str, fallacies: str) -> str:
    """Analyze the summary for logical fallacies and ethical issues."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=400,
    )
    prompt = ("You are an ethics professor analyzing this article summary. Review using the fallacy definitions again. Be succinct and easy to read, but intelligently draw upon all ethics rules. Provide:\n"
              "1. The single most impactful fallacy found out of fallacies in the summary and its description\n"
              "2. Why this fallacy might mislead readers\n"
              "3. One possible alternative interpretation of why the fallacy could have been included, as a counterfactual to finding the key fallacy\n\n"
              "Article summary: {summary}\nFallacies to consider:\n{fallacies}\n\nProfessor:\n")
    
    return llm.invoke(prompt.format(summary=summary, fallacies=fallacies))


# compare the user's claim with the news summary and analysis
@tool
def compare_claim(user_input: str, summary: str, analysis: str, link_content: str) -> str:
    """Compare the user's original claim and the summary of the link content with the news summary and analysis to determine if the claim is supported, contradicted, or inconclusive."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=500,
    )
    
    prompt = (
        "You are a fact-checking expert. Given the user's original claim, link content if URL was found on user input, the news summary, and the fallacy/ethics analysis, decide if the news evidence supports, contradicts, or is inconclusive about the user's claim. Write in 5 sentences and give an explanation.\n"
        "Be concise and clear.\n\n"
        "User's claim: {user_input}\n"
        "Link content: {link_content}\n"
        "News summary: {summary}\n"
        "Fact-check analysis: {analysis}\n\n"
        "Your verdict (Supported, Contradicted, Inconclusive, Under Investigation, Ongoing) Indicate what is being discussed in 5 clear sentences with detailed explanation about the link content, news summary, and user claim. please cross-check the link content and news summary:\n"
        "follow this format:"
        "1. Verdict: **[Supported/Contradicted/Inconclusive]**\n"
        "2. Explanation: [Provide a clear explanation of the reasoning behind the verdict."
        "3. Key points from cross checking the link content and news summary that support your verdict.\n"
        
    )
    
    return llm.invoke(prompt.format(
        user_input=user_input,
        link_content=link_content,
        summary=summary,
        analysis=analysis
    ))

# agent as tool

# keywords agent - this agent extracts keywords and named entities from the input text using KeyBERT and spaCy
@tool
def keywords_agent_tool(text: str) -> str:
    """Extracts keywords and named entities from the input text using KeyBERT and spaCy."""
    keywords = extract_keywords_and_entities(text)
    return ", ".join(keywords)


# scrape agent - this agent scrapes a URL from user input(w/wo URL) and summarizes its content
# If no link is found, it summarizes the user input instead.
# This is the first step in the workflow.
@tool
def scrape_agent_tool(user_input: str) -> str:
    """Scrape a URL from user input and return a summary of its content. If no link, summarize the user input."""
    match = re.search(url_pattern, user_input)
    # if no match, summarize the user input directly
    if not match:
        summary = summarize_link_content.invoke({"content": user_input[:2000]})  # Limit user input
        summary_text = summary["content"] if isinstance(summary, dict) and "content" in summary else (
            summary.content if hasattr(summary, "content") else str(summary)
        )
        display_results.append({"user_query_summary": summary_text})
        return summary_text
    #if url found, scrape url content
    url = match.group(0)
    article = asyncio.run(news_scrape.ainvoke(url))
    ext_html_text = str(article)
    ext_soup = BeautifulSoup(ext_html_text, 'html.parser')
    article_text = ext_soup.get_text(separator='\n', strip=True)
    article_text = article_text[:2000]  # limit to 2000 characters
    summary = summarize_link_content.invoke({"content": article_text})
    summary_text = summary["content"] if isinstance(summary, dict) and "content" in summary else (
        summary.content if hasattr(summary, "content") else str(summary)
    )
    display_results.append({"user_query_summary": summary_text})
    return summary_text

# news agent - this agent fetches and summarizes PH news articles based on keywords extracted from the link summary and/or user input
@tool
def news_agent_tool(keywords: str) -> str:
    """Fetch and summarize relevant PH news articles for given keywords. If no news, return an empty string."""
    news_content = asyncio.run(get_news.ainvoke(keywords))
    if not news_content or not str(news_content).strip():
        return ""
    news_content = news_content[:2000]  # limit to 2000 characters
    summary = summarize_news.invoke({"content": news_content})
    summary_text = summary["content"] if isinstance(summary, dict) and "content" in summary else (
        summary.content if hasattr(summary, "content") else str(summary)
    )
    display_results.append({"summary": summary_text})
    return summary_text

# analysis and verdict agent 

@tool
def analyze_and_verdict_agent_tool(user_input: str, link_summary: str, news_summary: str) -> str:
    """
    Analyze the news summary for fallacies and ethical issues, then immediately provide a verdict.
    If no news, base verdict on user input and link summary only.
    """
    if not news_summary or not str(news_summary).strip():
        verdict_text = "Inconclusive: No relevant news found. Based on the claim and available content, a verdict cannot be determined."
        display_results.append({"analysis_and_verdict": verdict_text})
        return verdict_text

    # Step 1: Analyze summary for fallacies
    analysis = analyze_summary.invoke({"summary": news_summary, "fallacies": FALLACIES_STR})
    analysis_text = analysis["content"] if isinstance(analysis, dict) and "content" in analysis else (
        analysis.content if hasattr(analysis, "content") else str(analysis)
    )
    display_results.append({"analysis": analysis_text})

    # Step 2: Decide verdict
    verdict = compare_claim.invoke({
        "user_input": user_input,
        "summary": news_summary,
        "analysis": analysis_text,
        "link_content": link_summary
    })
    result = verdict.content if hasattr(verdict, 'content') else (
        verdict["content"] if isinstance(verdict, dict) and "content" in verdict else str(verdict)
    )
    display_results.append({"verdict": result})
    
    # Step 3: Add references from news URLs
    references = ""
    for item in display_results:
        if "news_sources" in item:
            references = "\n**References:**\n"
            # use enumerate to number the news sources
            for i, url in enumerate(item["news_sources"], 1):
                # in bullet form
                references += f"• {url}\n"
            break

    # combine outputs
    combined = f"**Analysis:**\n{analysis_text}\n\n**Verdict:**\n{result}{references}"
    # display_results.append({"analysis_and_verdict": combined})
    return combined
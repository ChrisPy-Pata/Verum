# importing libraries/packages
import re
import streamlit as st
import requests
import logging

import os
import sys
import asyncio
import httpx
from crawl4ai import *

from bs4 import BeautifulSoup


from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# extracting keywords model
from keybert import KeyBERT
import spacy

from dotenv import load_dotenv

import pandas as pd


# to support Windows Proactor event loop -> asyncio on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are Verum, an AI assistant that helps users fact-check news and online claims. Your workflow is as follows:
        
        - If the user input contains a link, first scrape the link content (using the news_scrape tool), else proceed extracting keyword form user_input and get_news.
        - If the link is a Facebook post, extract any external news article link from the post and scrape that article if found; otherwise, extract the main post text.
        - Summarize the scraped content (using summarize_link_content) to get the main claim or topic. 
        - Use only the summary of the link content (not the raw HTML or full scraped text) as the main context for fact-checking (get summarize content from the summarize_link_content)  and evidence comparison.
        - Extract keywords from the summary and/or from the user's question (excluding the link itself).
        - Use the combined keywords to fetch relevant news articles (using get_news).
        - Summarize all fetched news articles together (using summarize_news).
        - Analyze the summary for logical fallacies and ethical issues (using analyze_summary).
        - Compare the user's original claim and the summary of the link content with the news summary and analysis to determine if the claim is supported, contradicted, or inconclusive (using compare_claim).
        Always use the provided tools for each step. Be clear, accurate, and never make things up.
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{{user_input}}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# store all chat history in this list
chat_history = []
url_pattern = r"https?://(?:www\.)?\S+"

load_dotenv(".env")
# logging setup
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
BRAVEAPI_KEY = os.environ.get("BRAVE_API")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

NEWS_PAGE_SIZE = 10
FALLACY_CSV = "fallacies.csv"

fallacies_df = pd.read_csv(FALLACY_CSV)
FALLACIES_STR = fallacies_df.to_string(index=False)

nlp = spacy.load("en_core_web_sm")

async def brave_search_news(api_key, user_input):
    """
    Use Brave Search to find news articles related to the user_input.
    Returns a list of relevant news based on query.
    """
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
                    "q": query,
                    "search_lang": "en",
                    "ui_lang": "en-US",
                    "country": "PH",
                    "safesearch": "off",
                    "count": "10",
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

    articles = data.get("results", [])
    if not articles:
        st.warning("No news articles found for your query on Brave Search.")
    else:
        logging.info(f"Fetched {len(articles)} articles from Brave Search.")

    return articles
    
    
def extract_keywords_and_entities(text, num_keywords=4):
    # KeyBERT keywords
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    keywords = [kw[0] for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 3), stop_words='english', top_n=num_keywords)]
    
    for keyword in keywords:
        st.write("keyword: ", keyword)
    
    # spaCy named entities
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "GPE", "PERSON", "EVENT")]
    
    for entity in entities:
        st.write("entity", entity)
    
    # Combine and deduplicate
    
    combined = list(dict.fromkeys(keywords + entities))  # Limit to 120 characters
    return combined

    

def news_to_docs(news_articles):
    
    # convert news articles to Document objects
    docs = []
    
    # fetch the title, description, content, source, url, and publishedAt from each article
    for article in news_articles:
        text = f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}\nContent: {article.get('content', '')}\nSource: {article.get('source', {}).get('name', '')}\nURL: {article.get('url', '')}\npublishedAt: {article.get('publishedAt', '')}"
        docs.append(Document(page_content=text, metadata={"source": "newsapi", "url": article.get("url", "")}))
    return docs

#TODO: FIND NEW DATA APIs
def fetch_news_articles(api_key, query):
    
    """Fetch news articles from the NewsAPI based on the query."""
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"q={query}&country=ph&pageSize={NEWS_PAGE_SIZE}&apiKey={api_key}"
    )
    
    # get the news articles from the API and convert it to json
    response = requests.get(url)
    data = response.json()
    
    # check if the request was successful
    if data.get("status") != "ok":
        st.error(f"Failed to fetch news articles: {data.get('message')}")
        return []
    
    articles = data.get("articles", [])
    logging.info(f"Fetched {len(articles)} articles from NEWSAPI.")
    
    return articles

@tool(parse_docstring=True)
async def news_scrape(url:str):
    """
    Scrape the content of a news article from the given URL.
    Returns the content in markdown format.

    Args:
        url: The URL of the news article to scrape.

    Returns:
        result of the url content markdown.

    Example:
        news_scrape(url="https://www.abs-cbn.com/news/nation/2025/6/30/vico-sotto-says-he-won-t-run-in-2028-elections-1115")
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        return result

@tool(parse_docstring=True)
def summarize_link_content(content: str) -> str:
    """
    Summarize the content of a scraped link to a concise paragraph for keyword extraction.

    Args:
        content: The raw HTML/text content scraped from the link.

    Returns a concise summary string.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=700,
    )
    
    prompt = (
        "You are a helpful assistant. Summarize the following article or web content in 3-5 sentences, focusing on the main claim or topic.\n"
        "Content: {content}\n\nSummary:"
    )
    return llm.invoke(prompt.format(content=content))

def filter_relevant_articles(user_input: str, docs: list) -> list:
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
        "Below is a list of news articles. For each one, reply YES if it is relevant to answering or fact-checking the user's question, or NO if not. "
        "Format your answer as a numbered list with YES or NO and a very short reason if YES. Example:\n"
        "1. YES - reason...\n"
        "2. NO\n"
        "3. YES - reason...\n\n"
    )
    for idx, doc in enumerate(docs, 1):
        short_text = doc.page_content[:500].replace('\n', ' ')
        prompt += f"{idx}. {short_text}\n\n"

    response = llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))

    # Debug print to see LLM output, uncomment for troubleshooting
    # print("LLM output:\n", response_text)

    #  match "N. YES" or "N) YES" or "N - YES"
    relevant_docs = []
    lines = response_text.splitlines()
    for idx in range(1, len(docs) + 1):
        pattern = re.compile(rf"^{idx}[\.\)\-]?\s*YES", re.IGNORECASE)
        for line in lines:
            if pattern.match(line.strip()):
                relevant_docs.append(docs[idx-1])
                break
    return relevant_docs



@tool(parse_docstring=True)
async def get_news(user_input: str) -> str:
    """
    Fetch news articles directly from the NewsAPI related/based on the extracted keywords.
    If NewsAPI returns no results, it falls back to Brave Search.
    Returns a list of PHILIPPINES news articles as Document objects.

    Args:
        user_input: string of user input to search for in the news articles.

    Returns:
        List[Document]: A list of Document objects containing the news articles.

    Example:
        fetch_news_articles(keywords="philippines, typhoon, disaster")
    """
    
    news_articles = fetch_news_articles(NEWSAPI_KEY, user_input)
    
    st.write("Fetched NEWSAPI articles:", news_articles)
    
    # if no relevant articles are found in NewsAPI, use Google Search
    if not news_articles:
        
        news_articles = await brave_search_news(BRAVEAPI_KEY,user_input)
        st.write("Fetch BRAVESEARCH API articles:", news_articles)
        # return news_articles
    
    docs = news_to_docs(news_articles)
    docs_keywords = filter_relevant_articles(user_input, docs)
    st.write("Relevant Articles (LLM-filtered):", docs_keywords)
    
    news_url = []
    
    for doc in docs_keywords:
        if doc.metadata and "url" in doc.metadata:
            news_url.append(doc.metadata["url"])
    
    st.write(news_url)        
    
    return docs_keywords

@tool(parse_docstring=True)
def summarize_news(content: str, fallacies: str) -> str:
    """
    Summarize news articles and check for fallacies.

    Args:
        content: The combined news article content.
        fallacies: The fallacy definitions.

    Returns a summary string.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=700,
    )
    
    prompt = ("You are a communication expert. Given the found relevant news articles, summarize it in five very clear sentences and be accurate.\n"
            "Do not make things up. Check to make sure you are correct. Think step by step.\n\n"
            "Fallacies to check:\n{fallacies}\n\nArticles: {content}\n\nCommunication expert:\nSummary:\n")
    summarize_result = llm.invoke(prompt.format(content=content, fallacies=fallacies))
    return summarize_result

@tool(parse_docstring=True)
def analyze_summary(summary: str, fallacies: str) -> str:
    """
    Analyze a summary for fallacies and provide an ethical review.

    Args:
        summary: The summary to analyze.
        fallacies: The fallacy definitions.

    Returns an analysis string.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=700,
    )
    
    prompt = ("You are an ethics professor analyzing this article summary. Review using the fallacy definitions again. Be succinct and easy to read, but intelligently draw upon all ethics rules. Provide:\n"
              "1. The single most impactful fallacy found out of fallacies in the summary\n"
              "2. Why this fallacy might mislead readers\n"
              "3. One possible alternative interpretation of why the fallacy could have been included, as a counterfactual to finding the key fallacy\n\n"
              "Article summary: {summary}\nFallacies to consider:\n{fallacies}\n\nProfessor:\n")
    analyze_result = llm.invoke(prompt.format(summary=summary, fallacies=fallacies))
    return analyze_result

@tool(parse_docstring=True)
def compare_claim(user_input: str, summary: str, analysis: str, link_content: str) -> str:
    """
    Compare the user's original claim/query and the content_link with the news summary and analysis to determine if the claim is supported, contradicted, or inconclusive.

    Args:
        user_input: The user's original query or claim.
        summary: The combined news summary.
        analysis: The fallacy/ethics analysis of the summary.

    Returns a verdict string (Supported, Contradicted, Inconclusive) with a short explanation.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=700,
    )
    
    prompt = (
        "You are a fact-checking expert. Given the user's original claim, the news summary, and the fallacy/ethics analysis, decide if the news evidence supports, contradicts, or is inconclusive about the user's claim.\n"
        "Be concise and clear.\n\n"
        "User's claim: {user_input}\n"
        "Link content: {link_content}\n"
        "News summary: {summary}\n"
        "Fact-check analysis: {analysis}\n\n"
        "Your verdict (Supported, Contradicted, Inconclusive) and an explanation:\n"
    )
    return llm.invoke(prompt.format(user_input=user_input, link_content=link_content, summary=summary, analysis=analysis))



tools_list = [  news_scrape, 
                summarize_link_content, 
                get_news, 
                summarize_news, 
                analyze_summary, 
                compare_claim
            ]


async def main(chat_history):
    st.title("Verum v0.5: AI News Assistance and Fact-Checking")
    if not BRAVEAPI_KEY:
        st.error("NO API KEY FOUND")
        return
    else:
        st.write("API KEY FOUND")
    user_input = st.text_input("Enter your news query:")
    html_content = ""
    combined_keywords = []
    if user_input:
        with st.spinner("Generating response..."):
            
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=MODEL_NAME,
                temperature=0.1,
                max_tokens=700,
            )
            
            ManagerAgent = create_tool_calling_agent(
                llm,
                tools_list,
                prompt,
            )
            ManagerAgent_executor = AgentExecutor(
                agent=ManagerAgent,
                tools=tools_list,
                verbose=True,  
            )
            match = re.search(url_pattern, user_input)
            if match:
                extracted_url = match.group(0)
                if extracted_url in user_input:
                    user_input_wo_url = user_input.replace(extracted_url, " ")
                else:
                    user_input_wo_url = user_input
                st.write("Extracted URL:", extracted_url)
                
                ext_article = await news_scrape.ainvoke(extracted_url)
                # Extract the HTML/text from ext_article
                if hasattr(ext_article, "content"):
                    ext_html_text = ext_article.content
                elif isinstance(ext_article, dict) and "content" in ext_article:
                    ext_html_text = ext_article["content"]
                else:
                    ext_html_text = str(ext_article)

                ext_soup = BeautifulSoup(ext_html_text, 'html.parser')
                result = ext_soup.find('article')
                if result:
                    html_content = str(result)
                else:
                    html_content = str(ext_soup.body)
                
                #  facebook post handling 
                if 'facebook.com' in extracted_url:
                    # Try to find external news link in the post
                    st.write("Detected Facebook post link...")
                    external_link = None
                    for a in ext_soup.find_all('a', href=True):
                        href = a['href']
                        if not href.startswith('https://www.facebook.com') and href.startswith('http'):
                            external_link = href
                            break
                    # if fb post has an external link
                    if external_link:
                        st.write(f"Found external news link in Facebook post: {external_link}")
                        ext_article = await news_scrape.ainvoke(external_link)
                        # Extract the HTML/text from ext_article
                        if hasattr(ext_article, "content"):
                            ext_html_text = ext_article.content
                        elif isinstance(ext_article, dict) and "content" in ext_article:
                            ext_html_text = ext_article["content"]
                        else:
                            ext_html_text = str(ext_article)

                        ext_soup = BeautifulSoup(ext_html_text, 'html.parser')
                        result = ext_soup.find_all('p')
                        if result:
                            html_content = str(result)
                        else:
                            html_content = str(ext_soup.body)
                        st.write("Scraped external article:", html_content)
                        summary_of_link = summarize_link_content.invoke({"content": html_content})
                        
                        # If summary_of_link is a dict or has a 'content' attribute, use only the summary text
                        if isinstance(summary_of_link, dict) and "content" in summary_of_link:
                            summary_text = summary_of_link["content"]
                        elif hasattr(summary_of_link, "content"):
                            summary_text = summary_of_link.content
                        elif isinstance(summary_of_link, str):
                            if summary_of_link.strip().startswith("content="):
                                match = re.search(r"content='([^']+)'", summary_of_link)
                                summary_text = match.group(1) if match else summary_of_link
                            else:
                                summary_text = summary_of_link
                        else:
                            summary_text = str(summary_of_link)
                        st.write("Summary of external article:", summary_text)
                        link_keywords = extract_keywords_and_entities(summary_text)
                        user_keywords = extract_keywords_and_entities(user_input_wo_url)
                        combined_keywords = list(dict.fromkeys(link_keywords + user_keywords))
                    # No external link, try to extract main post text
                    else:
                        st.write("No external link...")
                        post_text = ''
                        for tag in ext_soup.find_all(['p', 'div']):
                            txt = tag.get_text(strip=True)
                            if txt and len(txt) > 30:
                                post_text = txt
                                break
                        if not post_text:
                            st.warning("Could not extract meaningful text from the Facebook post. Fact-checking may be incomplete.")
                            post_text = ext_soup.get_text()[:500]
                        html_content = post_text
                        st.write("Extracted Facebook post text:", post_text)
                        summary_of_link = summarize_link_content.invoke({"content": post_text})
                        if isinstance(summary_of_link, dict) and "content" in summary_of_link:
                            summary_text = summary_of_link["content"]
                        elif hasattr(summary_of_link, "content"):
                            summary_text = summary_of_link.content
                        elif isinstance(summary_of_link, str):
                            if summary_of_link.strip().startswith("content="):
                                match = re.search(r"content='([^']+)'", summary_of_link)
                                summary_text = match.group(1) if match else summary_of_link
                            else:
                                summary_text = summary_of_link
                        else:
                            summary_text = str(summary_of_link)
                        st.write("Summary of Facebook post:", summary_text)
                        link_keywords = extract_keywords_and_entities(summary_text)
                        user_keywords = extract_keywords_and_entities(user_input_wo_url)
                        combined_keywords = list(dict.fromkeys(link_keywords + user_keywords))[:10]
                # Not a Facebook link, handle as before
                else:
                    st.write("Not a Facebook link: Extracting Link content...")
                    result = ext_soup.find_all('p')
                    if result:
                        html_content = str(result)
                    else:
                        html_content = str(ext_soup.body)
                    st.write("Scraped article:", html_content)
                    summary_of_link = summarize_link_content.invoke({"content": html_content})
                    if isinstance(summary_of_link, dict) and "content" in summary_of_link:
                        summary_text = summary_of_link["content"]
                    elif hasattr(summary_of_link, "content"):
                        summary_text = summary_of_link.content
                    elif isinstance(summary_of_link, str):
                        if summary_of_link.strip().startswith("content="):
                            match = re.search(r"content='([^']+)'", summary_of_link)
                            summary_text = match.group(1) if match else summary_of_link
                        else:
                            summary_text = summary_of_link
                    else:
                        summary_text = str(summary_of_link)
                    st.write("Summary of link content:", summary_text)
                    link_keywords = extract_keywords_and_entities(summary_text)
                    user_keywords = extract_keywords_and_entities(user_input_wo_url)
                    combined_keywords = list(dict.fromkeys(link_keywords + user_keywords))
                    
                    summary_of_link = summary_text
            else:
                st.write("No link found in user input. Extracting keywords from user input only...")
                combined_keywords = extract_keywords_and_entities(user_input)
                summary_of_link = user_input
                st.write("Extracted keywords from user input:", combined_keywords)
                query_string = " ".join(combined_keywords)
                st.write("Extracted keywords for search:", query_string)
                news_articles = await get_news.ainvoke(user_input)
                summary = summarize_news.invoke({"content": str(news_articles), "fallacies": FALLACIES_STR})
                analysis = analyze_summary.invoke({"summary": str(summary), "fallacies": FALLACIES_STR})
                verdict = compare_claim.invoke({"user_input": str(user_input), "summary": str(summary), "analysis": str(analysis), "link_content": str(summary_of_link)})
                with st.container(height=500, border=True):
                    st.markdown("**Combined News Summary:**")
                    st.write(summary)
                    st.markdown("**Fact-Check Analysis:**")
                    st.write(analysis)
                    st.markdown("**Final Verdict:**")
                    st.write(verdict)
                return
            
            st.write("Link content successfully scraped. Extracting keywords...")
            query_string = " ".join(combined_keywords)
            st.write("Extracted keywords for search:", query_string[:10])
            st.write("summary_of_link:", summary_of_link)
            news_articles = await get_news.ainvoke(query_string)
            summary = summarize_news.invoke({"content": str(news_articles), "fallacies": FALLACIES_STR})
            analysis = analyze_summary.invoke({"summary": str(summary), "fallacies": FALLACIES_STR})
            verdict = compare_claim.invoke({"user_input": str(user_input), "summary": str(summary), "analysis": str(analysis), "link_content": str(summary_of_link)})
            with st.container(height=500, border=True):
                st.markdown("**Combined News Summary:**")
                st.write(summary)
                st.markdown("**Fact-Check Analysis:**")
                st.write(analysis)
                st.markdown("**Final Verdict (Claim vs. News):**")
                st.write(verdict)
            return
                

if __name__ == "__main__":
    asyncio.run(main(chat_history))
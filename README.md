# Verum: AI News Fact-Checker

Verum is an AI-powered chatbot that helps users verify news claims and detect misinformation in Philippine news and social media posts. Built with a friendly journalist persona, Verum guides users through fact-checking processes using advanced language models.

## What is Verum?

Verum is a 39-year-old AI journalist with 10 years of fact-checking experience. It analyzes news claims, scrapes relevant articles, and provides clear verdicts on whether information is accurate, misleading, or inconclusive.

## Core Features

- **News Scraping**: Automatically extracts content from news URLs and summarizes key information
- **Keyword Extraction**: Uses KeyBERT and spaCy to identify relevant topics and entities
- **Fallacy Analysis**: Analyzes articles for logical fallacies and ethical issues using predefined criteria
- **Verdict Analysis**: Provides clear fact-check results (Supported/Contradicted/Inconclusive)
- **Interactive Chat**: Friendly conversation interface with typewriter effects

## How It Works

1. **Input Query or just chat with Verum**: Processes user queries and extracts URLs if present
2. **URL Content Scraping**: Retrieves and summarizes relevant content
3. **Keyword Extraction**: Identifies key terms for news searches
4. **News Gathering**: Fetches related Philippine news articles
5. **Analysis & Verdict**: Examines content for fallacies, provides verdict, and overall conclusion.

## Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **AI Models**: Mainly Groq's Llama 3 70B , but can also use other Groq's API models that supports tool/function tool calling.
- **Tools**: LangChain for agent orchestration and tool management
- **Web Scraping**: Crawl4AI for content extraction
- **NLP**: KeyBERT and spaCy for keyword and entity extraction
- **APIs**: NewsAPI and Brave Search for news data

## Why Fact-Checking Matters

In today's digital age, misinformation spreads rapidly through social media and news platforms.
Fact-checking helps:
- **Prevent Misinformation**: Stops false information from influencing public opinion
- **Promote Critical Thinking**: Encourages users to verify claims before sharing
- **Build Media Literacy**: Helps users identify reliable vs unreliable sources

## Getting Started

1. Clone the repository  
   ```powershell
   git clone https://github.com/ChrisPy-Pata/Verum
   ```

2. Set up environment variables in `.env` file  
   ```powershell
   # Example .env file
   GROQ_API_KEY=your_groq_api_key
   NEWSAPI_KEY=your_newsapi_key
   BRAVEAPI_KEY=your_braveapi_key
   ```

3. Install dependencies  
   ```powershell
   pip install -r requirements.txt
   ```

4. Run the application
   ```powershell
   streamlit run v1-with-ui.py
   ```

## Limitations

- Limited to available news sources and APIs
- Responses depend on data and model capabilities
- May not cover all news sources comprehensively

---

*Built to combat misinformation and promote factual news consumption in the Philippines.*

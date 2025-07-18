import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

from tools import scrape_agent_tool, keywords_agent_tool, news_agent_tool, analyze_and_verdict_agent_tool, display_results

load_dotenv(".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

# manager agent - this is the main agent that orchestrates the workflow
def make_manager_agent(tools):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=1000,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are Verum, a 39-year-old journalist at a top news company in the Philippines. 
        You have 10 years of experience in fact-checking and explaining news to the public. 
        You are friendly, professional, and sometimes a bit of a joker, especially when discussing news. 
        You love helping people, traveling, and cycling. 
        You always use clear and concise language, and occasionally use emojis to make the conversation lively.
        
        YOU MUST NOT call the user as "you" and refer to yourself as "I" or "Verum". Do not use "The user" when referring to the user, and do not refer to yourself as "the assistant" or "the agent".

        Your writing style is like a news reporter: clear, concise, and uses analogies to make difficult ideas accessible.
        
        DECISION LOGIC:
        First, determine the type of query:
        
        A) CASUAL CONVERSATION: Simple greetings, questions about yourself, general chitchat, personal questions
        - Respond directly without using any tools
        - Keep it friendly and conversational
        
        B) NEWS/FACT-CHECK QUERY: Claims to verify, news articles, current events, "Is it true that...", URLs, specific factual claims
        - Use ALL 4 tools in this exact order:
        1. Use scrape_agent_tool with user_input
        2. Use keywords_agent_tool with the link_summary from step 1
        3. Use news_agent_tool with the keywords from step 2  
        4. Use analyze_and_verdict_agent_tool with user_input, link_summary, and news_summary
        5. If news sources are available, include them as references with url and title in your final response.
        
        CRITICAL RULES FOR FACT-CHECKING:
        - You MUST call analyze_and_verdict_agent_tool as the final step for news queries
        - Even if any step returns empty results, continue to the next step
        - Do not provide a final answer until you have called all 4 tools in order
        - Complete ALL steps before responding to news/fact-check queries
        - If news_agent_tool returns empty, still proceed to analyze_and_verdict_agent_tool
        - If summaries are irrelevant to the user query, provide credible sources that user can refer to for more information in the conclusion step.
        
        
        Examples:
        - "Hello Verum" → Casual conversation (no tools)
        - "What's your name?" → Casual conversation (no tools)  
        - "Is it true that there's a typhoon coming?" → Fact-check query (use all 4 tools)
        - "Check this news: [URL]" → Fact-check query (use all 4 tools)
        """),
        ("human", "{user_input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return create_tool_calling_agent(llm, tools, prompt)


async def typewriter_effect(text, speed = 0.0002):
    """Typewriter effect for streaming AI output."""
    current = ""
    container = st.empty()
    for char in text:
        current += char
        container.markdown(current)
        
        await asyncio.sleep(speed)

async def main():
    
    st.set_page_config(page_title="Verum: AI News Fact-Checker",
                       page_icon="✔️",
                       layout="wide",
                       initial_sidebar_state="expanded")
    #title
    st.title("Verum: AI News Fact-Checker")

    # create sidebar to adjust model - FOR LATER
    with st.sidebar.title("About Verum"):
        st.expander("About Verum", expanded=False).markdown("""
        Verum is an AI-powered news fact-checking assistant that helps you verify the accuracy of news articles and claims. It uses advanced language models to analyze news content, extract key information, and provide verdicts on the veracity of claims.

        **Key Features:**
        - Scrapes news articles for relevant content
        - Extracts keywords and summaries
        - Analyzes news articles for factual accuracy
        - Provides verdicts on claims

        **How to Use:**
        1. Enter a news query or claim in the chat input.
        2. Verum will scrape relevant news articles and analyze them.
        3. It will provide a summary and verdict on the claim.
        
        You can also just have a conversation with Verum!

        **Limitations:**
        - May not have access to all news sources
        - Responses may be limited by available data
        """)
    
    
    # initialize message to keep previous messages/chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


    # update the chat interface with the previous messages
    for message in st.session_state['messages']:
        with st.chat_message(message['role'], avatar="🫠" if message['role'] == "assistant" else None):
            
            st.markdown(message['content'])

    prompt = st.chat_input("Ask Verum")
    
    # Show welcome message only if there are no messages and no prompt
    if not st.session_state['messages'] and not prompt:
        with st.chat_message("assistant", avatar="🫠"):
            await typewriter_effect("👋 **Hi, I'm Verum!** Your friendly news fact-checker. Ask me anything about the news, and I'll help you get to the facts! 📰🚴‍♂️")

    # create the chat interface
    if prompt:
        # add the user message to the chat history
        st.session_state['messages'].append({"role": "user", "content": prompt})
        
        # display the user message in the chat interface
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        
        # generate a response from the model
        with st.chat_message("assistant", avatar="🫠"):
            with st.spinner("Searching News..."):
                # display_results.clear()
                tools = [
                    scrape_agent_tool,
                    keywords_agent_tool,
                    news_agent_tool,
                    analyze_and_verdict_agent_tool,
                ]
                manager_agent = make_manager_agent(tools)
                manager_executor = AgentExecutor(
                    agent=manager_agent,
                    tools=tools,
                    verbose=True,
                )
                try:
                    display_results.clear()  # clear previous results
                    
                    #placeholder for the response text
                    full_response = ""
                    result_dict = {}
                    temp = []
                    
                    async for chunk in manager_executor.astream({"user_input": prompt}):
                        
                        # incrementally build the response for each finished tool step
                        if "output" in chunk:
                            full_response += chunk["output"]
                        
                        # display tool results after they are complete
                        with st.container(border=True):
                            for result in display_results:
                                    result_dict.update(result)
                                
                            for key, value in result_dict.items():
                                if key and value and value not in temp:
                                    # keep track of already displayed values
                                    temp.append(value)
                                    # for formatting, mkae a bullet for each URLs
                                    if key == "news_sources":
                                        st.subheader(key.replace("_", " ").title())
                                        for val in value:
                                            await typewriter_effect(f"- {val} \n")
                                    else:
                                        st.subheader(key.replace("_", " ").title())
                                        await typewriter_effect(str(value))
                                    
                    # finally, display the full response after all tools are done
                    if full_response:
                        await typewriter_effect(full_response)
                        st.session_state['messages'].append({"role": "assistant", "content": full_response})
                        st.write(agent)
                            
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)

if __name__ == "__main__":
    asyncio.run(main())
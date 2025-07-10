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
        
        YOU MUST always call the user as "you" and refer to yourself as "I" or "Verum". Do not use "The user" when referring to the user, and do not refer to yourself as "the assistant" or "the agent"

        Your writing style is like a news reporter: clear, concise, and uses analogies to make difficult ideas accessible. 
        Your goal is to help users of all ages in the Philippines understand their news queries and provide a verdict on whether a claim is fact or fake, following the step-by-step process below.
        
        You MUST complete ALL 5 steps in exact order. DO NOT STOP until all steps are completed:
        1. FIRST: Evaluate if user is trying to have a conversation with you, Verum, or is trying to fact-check a news article or claim. If it is a conversation, you can just chat with the user and answer their questions. If it is a fact-checking query, you must follow the steps below.
        2. SECOND: Use scrape_agent_tool with user_input
        3. THIRD: Use keywords_agent_tool with the summary from step 1
        4. FOURTH: Use news_agent_tool with the keywords from step 2
        5. FIFTH: Use analyze_and_verdict_agent_tool with user_input, link_summary, news_summary. this is REQUIRED step

        CRITICAL RULES:
        - You MUST call analyze_and_verdict_agent_tool as the final step 
        - Even if any step returns empty results, you must still call analyze_and_verdict_agent_tool
        - Do not provide a final answer until you have called all 5 tools in order
        - If news_agent_tool returns empty, still proceed to analyze_and_verdict_agent_tool
        - please provide the **References** of news url, IN BULLETS, where you BASED your news sources on your final response. The URLs can be found in the get_news url_news list.

        STOP ONLY after calling verdict_agent_tool.
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
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask Verum")
    
    # Show welcome message only if there are no messages and no prompt
    if not st.session_state['messages'] and not prompt:
        with st.chat_message("assistant"):
            await typewriter_effect("👋 **Hi, I'm Verum!** Your friendly news fact-checker. Ask me anything about the news, and I'll help you get to the facts! 📰🚴‍♂️")

    # create the chat interface
    if prompt:
        # add the user message to the chat history
        st.session_state['messages'].append({"role": "user", "content": prompt})
        
        # display the user message in the chat interface
        with st.chat_message("user"):
            st.markdown(prompt)

        
        # generate a response from the model
        with st.chat_message("assistant"):
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
                    response_container = st.empty()
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
                                    temp.append(value)
                                    st.subheader(key.replace("_", " ").title())
                                    await typewriter_effect(str(value))
                                    
                    # finally, display the full response after all tools are done          
                    if full_response:
                        st.markdown("### Conclusion")
                        await typewriter_effect(full_response)
                        st.session_state['messages'].append({"role": "assistance", "content": full_response})
                            
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)

if __name__ == "__main__":
    asyncio.run(main())
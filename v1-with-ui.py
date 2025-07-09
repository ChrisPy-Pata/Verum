import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

from v1_agent_as_tool import scrape_agent_tool, keywords_agent_tool, news_agent_tool, analyze_and_verdict_agent_tool, display_results

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
        ("system", """You are Verum, an AI fact-checking assistant. You MUST complete ALL 5 steps in exact order. DO NOT STOP until all steps are completed:

    1. FIRST: Use scrape_agent_tool with user_input
    2. SECOND: Use keywords_agent_tool with the summary from step 1
    3. THIRD: Use news_agent_tool with the keywords from step 2
    4. FOURTH: Use analyze_and_verdict_agent_tool with user_input, link_summary, news_summary. this is REQUIRED step

    CRITICAL RULES:
    - You MUST call analyze_and_verdict_agent_tool as the final step 
    - Even if any step returns empty results, you must still call analyze_and_verdict_agent_tool
    - Do not provide a final answer until you have called all 5 tools in order
    - If news_agent_tool returns empty, still proceed to analyze_and_verdict_agent_tool

    STOP ONLY after calling verdict_agent_tool."""),
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
    # container.markdown(current)

async def main():
    #title
    st.title("VERUM")

    # create sidebar to adjust model - FOR LATER
    st.sidebar.title("Adjust Models Options")


    # initialize message to keep previous messages/chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


    # update the chat interface with the previous messages
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


    # create the chat interface
    if prompt := st.chat_input("Ask Verum about your news query"):
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
                    #placeholder for the response text
                    response_container = st.empty()
                    full_response = ""
                    result_dict = {}
                    temp = []
                    
                    with st.container(border=True):
                        async for chunk in manager_executor.astream({"user_input": prompt}):
                            for result in display_results:
                                result_dict.update(result)
                            for key, value in result_dict.items():
                                if key and value not in temp:
                                    temp.append(value)
                                
                                    st.subheader(await typewriter_effect(key))
                                    st.markdown(await typewriter_effect(value))
                                    

                            if "output" in chunk:
                                full_response += chunk["output"]
                                response_container.markdown(full_response)
                                
                            if full_response:
                                response_container.markdown(full_response)
                                st.session_state['messages'].append({"role": "assistance", "content": full_response})
                            
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)

if __name__ == "__main__":
    asyncio.run(main())
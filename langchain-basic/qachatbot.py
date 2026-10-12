"""This is a simple chatbot that uses the LangChain library to generate responses to user messages.
Args:
    api_key (str): The API key for the Groq model
    model_name (str): The name of the Groq model
Returns:
    chain: The chain object
"""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# page config
st.set_page_config(page_title="LangChain QA Chatbot", page_icon=":robot_face:")

with st.sidebar:
    st.header("Settings")
    
    # API key
    api_key = st.text_input("API Key", type="password", help="Enter your API key ( Groq ( console.groq.com))")

    # Model Selection
    model_name = st.selectbox(
        "Model", [
            "qwen/qwen3.8-27b",  # Updated from llama3-8b-8192
            "meta-llama/llama-prompt-guard-2-22m", # Another great option
            "meta-llama/llama-prompt-guard-2-86m"
        ],
        index=0,
        )
    
    #clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
# Title
st.title("LangChain QA Chatbot")
st.markdown("This is a simple chatbot that uses the LangChain library to generate responses to user messages.")

# Example
st.markdown("---")
st.markdown("### Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**What is the capital of France?**")
    st.markdown("Explain Groq's LPU technology")
with col2:
    st.markdown("How do I learn programming?")
    st.markdown("Write a haiku about AI")
# Footer 
st.markdown("---")
st.markdown("Built with Langchain and Groq")

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None
    
    # Initialize the chat model
    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.7,
        streaming=True,
    )

    # create prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant powered by Groq.Answer questions clearly and concisely"),
            ("user", "{question}"),
        ]
    )
    
    # create chain
    chain = prompt_template | llm | StrOutputParser()
    return chain

# get chain
chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your API key and select a model")
    st.markdown("[Get your free API key from Groq](https://console.groq.com)")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# chat input
if question := st.chat_input("Ask me anything"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # stream response from LLM
            for chunk in chain.stream({"question": question}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})



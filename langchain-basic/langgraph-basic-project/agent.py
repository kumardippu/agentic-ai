import os


from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage,SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()
DOCS_DIR = "sample_docs"
INDEX_DIR = "faiss_index"

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers together"""
    return a / b    

@tool
def search_docs(query: str) -> str:
    """Search the LangChain documentation for the given query
    it contains the information about the LangChain framework, RAG , embeddings, and related topics
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

tools = [add, multiply, divide, search_docs]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0,
)

model_with_tools = model.bind_tools(tools)

def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the given messages"""
    response = model_with_tools.invoke(
        [SystemMessage(content= ("You are a helpful assistant that can perform arithmetic"
        "and answer querstions about AI/ML concepts"
        "use search_docs for AI/ML questions, math tools for calculations"
        ))]+ state["messages"]
        )
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


def tool_node(state: MessagesState) -> dict:
    """Use the tools to answer the question"""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": results}

def should_continue(state: MessagesState) -> Literal["tool_name", "__end__"]:
    """Decide whether to continue the conversation or end it"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
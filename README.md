# agentic-ai

A hands-on learning repo for building LLM apps and agents with **LangChain** and **LangGraph**. It goes from a first LLM call to a tool-using RAG agent with a Streamlit UI.

Everything runs on free or cheap hosted models. The default LLM is **Groq** and the embeddings come from **Google Gemini**.

## What's inside

| Path | What it is | Type |
|---|---|---|
| `langchain-basic/langchainbasics.ipynb` | LangChain fundamentals: model init, invoke, streaming, prompt templates, chains, conversational Q&A | Notebook |
| `langchain-basic/tools.ipynb` | Custom tools (`@tool`, async, `StructuredTool`), built-in tools (Wikipedia, Tavily search), and tool calling with an LLM | Notebook |
| `langchain-basic/langgraphgasics.ipynb` | LangGraph agent built step by step: FAISS index, tools, state, graph, run | Notebook |
| `langchain-basic/qachatbot.py` | Streamlit chatbot (LangChain + Groq), with the API key entered in the sidebar | App |
| `langchain-basic/langgraph-basic-project/` | The "final" project: LangGraph agent with calculator tools and RAG, plus a Streamlit UI that shows the tool trace | App |
| `langchain-basic/sample_docs/` | Text files used by the notebooks as the RAG knowledge base | Data |
| `langchain-basic/faiss_index/` | Prebuilt FAISS index for the notebooks | Generated |
| `main.py` | Placeholder from `uv init`. It does nothing useful. | - |
| `flow.excalidraw` | Empty diagram file | - |

Suggested reading order: `langchainbasics.ipynb`, then `tools.ipynb`, then `langgraphgasics.ipynb`, then the `langgraph-basic-project/` app.

## Prerequisites

- **Python 3.11** (pinned in `.python-version`; `pyproject.toml` requires `>=3.11`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management. Install it with `brew install uv` or `pip install uv`.
- API keys (all have free tiers):

| Key | Used for | Get it at |
|---|---|---|
| `GROQ_API_KEY` | Chat LLM (required for everything) | https://console.groq.com |
| `GOOGLE_API_KEY` | Gemini embeddings for RAG (required for `langgraph-basic-project` and `langgraphgasics.ipynb`) | https://aistudio.google.com/apikey |
| `TAVILY_API_KEY` | Web search tool (only `tools.ipynb`) | https://tavily.com |
| `OPENAI_API_KEY` | Not used by any current code. `langchain-openai` is installed but unused. | - |

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> agentic-ai
cd agentic-ai

# 2. Install dependencies into .venv (uses uv.lock, so versions match exactly)
uv sync

# 3. Create your .env file, then fill in your keys (see "Configuration" below)
cp .env.example .env

# 4. Run the main demo
cd langchain-basic/langgraph-basic-project
uv run streamlit run app.py
```

Streamlit opens at http://localhost:8501. Try:

- `What is 12 * 7, then add 5?` The agent calls the math tools.
- `What is RAG?` The agent calls `search_docs` and answers from the knowledge base.

The right sidebar shows every tool call, with its input and output, for each turn.

## Configuration

Create a `.env` file in the **repo root**. It is git-ignored, so never commit it.

```env
GROQ_API_KEY=your-groq-key
GOOGLE_API_KEY=your-google-key
TAVILY_API_KEY=your-tavily-key
OPENAI_API_KEY=            # optional, currently unused
```

`python-dotenv` finds the file by walking up from the current directory, so the repo-root `.env` works for scripts run from subfolders. `langchain-groq` and `langchain-google-genai` read their keys from the environment automatically.

Exception: `qachatbot.py` ignores `.env`. You paste the Groq key into the app's sidebar instead.

## Running each piece

### 1. Notebooks

```bash
uv run jupyter lab        # or open the .ipynb files in VS Code / Cursor
```

Select the **`.venv` (Python 3.11)** kernel. `ipykernel` is already a dependency, so no extra setup is needed.

Run the cells top to bottom. Some cells call paid or rate-limited APIs, so do not run-all repeatedly.

### 2. QA chatbot (`qachatbot.py`)

```bash
cd langchain-basic
uv run streamlit run qachatbot.py
```

Paste your Groq API key in the sidebar and pick a model.

### 3. LangGraph RAG agent (`langgraph-basic-project`)

This project has three files:

| File | Role |
|---|---|
| `ingest.py` | Reads `sample_docs/*.txt`, splits them into ~500-character chunks, embeds them with Gemini (`models/gemini-embedding-001`), and saves a FAISS index to `faiss_index/` |
| `agent.py` | Defines the tools (`add`, `multiply`, `divide`, `search_docs`) and the LangGraph graph, using Groq `openai/gpt-oss-20b` |
| `app.py` | Streamlit UI that calls the agent and shows the tool trace |

**You must run these from inside `langchain-basic/langgraph-basic-project/`.** `DOCS_DIR` and `INDEX_DIR` are relative paths (`sample_docs`, `faiss_index`), so running from anywhere else fails with a missing-index error.

```bash
cd langchain-basic/langgraph-basic-project

# Only needed if you change sample_docs/ or the faiss_index/ folder is missing
uv run python ingest.py

# Start the UI
uv run streamlit run app.py
```

A prebuilt `faiss_index/` is committed, so you can skip `ingest.py` on first run.

#### How the agent works

```
        ┌───────┐
START ─▶│llm_call│──── tool calls in last message? ──yes──▶ tool_node
        └───────┘                                              │
            ▲        no                                        │
            │        ▼                                         │
            │       END                                        │
            └──────────────────────────────────────────────────┘
```

1. `llm_call` sends the system prompt plus the conversation to Groq, with the tools bound.
2. If the model's reply contains tool calls, `tool_node` runs them and appends `ToolMessage`s.
3. The graph loops back to `llm_call` until the model answers without a tool call, then ends.

State is `MessagesState`: a `messages` list (appended with `operator.add`) and an `llm_calls` counter.

#### Adding your own knowledge

1. Drop `.txt` files into `langchain-basic/langgraph-basic-project/sample_docs/`.
2. Run `uv run python ingest.py`.
3. Restart the Streamlit app.

#### Adding a new tool

In `agent.py`, define a function with `@tool` and a docstring (the LLM reads the docstring to decide when to call it), then add it to the `tools` list:

```python
@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a"""
    return a - b

tools = [add, multiply, divide, search_docs, subtract]
```

## Project layout

```
agentic-ai/
├── pyproject.toml            # dependencies (source of truth)
├── uv.lock                   # locked versions
├── requirements.txt          # partial pip fallback, prefer uv (see Troubleshooting)
├── .python-version           # 3.11
├── .env                      # your API keys (git-ignored)
├── main.py                   # placeholder
└── langchain-basic/
    ├── langchainbasics.ipynb
    ├── tools.ipynb
    ├── langgraphgasics.ipynb
    ├── qachatbot.py
    ├── sample_docs/          # knowledge base for the notebooks
    ├── faiss_index/          # index for the notebooks
    └── langgraph-basic-project/
        ├── agent.py
        ├── app.py
        ├── ingest.py
        ├── sample_docs/      # knowledge base for the app
        └── faiss_index/      # index for the app
```

Note that `sample_docs/` and `faiss_index/` exist twice, once for the notebooks and once for the app. They are separate copies. If you edit one, the other does not change.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError` when using `pip install -r requirements.txt` | `requirements.txt` is missing `faiss-cpu`, `langchain-community`, `langchain-tavily` and `ipykernel`. Use `uv sync` instead. |
| `RuntimeError` or `AssertionError` about `GROQ_API_KEY` or `GOOGLE_API_KEY` | The `.env` file is missing or a key is empty. Check the file is in the repo root and the variable names match exactly. |
| `FileNotFoundError` for `faiss_index` or `index.faiss` | You ran from the wrong directory. `cd` into `langchain-basic/langgraph-basic-project/` first. |
| `OMP: Error #15: Initializing libomp...` on macOS | FAISS and another library both load OpenMP. The code already sets `KMP_DUPLICATE_LIB_OK=TRUE` at the top of each script. If you hit this in a new script or notebook, set it before importing FAISS. |
| `ValueError: ... allow_dangerous_deserialization` | FAISS stores metadata as a pickle. `agent.py` passes `allow_dangerous_deserialization=True`. Only load indexes you built yourself. |
| Streamlit shows a model-not-found or decommissioned error | Groq retires models regularly. Pick a current one from https://console.groq.com/docs/models and update the model name in `agent.py` / `qachatbot.py`. |
| `429` rate limit errors | Free tiers have per-minute and per-day limits. Wait and retry, or switch models. |
| Wrong kernel in notebooks | Pick the interpreter at `.venv/bin/python`. |

## Known issues

These are real bugs or rough edges in the current code. They are good first fixes if you're picking this up.

1. **`agent.py` `tool_node` only runs the first tool call.** The `return` is indented inside the `for` loop, so if the model requests two tools in one turn (for example "2*5 and what is RAG?"), the second one is dropped. Fix: dedent `return {"messages": results}` one level.
2. **`agent.py` `should_continue` type hint** says `Literal["tool_name", "__end__"]`, but it actually returns `"tool_node"`. It works at runtime, but the hint is wrong.
3. **`ingest.py` drops the last chunk of every file.** After the paragraph loop, the final `current` buffer is never appended. Fix: append it after the loop. Also remove the stray `import chunk`.
4. **`divide`** is typed as `-> int` but returns a float, and divides by zero without handling it.
5. **`app.py` caption says "powered by Claude"**, but the model is Groq `openai/gpt-oss-20b`.
6. **`app.py` only sends the latest prompt to the agent** (`initial_state` has one `HumanMessage`), so the agent has no memory of earlier turns even though the UI displays them.
7. **`qachatbot.py` model list** includes `llama-prompt-guard-2-*` models, which are safety classifiers, not chat models. They will not give useful answers. It also appends the assistant message to history twice on success.
8. **`requirements.txt` is incomplete** (see Troubleshooting). `pyproject.toml` / `uv.lock` are the source of truth.
9. **Notebooks contain stray auto-imports** (`from email import message`, `from pydoc import doc`, `import chunk`, and similar). They are harmless but can be deleted.

## Dependencies

Managed in `pyproject.toml`. The main ones:

- `langchain`, `langchain-community`, `langgraph`: framework and agent graph
- `langchain-groq`: Groq chat models
- `langchain-google-genai`: Gemini embeddings
- `langchain-openai`: installed, currently unused
- `langchain-tavily`: Tavily web search (used in `tools.ipynb`)
- `faiss-cpu`: local vector store
- `wikipedia`: Wikipedia tool (used in `tools.ipynb`)
- `streamlit`: UI
- `ipykernel`: notebook kernel
- `load-dotenv`: loads `.env`

Add a package with `uv add <package>`; this updates both `pyproject.toml` and `uv.lock`.

## Conventions

- Secrets live only in `.env`. It is git-ignored; do not commit keys, and rotate any key that was ever committed or shared.
- Use `uv run ...` so commands always use the project's `.venv`.
- Commit `uv.lock` together with any dependency change.
- The `.venv/` folder and `__pycache__/` are git-ignored.
- `.agents/skills/` and `.claude/skills/` hold symlinks into `.venv` (Streamlit's AI-assistant skill). They break until you run `uv sync`, which is expected.

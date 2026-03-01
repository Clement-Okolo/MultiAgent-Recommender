# 🎬 MultiAgent Recommender System

A conversational movie recommendation system powered by a **three-agent AI crew** that collaborates in real-time to deliver personalised film suggestions. Built with [CrewAI](https://docs.crewai.com/), [GroqCloud](https://console.groq.com/), [Redis](https://redis.io/), and [Streamlit](https://streamlit.io/).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Using the Notebook](#using-the-notebook)
- [How It Works](#how-it-works)
- [Dataset](#dataset)

---

## Overview

The MultiAgent Movie Recommender accepts a natural language query (e.g. *"feel-good comedies"*, *"mind-bending sci-fi"*) and passes it through a sequential pipeline of three specialised AI agents. Each agent has a distinct role — analysing preferences, matching candidate films, and generating a personalised reply — before the final recommendation is streamed back to the user through a polished chat interface.

Chat context is persisted in **Redis** so the system remembers earlier messages within a session, enabling follow-up queries like *"something similar but more recent"*.

---

## Architecture

```
User Input (Streamlit Chat)
        │
        ▼
┌═══════════════════════════════════════════════════════════════╗
║               Workflow Graph (LangGraph)                      ║
║                                                               ║
║                                                               ║
║                                                               ║
║                                                               ║
║        ┌─────────────────────────────────────────┐            ║
║        │  Node: "run_crew"                        │           ║
║        │                                          │           ║
║        │  ┌───────────────────────────────────┐  │           ║
║        │  │    Agent Orchestration (CrewAI)   │  │           ║
║        │  │                                   │  │           ║
║        │  │  ┌─────────────┐  ┌────────────┐  │  │           ║
║        │  │  │ Preference  │─▶│  Movie     │  │  │           ║
║        │  │  │ Analyst     │  │  Matcher   │  │  │           ║
║        │  │  │ (Agent 1)   │  │  (Agent 2) │  │  │           ║
║        │  │  └─────────────┘  └─────┬──────┘  │  │           ║
║        │  │                         │         │  │           ║
║        │  │               ┌─────────▼──────┐  │  │           ║
║        │  │               │ Recommendation │  │  │           ║
║        │  │               │ Generator      │  │  │           ║
║        │  │               │ (Agent 3)      │  │  │           ║
║        │  │               └─────────┬──────┘  │  │           ║
║        │  └─────────────────────────┼─────────┘  │           ║
║        └─────────────────────────────────────────┘           ║
║                                     │  result                 ║
║                                    END                        ║
╚═══════════════════════════════════════════════════════════════╝
                                     │
                    ┌────────────────▼────────────────┐
                    │      Redis (Cloud / Local)       │
                    │  • Vector DB (embeddings)     │
                    │  • LLM Cache                     │
                    │  • Chat Message History          │
                    └─────────────────────────────────┘
```

---

## Agent Pipeline

| # | Agent | Role | Key Responsibility |
|---|-------|------|--------------------|
| 1 | **Preference Analyst** | Understands the user | Parses the query + chat history to build a detailed taste profile (genres, themes, mood) |
| 2 | **Movie Matcher** | Searches the catalogue | Uses semantic similarity search against the Redis vector store to surface the best candidate films |
| 3 | **Recommendation Generator** | Crafts the reply | Ranks candidates by relevance and writes a personalised, conversational recommendation with reasons |

All three agents share the **Movie Database Lookup** tool — a CrewAI `@tool` that performs vector similarity search over the 3,000-title MovieLens index stored in Redis.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Agent orchestration** | CrewAI |
| **Workflow graph** | LangGraph (`StateGraph`) |
| **Embeddings** | HuggingFace Inference API — `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector DB** | Redis (via `langchain-redis`) |
| **Chat memory** | `RedisChatMessageHistory` (langchain-redis) |
| **UI** | Streamlit |
| **LLM proxy** | `litellm` (CrewAI dependency) |
| **Dataset** | MovieLens `ml-latest-small` |

---

## Project Structure

```
multiagent/
├── streamlit_app.py        # Streamlit web application (main entry point)
├── multiagent.ipynb        # Jupyter notebook (exploration & prototyping)
├── credentials.py          # API keys and Redis connection details
├── requirements.txt        # Python dependencies
├── ml-latest-small/        # MovieLens dataset
   ├── movies.csv

```

---

## Prerequisites

- **Python 3.10+**
- A free [Groq API key](https://console.groq.com/)
- A free [HuggingFace account & token](https://huggingface.co/settings/tokens)
- A **Redis** instance — [Redis Cloud](https://redis.com/try-free/)

---

## Installation

```bash
# 1. Clone / download the project
cd multiagent

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/Scripts/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a file called `credentials.py` in the `multiagent/` directory:

```python
# credentials.py
GROQ_API_KEY   = "gsk_..."          # Groq API key
HF_TOKEN       = "hf_..."           # HuggingFace token

REDIS_HOST     = "your-redis-host"  # e.g. redis-12345.c1.us-east-1-1.ec2.cloud.redislabs.com
REDIS_PORT     = 12345              # your Redis port (integer)
REDIS_PASSWORD = "your-password"    # Redis password
REDIS_URL      = f"redis://default:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
```

---

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

On first run it will:
1. Connect to Redis and verify the connection.
2. Download the `sentence-transformers/all-MiniLM-L6-v2` embedding model via the HuggingFace Inference API.
3. Read `ml-latest-small/movies.csv`, embed the first 3,000 titles, and index them in Redis.
4. Initialise the three CrewAI agents and the Groq LLM.

Subsequent runs re-use the cached resources (Streamlit `@st.cache_resource`), so startup is much faster.

---

## Using the Notebook

Open `multiagent.ipynb` in VS Code or JupyterLab to explore the system interactively:

```bash
jupyter lab multiagent.ipynb
```

The notebook walks through:

1. Installing / verifying package versions
2. Connecting to Redis
3. Loading and embedding the MovieLens dataset
4. Defining the three agents and their tasks
5. Assembling the `Crew`
6. Wrapping the crew in a **LangGraph** `StateGraph`
7. Running an interactive terminal-based recommendation loop

---

## How It Works

### 1. Vector Store Indexing

The first 3,000 movie titles from `movies.csv` are embedded with `sentence-transformers/all-MiniLM-L6-v2` (via the HuggingFace Inference API — no local GPU required) and stored in a Redis `SearchIndex` under the key `movie_recommendations`.

### 2. User Query

The user types a free-form request in the Streamlit chat input, or clicks one of eight pre-built suggestion chips (e.g. *"🚀 Sci-fi adventures"*).

### 3. Agent Execution

```
kickoff(inputs={"user_input": "...", "chat_history": [...]})
```

CrewAI runs the three agents **sequentially**:

- **Preference Analyst** examines the query and recent chat turns to extract genres, themes, and mood signals.
- **Movie Matcher** invokes the `Movie Database Lookup` tool (Redis similarity search) and surfaces the top candidates.
- **Recommendation Generator** ranks those candidates and returns a conversational reply with a reason for each pick.

### 4. Live Agent Visibility

While the crew is running, the Streamlit UI shows an inline live progress view with:
- A **green block** for each completed agent task (first 400 characters of output).
- A **blue block** for the current agent step / tool call (first 300 characters).

Once the crew finishes, the live cards are replaced by a collapsed **🧠 Agent reasoning** expander showing the full output of every completed task.

### 5. Redis Memory

Every user message and AI reply is appended to `RedisChatMessageHistory`. On the next query the full history is serialised and sent to the crew so the agents can maintain conversational context.

---

## Dataset

[MovieLens Small](https://grouplens.org/datasets/movielens/latest/) by GroupLens Research:

| File | Description |
|------|-------------|
| `movies.csv` | 9,742 movies with title and genres |
| `ratings.csv` | 100,836 ratings from 610 users |
| `tags.csv` | User-applied tags |
| `links.csv` | TMDb / IMDb identifiers |

Only the first 3,000 rows of `movies.csv` are indexed into the vector store to stay within Redis free-tier memory limits. You can adjust this in your code:

```python
sample_df = movies_df.head(3000)   # ← change to index more titles if your Redis plan allows
```

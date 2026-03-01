import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import re
import random
import zipfile

import streamlit as st
import pandas as pd
import redis

from typing import List, Dict, Union
from langchain_redis import RedisVectorStore, RedisCache, RedisChatMessageHistory
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from credentials import GROQ_API_KEY, HF_TOKEN, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

# ─── Environment ────────────────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

REDIS_URL = f"redis://default:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

# ─── Quick-prompt suggestions ────────────────────────────────────────────────────
SUGGESTIONS = [
    "🎭 Feel-good comedies",
    "🔥 Action thrillers",
    "💀 Scary horror films",
    "💘 Romantic movies",
    "🚀 Sci-fi adventures",
    "🕵️ Mystery & crime",
    "🧠 Mind-bending plots",
    "👨‍👩‍👧 Family movies",
]

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 MultiAgent Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #12102b 50%, #0f2040 100%);
    border-radius: 16px;
    padding: 28px 36px 24px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(0,0,0,.45);
    border: 1px solid #1e2a45;
}
.hero h1 { color: #e94560; font-size: 2.2rem; font-weight: 700;
           margin: 0 0 6px; letter-spacing: -.5px; }
.hero .sub { color: #7eb8f7; font-size: .9rem; margin: 0 0 10px; }
.hero .badges span {
    display: inline-block; background: #1a2540; color: #8ab4f8;
    border: 1px solid #2d4a6e; border-radius: 20px;
    padding: 3px 11px; font-size: .73rem; margin: 2px 3px;
}

/* ── Suggestion chips ── */
.chip-label { color: #8899aa; font-size: .82rem; margin: 16px 0 6px; }

/* ── Agent pipeline card ── */
.agent-card {
    background: #111827; border-radius: 12px;
    padding: 12px 15px; margin-bottom: 8px;
    border-left: 4px solid #e94560;
}
.agent-card.a1 { border-left-color: #f7c948; }
.agent-card.a2 { border-left-color: #4caf50; }
.agent-card.a3 { border-left-color: #e94560; }
.agent-card h4 { color: #e2e8f0; margin: 0 0 3px; font-size: .88rem; }
.agent-card p  { color: #64748b; margin: 0; font-size: .76rem; line-height:1.5; }

/* ── Stat badges ── */
.stats { display: flex; gap: 8px; margin-top: 16px; }
.stat  { flex:1; background:#111827; border-radius:10px;
         padding:10px 6px; text-align:center; }
.stat .n { font-size:1.5rem; font-weight:700; color:#e94560; line-height:1; }
.stat .l { font-size:.68rem; color:#4a5568; text-transform:uppercase;
           margin-top:2px; }

/* ── Welcome screen ── */
.welcome {
    background: linear-gradient(135deg, #0d0d1a, #0f2040);
    border-radius: 18px; padding: 44px 36px; text-align: center;
    margin: 20px auto; max-width: 530px;
    box-shadow: 0 4px 32px rgba(0,0,0,.3);
    border: 1px solid #1e3055;
}
.welcome .icon { font-size: 3.2rem; margin-bottom: 14px; }
.welcome h3 { color: #e2e8f0; font-size: 1.35rem; margin: 0 0 10px; }
.welcome p  { color: #64748b; font-size: .88rem; line-height: 1.6; margin: 0; }

/* ── Footer ── */
.footer {
    text-align: center; color: #2d3748; font-size: .72rem;
    margin-top: 28px; padding-top: 14px;
    border-top: 1px solid #1e2a3a;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 MultiAgent Movie Recommender</h1>
  <p class="sub">Three specialised AI agents working together to find your perfect film</p>
  <div class="badges">
    <span>🤖 CrewAI</span><span>⚡ Groq Llama-3.3-70B</span>
    <span>🔴 Redis</span><span>🤗 HuggingFace</span><span>🦜 LangGraph</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Cached resource initialisation ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Redis…")
def get_redis_client():
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    client.ping()
    return client


@st.cache_resource(show_spinner="Loading embeddings model…")
def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN,
    )


@st.cache_resource(show_spinner="Building movie vector store…")
def get_vector_store(_embeddings):
    """Load movies CSV and index the first 3 000 titles into Redis."""
    movies_path = "ml-latest-small/movies.csv"
    if not os.path.exists(movies_path):
        with zipfile.ZipFile("ml-latest-small.zip", "r") as z:
            z.extractall(".")
    movies_df = pd.read_csv(movies_path)
    sample_df = movies_df.head(3000)
    vs = RedisVectorStore.from_texts(
        texts=sample_df["title"].tolist(),
        metadatas=sample_df.to_dict("records"),
        embedding=_embeddings,
        redis_url=REDIS_URL,
        index_name="movie_recommendations",
    )
    return vs


@st.cache_resource(show_spinner="Initialising LLM & agents…")
def get_crew(_vector_store, _embeddings):
    """Build the three CrewAI agents, tasks, and the Crew."""

    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=GROQ_API_KEY,
    )

    @tool("Movie Database Lookup")
    def retriever_tool(query: str) -> str:
        """Search for movies in the database based on titles or descriptions."""
        results = _vector_store.similarity_search(query, k=5)
        return "\n".join(f"{i+1}. {doc.page_content}" for i, doc in enumerate(results))

    preference_analyst = Agent(
        role="Preference Analyst",
        goal="Analyze user preferences based on their input and chat history",
        backstory="You are an expert in understanding user preferences for movies",
        tools=[retriever_tool],
        llm=llm,
        verbose=True,
    )

    movie_matcher = Agent(
        role="Movie Matcher",
        goal="Find movies that match user preferences",
        backstory="You are an expert in matching user preferences to movies in the database",
        tools=[retriever_tool],
        llm=llm,
        verbose=True,
    )

    recommendation_generator = Agent(
        role="Recommendation Generator",
        goal="Generate personalized movie recommendations",
        backstory="You are an expert in creating engaging and personalized movie recommendations",
        tools=[retriever_tool],
        llm=llm,
        verbose=True,
    )

    analyze_preferences_task = Task(
        description=(
            "Analyze the following user preferences and chat history to understand "
            "what kind of movies they enjoy.\n\nUser input: {user_input}\n"
            "Chat history: {chat_history}"
        ),
        agent=preference_analyst,
        expected_output="A detailed analysis of the user's movie preferences",
    )

    match_movies_task = Task(
        description=(
            "Find movies in the database that match the analyzed preferences.\n\n"
            "Original user request: {user_input}"
        ),
        agent=movie_matcher,
        expected_output="A list of movies matching the user's preferences",
    )

    generate_recommendations_task = Task(
        description=(
            "Generate personalized movie recommendations based on the matched movies.\n\n"
            "Original user request: {user_input}"
        ),
        agent=recommendation_generator,
        expected_output="A personalized list of movie recommendations with reasons",
    )

    crew = Crew(
        agents=[preference_analyst, movie_matcher, recommendation_generator],
        tasks=[analyze_preferences_task, match_movies_task, generate_recommendations_task],
        verbose=True,
    )

    return crew


@st.cache_resource(show_spinner="Building LangGraph workflow…")
def get_langgraph_app(_crew):
    """Wrap the CrewAI crew in a single-node LangGraph StateGraph."""

    class UserInput(TypedDict):
        user_input: str
        chat_history: list

    class MovieOutput(TypedDict):
        result: str

    class MovieState(TypedDict):
        user_input: str
        chat_history: list
        result: str

    def run_crew(state):
        result = _crew.kickoff(inputs={
            "user_input": state["user_input"],
            "chat_history": state.get("chat_history", []),
        })
        return {"result": str(result)}

    workflow = StateGraph(MovieState, input_schema=UserInput, output_schema=MovieOutput)
    workflow.add_node("run_crew", run_crew)
    workflow.add_edge(START, "run_crew")
    workflow.add_edge("run_crew", END)
    return workflow.compile()


# ─── Session state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content}, …] for display
if "redis_history" not in st.session_state:
    st.session_state.redis_history = None   # RedisChatMessageHistory
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None  # chip-triggered prompt

# ─── Initialise resources ────────────────────────────────────────────────────────
try:
    redis_client = get_redis_client()
    embeddings   = get_embeddings()
    vector_store = get_vector_store(embeddings)
    crew         = get_crew(vector_store, embeddings)
    app          = get_langgraph_app(crew)

    if st.session_state.redis_history is None:
        st.session_state.redis_history = RedisChatMessageHistory(
            "streamlit_movie_rec", redis_url=REDIS_URL
        )
except Exception as exc:
    st.error(f"Initialisation error: {exc}")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧩 Agent Pipeline")
    st.markdown("""
<div class="agent-card a1">
  <h4>🔍 Preference Analyst</h4>
  <p>Reads your message and past conversation to build a precise taste profile.</p>
</div>
<div class="agent-card a2">
  <h4>🎯 Movie Matcher</h4>
  <p>Queries the 3 000-title vector store with semantic search to find candidates.</p>
</div>
<div class="agent-card a3">
  <h4>✨ Recommendation Generator</h4>
  <p>Crafts a personalised, engaging reply with reasons for each pick.</p>
</div>
""", unsafe_allow_html=True)

    # ── Session stats ──
    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
    n_rec  = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    st.markdown(f"""
<div class="stats">
  <div class="stat"><div class="n">{n_user}</div><div class="l">Queries</div></div>
  <div class="stat"><div class="n">3</div><div class="l">Agents</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.redis_history.clear()
        st.rerun()

    if st.button("🗄️ Clear vector store", use_container_width=True):
        try:
            vector_store.index.delete(drop=True)
            st.success("Vector store cleared.")
        except Exception as e:
            st.error(f"Could not clear vector store: {e}")

    st.markdown("---")
    st.markdown(
        "<div style='color:#4a5568;font-size:.75rem;text-align:center;'>"
        "Powered by <b style='color:#7eb8f7'>Groq</b> · "
        "<b style='color:#7eb8f7'>Redis</b> · "
        "<b style='color:#7eb8f7'>CrewAI</b></div>",
        unsafe_allow_html=True,
    )

# ─── Suggestion chips (shown only when chat is empty) ───────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="welcome">
  <div class="icon">🎬</div>
  <h3>What are you in the mood for?</h3>
  <p>Describe a vibe, genre, actor, or feeling — three AI agents will
  collaborate to find your perfect film from our catalogue.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="chip-label">✨ Quick suggestions — click to try</div>',
                unsafe_allow_html=True)
    chip_cols = st.columns(4)
    for idx, sug in enumerate(SUGGESTIONS):
        if chip_cols[idx % 4].button(sug, key=f"chip_{idx}", use_container_width=True):
            st.session_state.pending_prompt = sug
            st.rerun()

# ─── Chat display ────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Resolve prompt (typed or chip) ─────────────────────────────────────────────
_chip_prompt = st.session_state.pop("pending_prompt", None)

# ─── Input & inference ───────────────────────────────────────────────────────────
if prompt := (_chip_prompt or st.chat_input("What kind of movie are you in the mood for?")):
    # Show the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Persist to Redis
    st.session_state.redis_history.add_user_message(prompt)

    # Build serialisable history for the crew
    def _msg_to_dict(m):
        return {
            "type": "human" if isinstance(m, HumanMessage) else "ai",
            "content": m.content,
        }

    serializable_history = [_msg_to_dict(m) for m in st.session_state.redis_history.messages]

    # Run crew with live agent visibility
    with st.chat_message("assistant"):
        # Shared state for callbacks
        _task_log: list = []   # [(agent_role, output_text), ...]
        _step_log: list = []   # [step_text, ...]
        _done: list    = [False]  # flag to stop callbacks re-rendering after finish

        _progress_ph = st.empty()   # live progress → collapsed expander when done
        _result_ph   = st.empty()   # final recommendation (below)

        AGENT_ICONS = {
            "Preference Analyst": "🔍",
            "Movie Matcher": "🎯",
            "Recommendation Generator": "✨",
        }

        def _render_progress():
            """Render live pipeline progress inline while agents are running."""
            if _done[0]:
                return
            with _progress_ph.container():
                for _role, _out in _task_log:
                    icon = AGENT_ICONS.get(_role, "✅")
                    st.markdown(
                        f"""<div style='border-left:4px solid #4CAF50;
                                      padding:8px 14px;margin-bottom:8px;
                                      border-radius:6px;background:#0d1f0d;
                                      color:#c8f0c8'>
                        <b>{icon} {_role} — done</b><br>
                        <small style='white-space:pre-wrap;color:#90c890'>{_out[:400]}
                        {'…' if len(_out)>400 else ''}</small>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                if _step_log:
                    latest = _step_log[-1]
                    st.markdown(
                        f"""<div style='border-left:4px solid #2196F3;
                                      padding:8px 14px;border-radius:6px;
                                      background:#0d1828;color:#a8d4f5'>
                        <b>🔄 Working…</b><br>
                        <small style='white-space:pre-wrap;color:#7ab8e8'>{latest[:300]}
                        {'…' if len(latest)>300 else ''}</small>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                elif not _task_log:
                    st.markdown(
                        "<div style='color:#7eb8f7;font-size:.9rem'>⏳ Agents starting…</div>",
                        unsafe_allow_html=True,
                    )

        def _render_expander():
            """Replace live cards with a collapsed expander (thinking-model style)."""
            _done[0] = True          # block any late-firing callbacks
            _progress_ph.empty()     # wipe placeholder cleanly first
            with _progress_ph.container():
                with st.expander("🧠 Agent reasoning", expanded=False):
                    for _role, _out in _task_log:
                        icon = AGENT_ICONS.get(_role, "✅")
                        st.markdown(
                            f"""<div style='border-left:4px solid #4CAF50;
                                          padding:8px 14px;margin-bottom:8px;
                                          border-radius:6px;background:#0d1f0d;
                                          color:#c8f0c8'>
                            <b>{icon} {_role}</b><br>
                            <small style='white-space:pre-wrap;color:#90c890'>{_out}</small>
                            </div>""",
                            unsafe_allow_html=True,
                        )

        def _on_task_complete(task_output):
            role = getattr(task_output, "agent", "Agent")
            raw  = getattr(task_output, "raw", str(task_output))
            _task_log.append((role, raw))
            _step_log.clear()
            _render_progress()

        def _on_step(step_output):
            text = str(step_output).strip()
            if text:
                _step_log.append(text)
            _render_progress()

        # Attach callbacks
        try:
            object.__setattr__(crew, "task_callback", _on_task_complete)
            object.__setattr__(crew, "step_callback", _on_step)
        except Exception:
            crew.task_callback = _on_task_complete
            crew.step_callback = _on_step

        # Show initial state
        _render_progress()

        try:
            result = app.invoke({
                "user_input": prompt,
                "chat_history": serializable_history,
            })
            recommendation = result["result"]
            _render_expander()
            _result_ph.markdown(recommendation)
        except Exception as exc:
            recommendation = f"Sorry, something went wrong: {exc}"
            _render_expander()
            _result_ph.error(recommendation)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": recommendation})
    st.session_state.redis_history.add_ai_message(recommendation)

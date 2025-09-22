"""
Minimal News Digest — Multi-Agent System (LangGraph + GPT-4o-mini)
Python 3.12 compatible, single-file project.

Features
- 3 Agents (Fetcher, Summarizer, Editor), each with multiple tools (with access control)
- Orchestrator / Router pipeline in LangGraph
- Shared + per-agent state
- Mock LLM fallback (runs even without OPENAI_API_KEY)
- Deterministic local "news" corpus so the demo is reliable/offline

Run:
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...   # optional; if omitted, uses MockLLM
  python news_digest_agents.py "AI in healthcare" --style=markdown --tone=concise

Author: you
"""

from __future__ import annotations
import os
import sys
import json
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, Literal
from dotenv import load_dotenv
load_dotenv()

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

# OpenAI (SDK v1.x)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# LLM WRAPPER (GPT-4o-mini or Mock)
# -----------------------------

class LLM:
    """A tiny LLM wrapper that prefers GPT-4o-mini, falls back to a mock if no API key."""
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_output_tokens: int = 600) -> str:
        if self.client is None:
            # Mock mode (deterministic, explanation-rich but fast)
            return self._mock_response(system_prompt, user_prompt)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_output_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            # Fail gracefully to mock if API hiccups
            return self._mock_response(system_prompt, user_prompt, note=f"(LLM error: {e})")

    def _mock_response(self, system_prompt: str, user_prompt: str, note: str = "") -> str:
        # Very small heuristic mock to keep the demo working offline.
        up = user_prompt.lower()
        if "summarize" in up or "tl;dr" in up or "summary" in up:
            return f"{note} Mock summary: Key developments, stakeholder quotes, and implications distilled."
        if "extract highlights" in up or "bullet points" in up:
            return f"{note} • Highlight 1\n• Highlight 2\n• Highlight 3"
        if "bias" in up or "stance" in up:
            return f"{note} Bias: neutral; stance: balanced; reasoning: mixed sources."
        if "format digest" in up:
            return f"{note} # Mock Digest\n- Item 1\n- Item 2\n- Item 3\n\n(Formatted by Mock)"
        # Generic rewrite
        return f"{note} {user_prompt[:200]} ..."


# -----------------------------
# STATE DEFINITIONS
# -----------------------------

class Article(TypedDict):
    id: str
    title: str
    source: str
    url: str
    published: str
    content: str
    topic: str

class Summary(TypedDict):
    article_id: str
    title: str
    key_points: List[str]
    bias_note: str
    summary_text: str

class SharedState(TypedDict, total=False):
    # Inputs
    user_query: str
    tone: Literal["concise", "neutral", "friendly", "formal"]
    style: Literal["markdown", "html", "text"]

    # Fetcher outputs
    raw_articles: List[Article]
    validated_sources: List[Article]
    trend_clusters: Dict[str, List[str]]  # cluster_key -> list[article_id]

    # Summarizer outputs
    summaries: List[Summary]
    extracted_highlights: List[str]

    # Editor outputs (final)
    digest: str

class FetcherState(TypedDict, total=False):
    fetch_attempts: int
    articles_found: int
    validator_notes: List[str]

class SummarizerState(TypedDict, total=False):
    items_summarized: int
    bias_checks: Dict[str, str]  # article_id -> bias note
    highlight_count: int

class EditorState(TypedDict, total=False):
    formatting_style: str
    personalization: Dict[str, Any]
    item_count: int


# -----------------------------
# LOCAL "TOOLS" (No external I/O)
# -----------------------------

def tool_fetch_news(topic: str) -> List[Article]:
    """
    Search web for information
    """
    search = TavilySearchResults(max_results=3)
    results = search.invoke(topic)
    results = [{**x, 'id': ind + 1} for ind, x in enumerate(results)]
    return results

def tool_trend_cluster(articles: List[Article]) -> Dict[str, List[int]]:
    """
    Cluster articles using embeddings. Returns {cluster_label: [article_ids]}.
    Uses KMeans and generates human-readable cluster labels.
    """
    from sklearn.cluster import KMeans
    import numpy as np
    import re
    from collections import Counter
    from openai import OpenAI

    if not articles:
        return {}

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_model = "text-embedding-3-small"

    texts = [f"{a.get('title','')} {a.get('content','')}" for a in articles]

    # Get embeddings
    embeddings = []
    for txt in texts:
        resp = client.embeddings.create(model=embedding_model, input=txt[:3000])
        embeddings.append(resp.data[0].embedding)
    X = np.array(embeddings)

    # Decide number of clusters
    n_clusters = min(len(articles), 2 if len(articles) < 4 else 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters: Dict[str, List[int]] = {}
    for c in range(n_clusters):
        cluster_articles = [articles[i] for i, lbl in enumerate(labels) if lbl == c]
        cluster_ids = [a["id"] for a in cluster_articles]

        # Use article titles as context
        titles = [a["title"] for a in cluster_articles]
        text_blob = " ".join(titles)

        # Extract top non-stopword terms
        stopwords = {"the","and","your","for","with","that","this","from","into","have","are","was","were",
                     "been","will","would","could","should","about","after","before","over","under","between",
                     "among","onto","in","on","of","to","by","at","as","an","a","or","you"}
        tokens = re.findall(r"\b[a-z]{3,}\b", text_blob.lower())
        tokens = [tok for tok in tokens if tok not in stopwords]
        common_terms = [w for w, _ in Counter(tokens).most_common(3)]

        # Generate human-readable label using LLM (optional)
        label_prompt = f"Summarize these keywords into a short topic (3-5 words): {', '.join(common_terms)}. Titles: {titles}"
        try:
            label = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise topic labeler."},
                    {"role": "user", "content": label_prompt},
                ],
                max_tokens=30,
            ).choices[0].message.content.strip()
        except Exception:
            label = ", ".join(common_terms) if common_terms else f"Cluster {c+1}"

        clusters[label] = cluster_ids

    return clusters

def tool_summarize_article(llm: LLM, article: Article, style: str = "markdown") -> str:
    sys_prompt = "You are a precise news summarizer. Output 3–5 sentences. Be factual."
    user_prompt = (
        f"Summarize the following article in {style}.\n\n"
        f"TITLE: {article['title']}\n"
        f"URL: {article['url']}\n\nCONTENT:\n{article['content']}\n"
    )
    return llm.generate(sys_prompt, user_prompt)

def tool_extract_highlights(llm: LLM, text: str) -> List[str]:
    sys_prompt = "Extract 3–5 crisp bullet highlights from the text. Avoid redundancy."
    resp = llm.generate(sys_prompt, f"Extract highlights:\n\n{text}\n\nReturn bullet points.")
    # Parse bullets robustly
    lines = [ln.strip("-• ").strip() for ln in resp.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln and not ln.lower().startswith("mock digest")]
    return bullets[:5] if bullets else ["Key point 1", "Key point 2"]

def tool_detect_bias(llm: LLM, text: str) -> str:
    sys_prompt = "Assess bias and stance of the text in one sentence."
    return llm.generate(sys_prompt, f"Bias/stance check for:\n{text}\n")

def tool_format_digest(
    summaries: List[Summary],
    highlights: List[str],
    clusters: Dict[str, List[int]],
    prefs: Dict[str, Any] | None = None,
    style: str = "markdown",
    tone: str = "concise",
) -> str:
    prefs = prefs or {}
    title = prefs.get("title", "Daily Digest")
    intro = "Here’s your curated digest." if tone != "formal" else "Below is the curated digest."

    # Helper to lookup article titles by ID
    id_to_title = {s["article_id"]: s["title"] for s in summaries}
    
    if style == "html":
        parts = [f"<h1>{title}</h1><p>{intro}</p>"]
        if clusters:
            parts.append("<h2>Trends</h2><ul>")
            for label, ids in clusters.items():
                titles = [id_to_title.get(i, str(i)) for i in ids]
                parts.append(f"<li><b>{label.title()}</b>: {', '.join(titles)}</li>")
            parts.append("</ul>")
        if highlights:
            parts.append("<h2>Highlights</h2><ul>")
            for h in highlights:
                parts.append(f"<li>{h}</li>")
            parts.append("</ul>")
        parts.append("<h2>Articles</h2>")
        for s in summaries:
            parts.append(f"<h3>{s['title']}</h3><p>{s['summary_text']}</p><p><i>{s['bias_note']}</i></p>")
        return "\n".join(parts)

    elif style == "text":
        parts = [f"{title}\n{'='*len(title)}\n{intro}\n"]
        if clusters:
            parts.append("Trends:")
            for label, ids in clusters.items():
                titles = [id_to_title.get(i, str(i)) for i in ids]
                parts.append(f"- {label.title()}: {', '.join(titles)}")
            parts.append("")
        if highlights:
            parts.append("Highlights:")
            for h in highlights:
                parts.append(f"- {h}")
            parts.append("")
        parts.append("Articles:")
        for s in summaries:
            parts.append(f"- {s['title']}\n  {s['summary_text']}\n  ({s['bias_note']})")
        return "\n".join(parts)

    else:  # markdown (default)
        parts = [f"# {title}\n\n{intro}\n"]
        if clusters:
            parts.append("## Trends\n")
            for label, ids in clusters.items():
                titles = [id_to_title.get(i, str(i)) for i in ids]
                parts.append(f"- **{label.title()}**: {', '.join(titles)}")
            parts.append("")
        if highlights:
            parts.append("## Highlights\n")
            for h in highlights:
                parts.append(f"- {h}")
            parts.append("")
        parts.append("## Articles\n")
        for s in summaries:
            parts.append(f"### {s['title']}\n{s['summary_text']}\n\n*{s['bias_note']}*\n")
        return "\n".join(parts)


# -----------------------------
# AGENTS (multi-tool, restricted access)
# -----------------------------

@dataclass
class FetcherAgent:
    llm: LLM
    tools: Dict[str, Any] = field(default_factory=lambda: {
        "fetch_news": tool_fetch_news,
        "trend_cluster": tool_trend_cluster,
    })

    def run(self, state: SharedState, astate: FetcherState) -> tuple[SharedState, FetcherState]:
        topic = state["user_query"]
        fetched = self.tools["fetch_news"](topic)
        clusters = self.tools["trend_cluster"](fetched)

        astate["fetch_attempts"] = astate.get("fetch_attempts", 0) + 1
        astate["articles_found"] = len(fetched)

        state["raw_articles"] = fetched
        state["trend_clusters"] = clusters
        return state, astate

@dataclass
class SummarizerAgent:
    llm: LLM
    tools: Dict[str, Any] = field(default_factory=lambda: {
        "summarize": tool_summarize_article,
        "bias_check": tool_detect_bias,
        "extract_highlights": tool_extract_highlights,
    })

    def run(self, state: SharedState, astate: SummarizerState) -> tuple[SharedState, SummarizerState]:
        articles = state.get("raw_articles", [])
        style = state.get("style", "markdown")

        summaries: List[Summary] = []
        bias_map: Dict[str, str] = {}

        for art in articles:
            stext = self.tools["summarize"](self.llm, art, style=style)
            bias = self.tools["bias_check"](self.llm, stext)
            bullets = self.tools["extract_highlights"](self.llm, stext)
            bias_map[art["id"]] = bias
            summaries.append(Summary(
                article_id=art["id"],
                title=art["title"],
                key_points=bullets[:3],
                bias_note=bias,
                summary_text=stext
            ))

        # All-articles highlights (simple union of first N key points)
        extracted: List[str] = []
        for s in summaries:
            for b in s["key_points"]:
                if b not in extracted:
                    extracted.append(b)
        extracted = extracted[:10]

        astate["items_summarized"] = len(summaries)
        astate["bias_checks"] = bias_map
        astate["highlight_count"] = len(extracted)

        state["summaries"] = summaries
        state["extracted_highlights"] = extracted
        return state, astate


@dataclass
class EditorAgent:
    llm: LLM  # Not strictly required here, but kept for symmetry/extensibility.
    tools: Dict[str, Any] = field(default_factory=lambda: {
        "format_digest": tool_format_digest,
    })

    def run(self, state: SharedState, astate: EditorState) -> tuple[SharedState, EditorState]:
        style = state.get("style", "markdown")
        tone = state.get("tone", "concise")
        prefs = {"title": f"Daily Digest: {state.get('user_query','')}"}

        digest = self.tools["format_digest"](
            summaries=state.get("summaries", []),
            highlights=state.get("extracted_highlights", []),
            clusters=state.get("trend_clusters", {}),
            prefs=prefs,
            style=style,
            tone=tone,
        )

        astate["formatting_style"] = style
        astate["personalization"] = {"tone": tone}
        astate["item_count"] = len(state.get("summaries", []))
        state["digest"] = digest
        return state, astate


# -----------------------------
# ORCHESTRATOR (LangGraph)
# -----------------------------

def build_graph(llm: LLM):
    """
    Structured workflow pipeline (NOT ReAct). Flow:
      Orchestrator(start) -> Fetcher -> Summarizer -> Editor -> END
    """
    graph = StateGraph(SharedState)

    # Agent instances
    fetcher = FetcherAgent(llm)
    summarizer = SummarizerAgent(llm)
    editor = EditorAgent(llm)

    # Local per-agent states (we keep them inside node closures)
    f_state: FetcherState = {}
    s_state: SummarizerState = {}
    e_state: EditorState = {}

    def node_fetcher(state: SharedState) -> SharedState:
        updated, _fa = fetcher.run(state, f_state)
        # side-effect: f_state mutated in place (demo simplicity)
        return updated

    def node_summarizer(state: SharedState) -> SharedState:
        updated, _sa = summarizer.run(state, s_state)
        return updated

    def node_editor(state: SharedState) -> SharedState:
        updated, _ea = editor.run(state, e_state)
        return updated

    # Register nodes
    graph.add_node("fetcher", node_fetcher)
    graph.add_node("summarizer", node_summarizer)
    graph.add_node("editor", node_editor)

    graph.set_entry_point("fetcher")
    graph.add_edge("fetcher", "summarizer")
    graph.add_edge("summarizer", "editor")
    graph.add_edge("editor", END)

    memory = MemorySaver()  # in-memory checkpointing for demo
    # app = graph.compile(checkpointer=memory)
    app = graph.compile()
    return app


# -----------------------------
# CLI / DEMO
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Minimal News Digest Multi-Agent Demo")
    p.add_argument("topic", nargs="?", default="AI in healthcare", help="Topic to build the digest for")
    p.add_argument("--tone", default="concise", choices=["concise", "neutral", "friendly", "formal"])
    p.add_argument("--style", default="markdown", choices=["markdown", "html", "text"])
    return p.parse_args()

def main():
    args = parse_args()
    llm = LLM(model="gpt-4o-mini")

    state: SharedState = {
        "user_query": args.topic,
        "tone": args.tone,     # shared state
        "style": args.style,   # shared state
    }

    app = build_graph(llm)
    # final = app.invoke(state, configurable={"thread_id": "demo-run"})
    final = app.invoke(state)

    # Demo log (concise)
    print("\n" + "="*60)
    print("FINAL DIGEST")
    print("="*60 + "\n")
    print(final.get("digest", "(no digest produced)"))

    # Optional: show some debug info
    print("\n" + "-"*60)
    print("DEBUG (State)")
    print("-"*60)
    debug_copy = {k: v for k, v in final.items() if k in ("user_query", "trend_clusters", "extracted_highlights")}
    print(json.dumps(debug_copy, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

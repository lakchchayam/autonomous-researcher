"""
Planner-Worker agent graph for autonomous research.

Implements a two-phase research pipeline:
1. Planner: Decomposes the research goal into search queries
2. Worker: Executes searches, scrapes pages, stores in memory
3. Synthesizer: Reads accumulated memory and drafts the final report

Uses LangGraph for stateful orchestration with self-correction loops.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Sequence

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .memory import VectorMemory
from .scraper import WebScraper

logger = structlog.get_logger(__name__)


# ─── State Schema ──────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Shared state for the research pipeline."""
    messages: Annotated[Sequence[Any], add_messages]
    research_goal: str
    search_queries: list[str]
    scraped_urls: list[str]
    scraped_content: list[dict[str, str]]
    memory_chunks_count: int
    draft_report: str | None
    critique: str | None
    iteration: int
    max_iterations: int
    is_complete: bool


# ─── System Prompts ────────────────────────────────────────────────

PLANNER_PROMPT = """You are a research planning agent. Given a research goal, 
generate a list of 3-5 specific search queries that would help gather 
comprehensive information on the topic.

Respond with ONLY a JSON array of search query strings. Example:
["query 1", "query 2", "query 3"]

Be specific. Include different angles: technical specs, comparisons, pricing, 
recent news, and expert opinions where relevant."""

SYNTHESIZER_PROMPT = """You are a research synthesis agent. Based on the scraped 
content provided, write a comprehensive Markdown report that addresses the 
original research goal.

Requirements:
- Use clear headers (##) to organize sections
- Include specific data points and numbers where available
- Add source citations as [Source: URL] at the end of relevant paragraphs
- If information is conflicting, note the discrepancy
- End with a "Key Findings" summary section
- Use tables for comparison data when appropriate"""

CRITIQUE_PROMPT = """You are a research quality auditor. Review the draft report 
below and identify:
1. Missing information that should be researched further
2. Claims that lack supporting evidence
3. Areas that could be more specific or detailed

If the report is comprehensive enough, respond with exactly: "APPROVED"
Otherwise, suggest 1-3 additional search queries to fill the gaps.
Respond as JSON: {"status": "needs_work", "additional_queries": ["query1", "query2"]}
or: {"status": "approved"}"""


# ─── Agent Functions ───────────────────────────────────────────────

async def planner_node(state: ResearchState) -> dict[str, Any]:
    """Decompose the research goal into specific search queries."""
    logger.info("Planner activated", goal=state["research_goal"][:100])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Research goal: {state['research_goal']}"),
    ])
    
    try:
        queries = json.loads(response.content)
        if not isinstance(queries, list):
            queries = [state["research_goal"]]
    except json.JSONDecodeError:
        queries = [state["research_goal"]]
    
    logger.info("Search plan created", num_queries=len(queries))
    
    return {
        "search_queries": queries,
        "iteration": state["iteration"] + 1,
    }


async def worker_node(state: ResearchState) -> dict[str, Any]:
    """Execute search queries, scrape results, store in memory."""
    scraper = WebScraper()
    new_content: list[dict[str, str]] = list(state.get("scraped_content", []))
    scraped_urls: list[str] = list(state.get("scraped_urls", []))
    
    for query in state["search_queries"]:
        # Build search URL (DuckDuckGo HTML for simplicity)
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        logger.info("Executing search", query=query)
        
        page = await scraper.scrape_url(search_url)
        
        if page.success and page.url not in scraped_urls:
            new_content.append({
                "url": page.url,
                "title": page.title,
                "content": page.text_content[:5000],
                "query": query,
            })
            scraped_urls.append(page.url)
    
    logger.info(
        "Worker completed",
        total_pages=len(new_content),
        total_urls=len(scraped_urls),
    )
    
    return {
        "scraped_content": new_content,
        "scraped_urls": scraped_urls,
        "memory_chunks_count": len(new_content),
    }


async def synthesizer_node(state: ResearchState) -> dict[str, Any]:
    """Synthesize all scraped content into a comprehensive report."""
    logger.info("Synthesizer activated", content_pieces=len(state.get("scraped_content", [])))
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # Build context from scraped content
    context_parts = []
    for item in state.get("scraped_content", []):
        context_parts.append(
            f"--- Source: {item['url']} ---\n"
            f"Title: {item.get('title', 'N/A')}\n"
            f"Content:\n{item['content']}\n"
        )
    
    context = "\n\n".join(context_parts)
    
    response = await llm.ainvoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(
            content=(
                f"Research goal: {state['research_goal']}\n\n"
                f"Scraped sources ({len(context_parts)} documents):\n\n{context}"
            )
        ),
    ])
    
    return {"draft_report": response.content}


async def critique_node(state: ResearchState) -> dict[str, Any]:
    """Review the draft report and decide if more research is needed."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    response = await llm.ainvoke([
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(
            content=(
                f"Research goal: {state['research_goal']}\n\n"
                f"Draft report:\n{state['draft_report']}"
            )
        ),
    ])
    
    critique_text = response.content
    
    try:
        critique_data = json.loads(critique_text)
        if critique_data.get("status") == "approved":
            logger.info("Report approved by critique agent")
            return {"critique": "APPROVED", "is_complete": True}
        else:
            additional = critique_data.get("additional_queries", [])
            logger.info("Report needs more research", additional_queries=len(additional))
            return {
                "critique": critique_text,
                "search_queries": additional,
                "is_complete": False,
            }
    except json.JSONDecodeError:
        if "APPROVED" in critique_text.upper():
            return {"critique": "APPROVED", "is_complete": True}
        return {"critique": critique_text, "is_complete": True}


# ─── Routing Logic ─────────────────────────────────────────────────

def _should_continue_research(state: ResearchState) -> str:
    """Decide whether to continue researching or finalize."""
    if state.get("is_complete"):
        return END
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        logger.warning("Max iterations reached, finalizing")
        return END
    return "worker"


# ─── Graph Builder ─────────────────────────────────────────────────

def build_research_graph() -> Any:
    """
    Build the autonomous research execution graph.
    
    Topology:
        Planner -> Worker -> Synthesizer -> Critique
                     ^                        |
                     |________________________|
                     (if needs more research)
    """
    graph = StateGraph(ResearchState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critique", critique_node)
    
    graph.set_entry_point("planner")
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "synthesizer")
    graph.add_edge("synthesizer", "critique")
    
    graph.add_conditional_edges(
        "critique",
        _should_continue_research,
        {"worker": "worker", END: END},
    )
    
    return graph.compile()


async def research(goal: str, max_iterations: int = 3) -> str:
    """
    Run the full autonomous research pipeline.
    
    Args:
        goal: The research objective in natural language.
        max_iterations: Maximum research-critique cycles.
        
    Returns:
        The final synthesized Markdown report.
    """
    graph = build_research_graph()
    
    initial_state: ResearchState = {
        "messages": [],
        "research_goal": goal,
        "search_queries": [],
        "scraped_urls": [],
        "scraped_content": [],
        "memory_chunks_count": 0,
        "draft_report": None,
        "critique": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
    }
    
    logger.info("Starting autonomous research", goal=goal[:100])
    final_state = await graph.ainvoke(initial_state)
    
    return final_state.get("draft_report", "Research failed to produce a report.")

# Autonomous Researcher

![Agentic](https://img.shields.io/badge/Paradigm-Agentic%20AI-red.svg)
![Playwright](https://img.shields.io/badge/Tool-Playwright-green.svg)

An autonomous, goal-oriented AI research agent that navigates the web, reads documentation, synthesizes information, and generates comprehensive reports. 

## 🧠 Core Concept

Unlike a standard RAG system that answers questions based on a pre-indexed corpus, the **Autonomous Researcher** uses a Planner-Worker paradigm to actively gather information from the open web or internal portals. It formulates search queries, scrapes web pages dynamically, handles pagination, and synthesizes data into markdown reports.

## 🚀 Features

- **Dynamic Web Scraping:** Uses Playwright to render JavaScript-heavy sites and extract clean text content using a combination of Readability.js and BeautifulSoup.
- **Self-Correction & Critique:** If the agent fails to find the target information, it critiques its own search strategy, refines its search queries, and tries again.
- **Vector Memory (Short-term):** Uses an in-memory FAISS index to store chunks of scraped text during a session, allowing the agent to perform semantic searches over the documents it just downloaded.
- **Structured Output:** Enforces strict JSON schemas for the Planner to ensure deterministic task decomposition.

## 🛠️ Architecture

1. **Planner Agent:** Takes the user's high-level goal and creates a step-by-step research plan.
2. **Web Browser Tool:** Executed by the Worker agent to search Google/DuckDuckGo and fetch page contents.
3. **Synthesizer Agent:** Reads the accumulated memory and drafts the final Markdown report with citations.

## 💡 Example Prompt
> "Research the differences in latency, cost, and context window size between Claude 3.5 Sonnet, GPT-4o, and Gemini 1.5 Pro. Format the output as a comparison table."

*The agent will autonomously search for each model's specs, read the pricing pages, and construct the table.*

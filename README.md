# Minimal News Digest — Multi-Agent System

## 📌 Overview

This project implements a **Minimal News Digest** system using a **multi-agent AI architecture** built with **LangGraph** and **GPT-4o-mini**.  

The goal is to simulate how multiple specialized AI agents can collaborate to solve a real-world task:  
**automatically fetching news, clustering articles into trends, summarizing them, and compiling a digest.**

### 🎯 What the system does
- **Fetcher Agent**  
  - Retrieves news articles (via Tavily search API).  
  - Groups them into **trends** using **embedding-based clustering**.  

- **Summarizer Agent**  
  - Summarizes each article into a few sentences.  
  - Extracts highlights and assesses possible bias.  

- **Editor Agent**  
  - Formats the final output (Markdown, HTML, or plain text).  
  - Organizes the digest by trends, highlights, and full summaries.  

### 🧠 Key Features
- **Multi-agent pipeline** (Fetcher → Summarizer → Editor).  
- **Multiple tools per agent** with explicit access control.  
- **Embedding-based clustering** (OpenAI `text-embedding-3-small` + KMeans) for trend detection.  
- **Shared + agent state management** so information flows between agents.  
- **Mock LLM fallback** → works even if no `OPENAI_API_KEY` is set (generates placeholder summaries).  
- **One-day build scope**: deterministic and reliable for demonstration.  

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- **Python 3.12**  
- Project uses the [uv package manager](https://docs.astral.sh/uv/) for dependency management.  

### 2. Install dependencies
If you’re using **uv** (recommended):  

```bash
uv sync
```

If you prefer **pip**:
```bash
pip install -r requirements.txt
```

Dependencies include:
```shell
openai
langgraph
python-dotenv
pydantic
langchain_community
scikit-learn
numpy
```

### 3. Configure API Key:
This project requires the following keys to be placed in a `.env` file:
```bash
OPENAI_API_KEY="enter your key"
TAVILY_API_KEY="enter your key"
DEBUG=true
```

You can get these keys from: 
1. `OPENAI API KEY` from `https://platform.openai.com/settings/organization/api-keys`
2. `TAVILY_API_KEY` from `https://app.tavily.com/home`

### 4. Run the system
```bash
python main.py "latest news on Agentic AI"
```

#### Command-line options
- topic (default: "AI in healthcare") → What topic to fetch news on.
- --style (markdown | html | text) → Output format of digest.
- --tone (concise | neutral | friendly | formal) → Writing style of summaries.

Example:
```bash
python news_digest_agents.py "latest news on Agentic AI" --style=html --tone=friendly
```

### Sample output:
```markdown
============================================================
FINAL DIGEST
============================================================

# Daily Digest: latest news on Agentic AI

Here’s your curated digest.

## Trends

- **Latest Trends In Agentic Ai**: Latest Agentic AI News Today | Trends, Predictions, & Analysis, Agentic AI: 4 reasons why it's the next big thing in AI research - IBM
- **Challenges Of Agentic Ai**: 4 Reasons Agentic AI Is Failing - The New Stack

## Highlights

- CrowdStrike launched the Agentic Security Platform, utilizing proactive AI agents to enhance cybersecurity.
- The deployment of AI agents in customer service has increased by 22 times since January.
- Solo.io introduced Kagent, designed to optimize cloud-native infrastructure for AI agents.
- Agentic AI merges the adaptability of large language models with the precision of traditional programming.
- It is poised to revolutionize AI development, particularly in automation and autonomous systems.
- The technology is expected to significantly impact customer service and contact centers, potentially redefining industry standards.
- Agentic AI technology, including the Model Context Protocol (MCP), is projected to be adopted by 34.1% of enterprises by summer 2025.
- Significant challenges are hindering the successful implementation of agentic AI.
- The article identifies four primary reasons for the failure of agentic AI adoption.

## Articles

### Latest Agentic AI News Today | Trends, Predictions, & Analysis
# Latest Agentic AI News Today | Trends, Predictions, & Analysis

As of September 22nd, the latest updates in Agentic AI highlight significant advancements and trends. CrowdStrike has introduced the Agentic Security Platform, enhancing cybersecurity with proactive AI agents. Additionally, the use of AI agents in customer service has surged by 22 times since January. Other developments include Solo.io's Kagent, which optimizes cloud-native infrastructure for AI agents, and discussions on the barriers hindering the broader adoption of Agentic AI across industries.

*The text presents a neutral and informative stance, focusing on recent advancements and trends in Agentic AI without expressing a clear bias or opinion.*

### Agentic AI: 4 reasons why it's the next big thing in AI research - IBM
### Summary of "Agentic AI: 4 reasons why it's the next big thing in AI research - IBM"

Agentic AI is emerging as a significant trend in artificial intelligence, combining the adaptability of large language models with the accuracy of traditional programming. This new approach is seen as a potential game-changer in AI development, particularly in enhancing automation and creating more sophisticated autonomous systems. IBM highlights the transformative impact of agentic AI on customer service and contact centers, suggesting it could redefine industry standards. The article also encourages IT leaders to explore the opportunities and risks associated with this innovation.

*The text exhibits a positive bias towards agentic AI, portraying it as a revolutionary advancement in AI research and emphasizing its potential benefits for industries, particularly in automation and customer service.*

### 4 Reasons Agentic AI Is Failing - The New Stack
### 4 Reasons Agentic AI Is Failing

Agentic AI technology, including frameworks like the Model Context Protocol (MCP), has recently become available, with 34.1% of enterprises adopting it by summer 2025, according to IDC research. However, the technology is facing significant challenges that hinder its successful implementation. The article outlines four primary reasons for the failure of agentic AI adoption, emphasizing the need for better resources and community support for developers.

*The text presents a critical stance on the current state of agentic AI, highlighting its challenges and failures while advocating for improved resources and support for developers.*


------------------------------------------------------------
DEBUG (State)
------------------------------------------------------------
{
  "user_query": "latest news on Agentic AI",
  "trend_clusters": {
    "Latest Trends in Agentic AI": [
      1,
      2
    ],
    "Challenges of Agentic AI": [
      3
    ]
  },
  "extracted_highlights": [
    "CrowdStrike launched the Agentic Security Platform, utilizing proactive AI agents to enhance cybersecurity.",
    "The deployment of AI agents in customer service has increased by 22 times since January.",
    "Solo.io introduced Kagent, designed to optimize cloud-native infrastructure for AI agents.",
    "Agentic AI merges the adaptability of large language models with the precision of traditional programming.",
    "It is poised to revolutionize AI development, particularly in automation and autonomous systems.",
    "The technology is expected to significantly impact customer service and contact centers, potentially redefining industry standards.",
    "Agentic AI technology, including the Model Context Protocol (MCP), is projected to be adopted by 34.1% of enterprises by summer 2025.",
    "Significant challenges are hindering the successful implementation of agentic AI.",
    "The article identifies four primary reasons for the failure of agentic AI adoption."
  ]
}
```
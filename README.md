# ML-GenAI

This repository contains a series of exploratory notebooks focused on integrating **Machine Learning (ML)** techniques with **Generative AI (GenAI)** workflows. It covers foundational and advanced topics in regression, clustering, and memory management within LLM-based systems.

## 📁 Project Structure

- **Clustering and Regression**  
  Exploratory notebooks on unsupervised learning (e.g., K-Means) and supervised regression models. Includes implementation and evaluation.

- **Regression and Assistant**  
  Combines regression models with assistant-style LLM interactions to support analytical reasoning and prediction tasks.

- **Memory Mechanism**  
  A deep dive into memory modules for conversational AI using LangChain:
  - `ConversationBufferMemory`
  - `ConversationBufferWindowMemory`
  - `ConversationSummaryMemory`
  - `ConversationSummaryBufferMemory`
  - `ConversationTokenBufferMemory`
  - LangGraph implementation and visualization
  - Token usage vs. memory tradeoffs

## 📌 Highlights

- 🔍 Hands-on tests of memory strategies using LangChain and LangGraph.
- 📊 Token usage visualization across memory types.
- 🧠 Use of summarization and window trimming for efficient context retention.
- 🛠️ Custom workflows using `StateGraph` and memory checkpointing.

## 🔧 Requirements

- Python 3.9+
- `langchain`, `openai`, `matplotlib`, `nbformat`, `uuid`, and other standard libraries.
- Jupyter Notebook or VS Code with Jupyter extension

## 📚 References

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Memory Guide](https://langchain-ai.github.io/langgraph/agents/memory/)

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/ML-GenAI.git

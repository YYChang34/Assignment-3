# Assignment 3: Autonomous Multi-Doc Financial Analyst

A state-aware RAG system built with LangGraph that analyzes Apple and Tesla financial reports (10-K / Q4 2024).

## Architecture

```
User Query
    │
    ▼
[Task B] Intelligent Router ──► Apple DB / Tesla DB / Both / None
    │
    ▼
[Task C] Relevance Grader
    │
    ├─ yes ──► [Task E] Final Generator ──► Answer
    │
    └─ no  ──► [Task D] Query Rewriter ──► Router (retry, max 2×)
```

The Legacy ReAct agent (Task A / LangChain) provides a baseline single-chain comparison.

## Project Structure

```
.
├── data/                    # PDF source documents
├── chroma_db/               # Vector databases
├── evaluation_log/          # Evaluation run logs
├── state.py                 # Shared AgentState TypedDict
├── config.py                # LLM and embedding configuration
├── task_a.py                # Task A: LangChain ReAct agent (baseline)
├── task_b.py                # Task B: Intelligent router node
├── task_c.py                # Task C: Relevance grader node
├── task_d.py                # Task D: Query rewriter node
├── task_e.py                # Task E: Final generator node
├── langgraph_agent.py       # Graph orchestrator (Tasks B–E) + legacy wrapper
├── build_rag.py             # ETL pipeline: PDF → ChromaDB
├── evaluator.py             # 14-case benchmark evaluation framework
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variables template
```

## Tasks Implemented

| Task | File | Description |
|------|------|-------------|
| A | `task_a.py` | LangChain ReAct agent — English-only, year precision, honesty constraints |
| B | `task_b.py` | LLM-based router → apple / tesla / both / none |
| C | `task_c.py` | Binary relevance grader: yes / no |
| D | `task_d.py` | Query rewriter using SEC 10-K terminology |
| E | `task_e.py` | Final generator with strict source citation |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env_example .env
```

### 3. Add financial PDF documents

Place the following files in the `data/` directory:

- `FY24_Q4_Consolidated_Financial_Statements.pdf` (Apple)
- `tsla-20241231-gen.pdf` (Tesla)

### 4. Build vector databases

```bash
# Default: all-MiniLM-L6-v2, chunk_size=2000
python build_rag.py

# Specify model and chunk size
python build_rag.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --chunk-size 2000

# Build all experiment configurations (6 combinations: 2 models × 3 chunk sizes)
python build_rag.py --all-experiments
```

### 5. Run evaluation

```bash
# Evaluate LangGraph agent (Tasks B–E) — set TEST_MODE = "GRAPH" in evaluator.py
python evaluator.py

# Evaluate Legacy ReAct agent (Task A) — set TEST_MODE = "LEGACY" in evaluator.py
python evaluator.py
```

Logs are saved to `evaluation_log/evaluation_log_YYYYMMDD_HHMM.txt`.

## Embedding Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| `all-MiniLM-L6-v2` | ~22M | English financial text (default, 14/14 score) |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~118M | Multilingual queries (12/14 score) |

## Evaluation Results

Final score using `all-MiniLM-L6-v2`, `chunk_size=2000`, `k=8`, LangGraph GRAPH mode:

| Agent | Score |
|-------|-------|
| LangChain ReAct (Task A baseline) | 6 / 14  (43%) |
| LangGraph (Tasks B–E) | **14 / 14  (100%)** |

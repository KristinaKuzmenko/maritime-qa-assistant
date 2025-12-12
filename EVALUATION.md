# RAG Evaluation Guide

## Overview

This guide explains how to evaluate the Maritime QA Assistant using RAGAS and custom metrics.

## Dataset Format

Evaluation dataset (`evaluation.json`) should contain questions with ground truth answers:

```json
[
  {
    "question": "What are the main functional sections of the incinerator?",
    "answer": {
      "answer_text": "The incinerator consists of...",
      "citations": [
        {
          "type": "text",
          "doc_id": "8130",
          "section_id": "2.1",
          "page": 6,
          "title": "The Incinerator",
          "doc_title": "MAXI T50 SL WS Instruction Book"
        }
      ],
      "figures": [
        {
          "schema_id": "basic-principle-diagram",
          "title": "Basic Principles Layout",
          "caption": "Diagram showing...",
          "url": "page6-diagram",
          "page": 6,
          "doc_title": "MAXI T50 SL WS Instruction Book"
        }
      ],
      "tables": [...]
    }
  }
]
```

## Metrics

### RAGAS Standard Metrics

1. **Faithfulness** (0-1)
   - Measures if answer is grounded in retrieved context
   - Penalizes hallucinations
   - Higher is better

2. **Answer Relevancy** (0-1)
   - Measures if answer addresses the question
   - Checks semantic alignment
   - Higher is better

3. **Context Precision** (0-1)
   - Measures if retrieved context is relevant
   - Checks if top results are useful
   - Higher is better

### Custom Metrics

**Tool Usage Analysis:**

For each tool (qdrant_search_text, qdrant_search_tables, qdrant_search_schemas, neo4j_entity_search, neo4j_query):
- **Expected**: How many questions required this tool (from expected_tools in dataset)
- **Used**: How many times agent actually called this tool
- **Correct**: Intersection of expected and used
- **Precision**: correct / used (tool call accuracy)
- **Recall**: correct / expected (tool coverage)
- **F1**: Harmonic mean of precision and recall

**Example:**
```
qdrant_search_text:
   Expected: 45, Used: 48, Correct: 43
   Precision: 89.6%  (43/48 - agent didn't over-call)
   Recall:    95.6%  (43/45 - agent didn't miss calls)
   F1:        0.925
```

**Resource Inclusion Metrics:**

1. **Schema Inclusion Score** (F1: 0-1)
   - Precision: % of returned schemas that should be there
   - Recall: % of expected schemas that were returned
   - F1: Harmonic mean
   - Measures if agent correctly identifies when to show diagrams

2. **Table Inclusion Score** (F1: 0-1)
   - Same as schema inclusion for tables
   - Measures if agent correctly identifies when to show tables

3. **Citation Accuracy** (F-beta: 0-1)
   - **Precision**: % of returned citations that are correct (soft penalty)
   - **Recall**: % of expected citations that were returned (strict penalty)
   - **F-beta (beta=2)**: Weighted harmonic mean favoring recall (2x importance)
   - **Soft penalty for extra citations**: Allows up to 50% extra citations without penalty
   - **Why F-beta**: Better to over-cite than under-cite; missing mandatory citations is critical
   - Matches citations by (doc_title, page) tuples

4. **Latency** (milliseconds)
   - **Average**: Mean response time across all questions
   - **Median**: 50th percentile latency
   - **P95**: 95th percentile (acceptable worst-case)
   - **P99**: 99th percentile (outliers)
   - **Min/Max**: Fastest and slowest responses
   - Measured from workflow invocation to final answer

## Installation

```bash
# Install evaluation dependencies
pip install -r requirements-eval.txt
```

### Question Type Analysis

The evaluation pipeline analyzes metrics by question type:

- **text**: Text-only questions (explanations, procedures)
- **table**: Questions requiring table data (specs, troubleshooting)
- **schema**: Questions requiring diagrams (visual representations)
- **mixed**: Complex queries needing multiple sources

**Metrics per type:**
- Schema F1 average
- Table F1 average  
- Citation accuracy average
- Answer rate (% of questions answered)

**Example output:**
```
Question Type: text (25 questions)
   Answer Rate:       100.0%
   Schema F1:         0.950
   Table F1:          0.850
   Citation Accuracy: 0.780

Question Type: table (15 questions)
   Answer Rate:       93.3%
   Schema F1:         0.600
   Table F1:          0.920
   Citation Accuracy: 0.850
```

---

## Running Evaluation

### Basic Usage

```bash
# Run evaluation on default dataset
python backend/evaluate_rag.py

# Specify custom dataset and output
python backend/evaluate_rag.py evaluation.json results.json
```

### Programmatic Usage

```python
import asyncio
from backend.evaluate_rag import evaluate_rag_system

results = asyncio.run(
    evaluate_rag_system(
        eval_data_path="evaluation.json",
        output_path="results.json",
        owner="test_owner",
        doc_ids=["8130"]  # Optional: limit to specific docs
    )
)

print(f"Schema F1: {results['custom_metrics']['schema_inclusion']['f1']:.3f}")
print(f"Table F1: {results['custom_metrics']['table_inclusion']['f1']:.3f}")
```

## Results Format

Results are saved in JSON format:

```json
{
  "metadata": {
    "eval_dataset": "evaluation.json",
    "num_examples": 60,
    "owner": "test_owner",
    "doc_ids": null,
    "timestamp": "2025-01-12T10:30:00"
  },
  "custom_metrics": {
    "schema_inclusion": {
      "precision": 0.95,
      "recall": 0.90,
      "f1": 0.92
    },
    "table_inclusion": {
      "precision": 0.88,
      "recall": 0.85,
      "f1": 0.86
    },
    "citation_accuracy": {
      "precision": 0.82,
      "recall": 0.95,
      "f1": 0.88
    }
  },
  "latency_stats": {
    "avg": 4250,
    "median": 3800,
    "p50": 3800,
    "p95": 7200,
    "p99": 9500,
    "min": 1200,
    "max": 12000
  },
  "tool_analysis": {
    "tool_usage_count": {
      "qdrant_search_text": 48,
      "qdrant_search_tables": 22,
      "qdrant_search_schemas": 18,
      "neo4j_entity_search": 12,
      "neo4j_query": 3
    },
    "tool_metrics": {
      "qdrant_search_text": {
        "expected": 45,
        "used": 48,
        "correct": 43,
        "precision": 0.896,
        "recall": 0.956,
        "f1": 0.925
      },
      "qdrant_search_tables": {
        "expected": 20,
        "used": 22,
        "correct": 18,
        "precision": 0.818,
        "recall": 0.900,
        "f1": 0.857
      }
    }
  },
  "type_analysis": {
    "text": {
      "count": 25,
      "answer_rate": 1.0,
      "schema_f1_avg": 0.95,
      "table_f1_avg": 0.85,
      "citation_accuracy_avg": 0.78
    },
    "table": {
      "count": 15,
      "answer_rate": 0.933,
      "schema_f1_avg": 0.60,
      "table_f1_avg": 0.92,
      "citation_accuracy_avg": 0.85
    }
  },
  "ragas_metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.88,
    "context_precision": 0.82,
    "context_recall": 0.79
  },
  "per_question_results": [...]
}
```

## Visualization

The evaluation pipeline automatically generates visualization plots:

### 1. Metrics by Question Type

**File**: `evaluation_plots/metrics_by_type.png`

4-panel chart showing:
- Schema Inclusion F1 by type
- Table Inclusion F1 by type
- Citation Accuracy by type
- Answer Rate by type

### 2. Tool Usage Analysis

**File**: `evaluation_plots/tool_analysis.png`

2-panel chart:
- **Left**: Horizontal bar chart of tool call frequency
- **Right**: Grouped bar chart of Precision/Recall/F1 per tool

### 3. Overall Metrics Comparison

**File**: `evaluation_plots/overall_metrics.png`

Bar chart comparing:
- Schema F1
- Table F1
- Citation F1
- Faithfulness (if RAGAS enabled)
- Answer Relevancy (if RAGAS enabled)
- Context Precision (if RAGAS enabled)
- Context Recall (if RAGAS enabled)

**Usage:**
```bash
python backend/evaluate_rag.py
# Plots saved to evaluation_plots/ directory
```

---

## Creating Evaluation Datasets

### Manual Curation

1. Select diverse questions covering:
   - Text-only answers
   - Questions requiring diagrams
   - Questions requiring tables
   - Equipment code queries (PU3, CP1, etc.)
   - Procedural questions

2. Run questions through the system manually

3. Review and correct answers, citations, figures, tables

4. Save in `evaluation.json` format

### Semi-Automatic Approach

```python
# Generate candidate answers
import asyncio
from backend.evaluate_rag import run_agent_on_question
from backend.workflow import create_workflow

workflow = create_workflow()

questions = [
    "What are the main functional sections?",
    "Show me the fuel connections diagram",
    "What is PU3?"
]

for q in questions:
    answer = asyncio.run(run_agent_on_question(workflow, q))
    print(f"\nQ: {q}")
    print(f"A: {answer['answer_text'][:100]}...")
    print(f"Figures: {len(answer['figures'])}")
    print(f"Tables: {len(answer['tables'])}")
    
    # Manually review and add to evaluation.json
```

## Best Practices

### Dataset Quality

- **Diversity**: Cover all question types (text, schema, table, mixed)
- **Difficulty**: Include easy, medium, hard questions
- **Specificity**: Include specific equipment codes and general concepts
- **Language**: Test both English and Russian questions

### Evaluation Frequency

- **After major changes**: workflow logic, retrieval strategy
- **Before releases**: ensure quality standards met
- **Regression testing**: catch performance degradation

### Metric Interpretation

**Tool Usage Metrics:**

**High Tool F1 (>0.85):**
- qdrant_search_text: F1=0.925 ✅ (best tool, most reliable)
- Agent correctly selects tools based on question type
- Good alignment between intent and tool calls

**Low Tool F1 (<0.60):**
- qdrant_search_schemas: F1=0.400 ⚠️ (low recall - agent not calling schemas enough)
- Check router agent prompts for schema keyword detection
- Review intent classification ("schema" intent not triggering tool)
- Verify entity hints not overshadowing schema needs

**Precision vs Recall Trade-off:**
- High Precision, Low Recall: Agent too conservative (missing tool calls)
- Low Precision, High Recall: Agent too aggressive (over-calling tools)

**Resource Inclusion Metrics:**

**High Schema/Table F1 (>0.85):**
- Agent correctly identifies when visual aids needed
- Good intent classification
- LLM properly references [DIAGRAM]/[TABLE] markers

**Low Schema/Table F1 (<0.60):**
- Check intent classification logic
- Review router agent prompts
- Verify Qdrant schema/table indexing
- Check LLM reasoning prompt (intent-based constraints)

**Citation Accuracy:**

**High Citation F1 (>0.80):**
- Correct F-beta=0.818 with recall=0.900 ✅
- Agent finding mandatory citations (high recall)
- Some extra citations okay (soft precision penalty)

**Low Citation F1 (<0.60):**
- Check if missing mandatory citations (recall issue)
- Review citation parsing logic
- Verify Neo4j metadata accuracy

**High Faithfulness (>0.80):**
- Answers grounded in context
- Minimal hallucination

**Low Faithfulness (<0.60):**
- Agent making unsupported claims
- Review LLM reasoning prompts
- Check if context is sufficient

**High Citation Accuracy (>0.75):**
- Correct source attribution
- Good citation extraction

**Low Citation Accuracy (<0.50):**
- Check citation parsing logic
- Review Neo4j metadata accuracy


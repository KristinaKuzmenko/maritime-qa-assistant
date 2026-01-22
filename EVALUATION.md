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

For each tool (qdrant_search_text, qdrant_search_tables, qdrant_search_schemas, neo4j_entity_search):
- **Expected**: How many questions required this tool (from expected_tools in dataset)
- **Used**: How many times agent actually called this tool
- **Correct**: Intersection of expected and used
- **Precision**: correct / used (tool call accuracy)
- **Recall**: correct / expected (tool coverage)
- **F1**: Harmonic mean of precision and recall

**Example:**
```
qdrant_search_text:
   Expected: 59, Used: 57, Correct: 56
   Precision: 98.2%  (56/57 - agent didn't over-call)
   Recall:    94.9%  (56/59 - agent didn't miss calls)
   F1:        0.966
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
     - Formula: `allowed_extra = max(1, int(len(expected_refs) * 0.5))`
     - Example: If 4 citations expected, up to 2 extra allowed (6 total) without penalty
   - **Why F-beta**: Better to over-cite than under-cite; missing mandatory citations is critical
   - Matches citations by (doc_title, page) tuples
   
   **Edge Cases:**
   - **No citations expected, none returned**: Perfect score (1.0, 1.0, 1.0)
   - **No citations expected, but returned**: Partial precision penalty (0.5, 1.0, 0.67)
   - **Citations expected, none returned**: CRITICAL FAILURE (0.0, 0.0, 0.0) - missing mandatory sources

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

**Metrics per type:**
- Schema F1 average
- Table F1 average  
- Citation accuracy average
- Answer rate (% of questions answered)

**Example output:**
```
Question Type: text (40 questions)
   Answer Rate:       100.0%
   Schema F1:         0.000
   Table F1:          0.000
   Citation Accuracy: 0.874

Question Type: table (10 questions)
   Answer Rate:       100.0%
   Schema F1:         0.000
   Table F1:          0.467
   Citation Accuracy: 0.661

Question Type: schema (12 questions)
   Answer Rate:       100.0%
   Schema F1:         0.736
   Table F1:          0.000
   Citation Accuracy: 0.782
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
    "num_examples": 62,
    "owner": null,
    "doc_ids": null
  },
  "custom_metrics": {
    "schema_inclusion": {
      "precision": 0.957,
      "recall": 0.957,
      "f1": 0.949
    },
    "table_inclusion": {
      "precision": 0.911,
      "recall": 0.919,
      "f1": 0.866
    },
    "citation_accuracy": {
      "precision": 0.898,
      "recall": 0.829,
      "f1": 0.822
    }
  },
  "latency_stats": {
    "avg": 7604,
    "median": 2974,
    "p50": 2988,
    "p95": 61430,
    "p99": 65859,
    "min": 1803,
    "max": 65859
  },
  "tool_analysis": {
    "tool_usage_count": {
      "qdrant_search_text": 57,
      "neo4j_entity_search": 23,
      "qdrant_search_tables": 22,
      "qdrant_search_schemas": 13
    },
    "tool_metrics": {
      "qdrant_search_text": {
        "expected": 59,
        "used": 57,
        "correct": 56,
        "precision": 0.982,
        "recall": 0.949,
        "f1": 0.966
      },
      "neo4j_entity_search": {
        "expected": 10,
        "used": 23,
        "correct": 7,
        "precision": 0.304,
        "recall": 0.700,
        "f1": 0.424
      },
      "qdrant_search_tables": {
        "expected": 19,
        "used": 22,
        "correct": 13,
        "precision": 0.591,
        "recall": 0.684,
        "f1": 0.634
      },
      "qdrant_search_schemas": {
        "expected": 14,
        "used": 13,
        "correct": 13,
        "precision": 1.000,
        "recall": 0.929,
        "f1": 0.963
      }
    }
  },
  "type_analysis": {
    "text": {
      "count": 40,
      "schema_count": 0,
      "table_count": 0,
      "schema_f1_avg": 0.0,
      "table_f1_avg": 0.0,
      "citation_accuracy_avg": 0.874,
      "answer_rate": 1.0
    },
    "table": {
      "count": 10,
      "schema_count": 0,
      "table_count": 10,
      "schema_f1_avg": 0.0,
      "table_f1_avg": 0.467,
      "citation_accuracy_avg": 0.661,
      "answer_rate": 1.0
    },
    "schema": {
      "count": 12,
      "schema_count": 12,
      "table_count": 0,
      "schema_f1_avg": 0.736,
      "table_f1_avg": 0.0,
      "citation_accuracy_avg": 0.782,
      "answer_rate": 1.0
    }
  },
  "ragas_metrics": {
    "faithfulness": 0.802,
    "answer_relevancy": 0.836,
    "context_precision": 0.848,
    "context_recall": 0.777
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
- qdrant_search_text: F1=0.966 ✅ (best tool, most reliable)
- qdrant_search_schemas: F1=0.963 ✅ (excellent precision)
- Agent correctly selects tools based on question type
- Good alignment between intent and tool calls

**Low Tool F1 (<0.60):**
- neo4j_entity_search: F1=0.424 ⚠️ (low precision - agent over-calling entities)
- qdrant_search_tables: F1=0.634 ⚠️ (moderate - needs improvement)
- Check router agent prompts for entity vs text search
- Review intent classification (entity search triggered too often)
- Verify table search keyword detection

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
- Correct F1=0.822 with recall=0.829 ✅
- Agent finding most mandatory citations (good recall)
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


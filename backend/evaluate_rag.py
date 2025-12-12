"""
RAG Evaluation Pipeline with RAGAS and Custom Metrics

Evaluates Maritime QA Assistant using:
- RAGAS standard metrics (faithfulness, answer relevancy, context precision)
- Custom metrics (schema/table inclusion accuracy)
- Question type analysis (text/table/schema/mixed)
- Tool usage analysis (which tools were called)
- Result visualization and breakdown
"""

import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict, Counter

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from openai import OpenAI
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root for relative imports
os.chdir(project_root)

from backend.workflow import build_qa_graph, GraphState
from backend.core.config import settings
from backend.services.vector_service import VectorService
from backend.services.embedding_service import EmbeddingService
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase



# Custom Metrics for Schema/Table Evaluation

class SchemaTableMetric:
    """
    Custom metric to evaluate if agent correctly included schemas/tables
    when they were expected in the ground truth.
    
    Measures:
    - Precision: % of returned schemas/tables that should be there
    - Recall: % of expected schemas/tables that were returned
    - F1: Harmonic mean of precision and recall
    """
    
    def __init__(self, metric_type: str = "schema"):
        """
        :param metric_type: 'schema' or 'table'
        """
        self.metric_type = metric_type
        self.name = f"{metric_type}_inclusion_score"
    
    def _extract_ids(self, item: Dict) -> set:
        """Extract (url, page) tuples from answer dict"""
        if self.metric_type == "schema":
            return {(fig.get("url", ""), fig.get("page", 0)) for fig in item.get("figures", [])}
        else:  # table
            return {(tbl.get("url", ""), tbl.get("page", 0)) for tbl in item.get("tables", [])}
    
    def score(self, ground_truth: Dict, prediction: Dict) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 for schema/table inclusion.
        
        :param ground_truth: Expected answer dict with figures/tables
        :param prediction: Agent's answer dict with figures/tables
        :return: Dict with precision, recall, f1
        """
        expected_ids = self._extract_ids(ground_truth)
        predicted_ids = self._extract_ids(prediction)
        
        if len(expected_ids) == 0 and len(predicted_ids) == 0:
            # Both empty - perfect match
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if len(expected_ids) == 0:
            # Expected none, but got some - false positives
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        
        if len(predicted_ids) == 0:
            # Expected some, but got none - false negatives
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
        
        # Calculate precision and recall
        true_positives = len(expected_ids & predicted_ids)
        precision = true_positives / len(predicted_ids) if predicted_ids else 0
        recall = true_positives / len(expected_ids) if expected_ids else 0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class CitationAccuracyMetric:
    """
    Custom metric to evaluate citation quality using precision/recall/F1.
    
    Checks if citations reference the correct documents and sections/tables/schemas.
    Uses F1 score to balance precision (not citing wrong docs) and recall (citing all relevant docs).
    """
    
    def score(self, ground_truth: Dict, prediction: Dict) -> Dict[str, float]:
        """
        Calculate citation accuracy with focus on mandatory citations.
        
        Scoring logic:
        - Recall (mandatory): Penalize heavily for missing required citations
        - Precision (soft): Penalize lightly for extra citations (better to over-cite than under-cite)
        - F1: Weighted average favoring recall (beta=2)
        
        Returns dict with precision, recall, and f1 scores.
        """
        expected_citations = ground_truth.get("citations", [])
        predicted_citations = prediction.get("citations", [])
        
        # Edge cases
        if len(expected_citations) == 0 and len(predicted_citations) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if len(expected_citations) == 0:
            # Should not cite, but did - not critical (extra context is okay)
            # Give partial credit for precision
            return {"precision": 0.5, "recall": 1.0, "f1": 0.67}
        
        if len(predicted_citations) == 0:
            # Should cite, but didn't - CRITICAL ERROR (missing mandatory citations)
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Extract citation keys: (doc_title, page)
        # Only consider document name and page number for citation matching
        def extract_citation_key(citation: Dict) -> tuple:
            doc_title = citation.get("doc_title", "")
            page = citation.get("page")
            
            # Normalize page to int if possible, otherwise keep as string
            if page is not None:
                try:
                    page = int(page)
                except (ValueError, TypeError):
                    pass
            
            return (doc_title, page)
        
        expected_refs = {extract_citation_key(c) for c in expected_citations}
        predicted_refs = {extract_citation_key(c) for c in predicted_citations}
        
        # Calculate TP, FP, FN
        tp = len(expected_refs & predicted_refs)  # Correct citations (MANDATORY)
        fp = len(predicted_refs - expected_refs)  # Extra citations (minor penalty)
        fn = len(expected_refs - predicted_refs)  # Missing citations (major penalty)
        
        # Calculate recall (mandatory coverage) - unchanged
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate precision with soft penalty for extra citations
        # Allow up to 50% extra citations without penalty
        allowed_extra = max(1, int(len(expected_refs) * 0.5))
        penalized_fp = max(0, fp - allowed_extra)
        
        precision = tp / (tp + penalized_fp) if (tp + penalized_fp) > 0 else 1.0
        
        # Calculate F-beta score with beta=2 (recall 2x more important than precision)
        beta = 2.0
        if precision + recall > 0:
            f1 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        else:
            f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }



# Evaluation Pipeline

async def run_agent_on_question(
    workflow,
    question: str,
    user_id: str = "eval_user",
    owner: str = "test_owner",
    doc_ids: List[str] = None,
    qdrant_client = None,
    neo4j_driver = None
) -> Dict[str, Any]:
    """
    Run workflow on a single question and return structured answer + metadata.
    
    :param workflow: LangGraph workflow
    :param question: User question
    :param user_id: User identifier
    :param owner: Owner filter for retrieval
    :param doc_ids: Document IDs to search (optional)
    :return: Dict with 'answer', 'contexts', 'tools_used', 'query_intent'
    """
    initial_state = GraphState(
        user_id=user_id,
        question=question,
        chat_history=[],
        owner=owner,
        doc_ids=doc_ids,
        query_intent="",
        messages=[],
        anchor_sections=[],
        search_results={"text": [], "tables": [], "schemas": []},
        neo4j_results=[],
        entity_results=None,
        enriched_context=[],
        answer={
            "answer_text": "",
            "citations": [],
            "figures": [],
            "tables": []
        }
    )
    
    result = await workflow.ainvoke(initial_state)
    
    # Extract tool usage from messages (deduplicate to avoid counting forced calls multiple times)
    from langchain_core.messages import AIMessage
    tools_used = []
    seen_tool_calls = set()
    
    for msg in result.get("messages", []):
        # Only count tool calls from AIMessage (not ToolMessage responses)
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name")
                tool_id = tc.get("id")
                # Deduplicate by (tool_name, tool_id) to avoid counting same call multiple times
                if tool_name and (tool_name, tool_id) not in seen_tool_calls:
                    tools_used.append(tool_name)
                    seen_tool_calls.add((tool_name, tool_id))
    
    # Extract contexts from enriched_context
    # Different item types have different text keys:
    # - text_chunk: "text"
    # - table_chunk: "text"
    # - schema: "llm_summary" or "text_context"
    contexts = []
    for ctx in result.get("enriched_context", []):
        ctx_type = ctx.get("type", "")
        if ctx_type == "text_chunk":
            contexts.append(ctx.get("text", ""))
        elif ctx_type == "table_chunk":
            contexts.append(ctx.get("text", ""))
        elif ctx_type == "schema":
            # Prefer llm_summary, fallback to text_context
            contexts.append(ctx.get("llm_summary") or ctx.get("text_context", ""))
    
    return {
        "answer": result["answer"],
        "contexts": contexts,
        "tools_used": tools_used,
        "query_intent": result.get("query_intent", "")
    }


def load_evaluation_dataset(json_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from JSON file.
    
    Expected format:
    [
        {
            "question": str,
            "answer": {
                "answer_text": str,
                "citations": [...],
                "figures": [...],
                "tables": [...]
            },
            "expected_type": {
                "primary_type": str,
                "expected_tools": [...]
            }
        }
    ]
    
    :param json_path: Path to evaluation.json
    :return: List of evaluation examples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def extract_ground_truth_contexts(answer: Dict[str, Any]) -> List[str]:
    """
    Extract context snippets from ground truth answer citations.
    
    Since we don't have the exact text chunks stored in evaluation.json,
    we use citation text and section titles as proxy for ground truth contexts.
    
    :param answer: Ground truth answer dict with citations
    :return: List of context strings
    """
    contexts = []
    
    # Extract from citations
    for citation in answer.get("citations", []):
        # Use section title as context indicator
        section = citation.get("section_title", "")
        text = citation.get("text", "")
        
        if text:
            contexts.append(text)
        elif section:
            contexts.append(f"Section: {section}")
    
    # If no contexts from citations, use answer text as fallback
    if not contexts and answer.get("answer_text"):
        contexts.append(answer["answer_text"])
    
    return contexts


def prepare_ragas_dataset(
    questions: List[str],
    ground_truths: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truth_contexts: List[List[str]] = None
) -> Dataset:
    """
    Prepare dataset in RAGAS format.
    
    :param questions: List of questions
    :param ground_truths: List of ground truth answers (answer_text)
    :param answers: List of agent answers (answer_text)
    :param contexts: List of context lists (retrieved chunks for each question)
    :param ground_truth_contexts: List of ground truth context lists (for context_recall)
    :return: HuggingFace Dataset for RAGAS
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    
    # Add ground_truth contexts if provided (for context_recall metric)
    if ground_truth_contexts:
        data["ground_truth_contexts"] = ground_truth_contexts
    
    return Dataset.from_dict(data)


# Analysis Functions

def analyze_by_question_type(
    eval_data: List[Dict],
    predictions: List[Dict],
    schema_scores: List[Dict],
    table_scores: List[Dict],
    citation_scores: List[Dict]
) -> Dict[str, Any]:
    """
    Analyze metrics by question type (text/table/schema/mixed).
    """
    type_metrics = defaultdict(lambda: {
        "count": 0,
        "schema_f1": [],
        "table_f1": [],
        "citation_accuracy": [],
        "has_answer": 0,
    })
    
    for i, example in enumerate(eval_data):
        q_type = example.get("expected_type", {}).get("primary_type", "unknown")
        
        type_metrics[q_type]["count"] += 1
        type_metrics[q_type]["schema_f1"].append(schema_scores[i]["f1"])
        type_metrics[q_type]["table_f1"].append(table_scores[i]["f1"])
        type_metrics[q_type]["citation_accuracy"].append(citation_scores[i]["f1"])
        
        if predictions[i]["answer_text"]:
            type_metrics[q_type]["has_answer"] += 1
    
    # Aggregate
    aggregated = {}
    for q_type, metrics in type_metrics.items():
        aggregated[q_type] = {
            "count": metrics["count"],
            "schema_f1_avg": sum(metrics["schema_f1"]) / len(metrics["schema_f1"]) if metrics["schema_f1"] else 0,
            "table_f1_avg": sum(metrics["table_f1"]) / len(metrics["table_f1"]) if metrics["table_f1"] else 0,
            "citation_accuracy_avg": sum(metrics["citation_accuracy"]) / len(metrics["citation_accuracy"]) if metrics["citation_accuracy"] else 0,
            "answer_rate": metrics["has_answer"] / metrics["count"] if metrics["count"] > 0 else 0,
        }
    
    return aggregated


def analyze_tool_usage(
    eval_data: List[Dict],
    tools_usage: List[List[str]],
    predictions: List[Dict]
) -> Dict[str, Any]:
    """
    Analyze which tools were used and their correlation with success.
    """
    # Count tool usage
    tool_counter = Counter()
    for tools in tools_usage:
        tool_counter.update(tools)
    
    # Analyze by expected tools
    tool_accuracy = defaultdict(lambda: {"expected": 0, "used": 0, "correct": 0})
    
    for i, example in enumerate(eval_data):
        expected_tools = set(example.get("expected_type", {}).get("expected_tools", []))
        actual_tools = set(tools_usage[i])
        
        for tool in expected_tools:
            tool_accuracy[tool]["expected"] += 1
            if tool in actual_tools:
                tool_accuracy[tool]["correct"] += 1
        
        for tool in actual_tools:
            tool_accuracy[tool]["used"] += 1
    
    # Calculate precision/recall for each tool
    tool_metrics = {}
    for tool, counts in tool_accuracy.items():
        precision = counts["correct"] / counts["used"] if counts["used"] > 0 else 0
        recall = counts["correct"] / counts["expected"] if counts["expected"] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        tool_metrics[tool] = {
            "expected": counts["expected"],
            "used": counts["used"],
            "correct": counts["correct"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    return {
        "tool_usage_count": dict(tool_counter),
        "tool_metrics": tool_metrics,
    }


def print_type_analysis(type_analysis: Dict[str, Any]):
    """Print question type analysis."""
    print("\nMetrics by Question Type:")
    print("-" * 80)
    
    for q_type in ["text", "table", "schema", "mixed"]:
        if q_type in type_analysis:
            metrics = type_analysis[q_type]
            print(f"\n{q_type.upper()} questions ({metrics['count']} total):")
            print(f"   Answer Rate:       {metrics['answer_rate']:.1%}")
            print(f"   Schema F1:         {metrics['schema_f1_avg']:.3f}")
            print(f"   Table F1:          {metrics['table_f1_avg']:.3f}")
            print(f"   Citation Accuracy: {metrics['citation_accuracy_avg']:.3f}")


def print_tool_analysis(tool_analysis: Dict[str, Any]):
    """Print tool usage analysis."""
    print("\nTool Usage Count:")
    for tool, count in sorted(tool_analysis["tool_usage_count"].items(), key=lambda x: -x[1]):
        print(f"   {tool}: {count}")
    
    print("\nTool Accuracy (Precision/Recall/F1):")
    print("-" * 80)
    for tool, metrics in sorted(tool_analysis["tool_metrics"].items()):
        print(f"\n{tool}:")
        print(f"   Expected: {metrics['expected']}, Used: {metrics['used']}, Correct: {metrics['correct']}")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall:    {metrics['recall']:.1%}")
        print(f"   F1:        {metrics['f1']:.3f}")


def visualize_results(results: Dict[str, Any], output_dir: str = "evaluation_plots"):
    """
    Create visualizations for evaluation results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Metrics by question type
    type_analysis = results["type_analysis"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metrics by Question Type", fontsize=16, fontweight='bold')
    
    types = list(type_analysis.keys())
    schema_f1 = [type_analysis[t]["schema_f1_avg"] for t in types]
    table_f1 = [type_analysis[t]["table_f1_avg"] for t in types]
    citation_acc = [type_analysis[t]["citation_accuracy_avg"] for t in types]
    answer_rate = [type_analysis[t]["answer_rate"] for t in types]
    
    axes[0, 0].bar(types, schema_f1, color='skyblue')
    axes[0, 0].set_title('Schema Inclusion F1')
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].bar(types, table_f1, color='lightcoral')
    axes[0, 1].set_title('Table Inclusion F1')
    axes[0, 1].set_ylim(0, 1)
    
    axes[1, 0].bar(types, citation_acc, color='lightgreen')
    axes[1, 0].set_title('Citation Accuracy')
    axes[1, 0].set_ylim(0, 1)
    
    axes[1, 1].bar(types, answer_rate, color='plum')
    axes[1, 1].set_title('Answer Rate')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_by_type.png", dpi=300, bbox_inches='tight')
    print(f"   📊 Saved: {output_dir}/metrics_by_type.png")
    plt.close()
    
    # 2. Tool usage and accuracy
    tool_analysis = results["tool_analysis"]
    tool_metrics = tool_analysis["tool_metrics"]
    
    tools = list(tool_metrics.keys())
    precision = [tool_metrics[t]["precision"] for t in tools]
    recall = [tool_metrics[t]["recall"] for t in tools]
    f1 = [tool_metrics[t]["f1"] for t in tools]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tool Usage Analysis", fontsize=16, fontweight='bold')
    
    # Tool usage count
    usage_count = tool_analysis["tool_usage_count"]
    tools_sorted = sorted(usage_count.items(), key=lambda x: -x[1])
    ax1.barh([t[0] for t in tools_sorted], [t[1] for t in tools_sorted], color='steelblue')
    ax1.set_xlabel('Usage Count')
    ax1.set_title('Tool Call Frequency')
    
    # Tool accuracy
    x = range(len(tools))
    width = 0.25
    ax2.bar([i - width for i in x], precision, width, label='Precision', color='skyblue')
    ax2.bar(x, recall, width, label='Recall', color='lightcoral')
    ax2.bar([i + width for i in x], f1, width, label='F1', color='lightgreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tools, rotation=45, ha='right')
    ax2.set_ylabel('Score')
    ax2.set_title('Tool Selection Accuracy')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tool_analysis.png", dpi=300, bbox_inches='tight')
    print(f"   📊 Saved: {output_dir}/tool_analysis.png")
    plt.close()
    
    # 3. Overall metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Schema F1', 'Table F1', 'Citation F1']
    metrics_values = [
        results["custom_metrics"]["schema_inclusion"]["f1"],
        results["custom_metrics"]["table_inclusion"]["f1"],
        results["custom_metrics"]["citation_accuracy"]["f1"],
    ]
    
    if results["ragas_metrics"]:
        metrics_names.extend(['Faithfulness', 'Relevancy', 'Ctx Precision', 'Ctx Recall'])
        metrics_values.extend([
            results["ragas_metrics"].get("faithfulness", 0),
            results["ragas_metrics"].get("answer_relevancy", 0),
            results["ragas_metrics"].get("context_precision", 0),
            results["ragas_metrics"].get("context_recall", 0),
        ])
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'gold', 'lightblue']
    ax.bar(metrics_names, metrics_values, color=colors[:len(metrics_names)])
    ax.set_ylabel('Score')
    ax.set_title('Overall Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_metrics.png", dpi=300, bbox_inches='tight')
    print(f"   📊 Saved: {output_dir}/overall_metrics.png")
    plt.close()


async def evaluate_rag_system(
    eval_data_path: str = "evaluation.json",
    output_path: str = "evaluation_results.json",
    owner: str = "test_owner",
    doc_ids: List[str] = None,
    num_questions: int = None
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Steps:
    1. Load ground truth data
    2. Run agent on each question
    3. Calculate RAGAS metrics (faithfulness, relevancy, precision)
    4. Calculate custom metrics (schema/table inclusion)
    5. Analyze by question type and tool usage
    6. Generate visualizations
    7. Save results
    
    :param eval_data_path: Path to evaluation.json
    :param output_path: Path to save results
    :param owner: Owner filter for retrieval
    :param doc_ids: Document IDs to evaluate on
    :param num_questions: Number of questions to test (None = all questions)
    :return: Evaluation results dict
    """
    print("=" * 80)
    print("🔍 MARITIME QA ASSISTANT - RAG EVALUATION")
    print("=" * 80)
    
    # Load evaluation dataset
    print(f"\n📂 Loading evaluation data from: {eval_data_path}")
    eval_data = load_evaluation_dataset(eval_data_path)
    
    # Limit number of questions if specified
    if num_questions is not None and num_questions > 0:
        eval_data = eval_data[:num_questions]
        print(f"✅ Loaded {len(eval_data)} evaluation examples (limited to {num_questions})")
    else:
        print(f"✅ Loaded {len(eval_data)} evaluation examples")
    
    # Initialize workflow
    print("\n🤖 Initializing workflow...")
    
    # Create clients
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    embedding_service = EmbeddingService(api_key=settings.openai_api_key)
    vector_service = VectorService(embedding_service=embedding_service)
    
    workflow = build_qa_graph(qdrant_client, neo4j_driver, vector_service)
    print("✅ Workflow ready")
    print("📋 Entity detection will be initialized on first question")
    
    # Run agent on all questions
    print(f"\n🚀 Running agent on {len(eval_data)} questions...")
    predictions = []
    ground_truths = []
    contexts_list = []
    tools_usage = []
    query_intents = []
    
    for idx, example in enumerate(eval_data, 1):
        question = example["question"]
        ground_truth = example["answer"]
        expected_type = example.get("expected_type", {})
        
        print(f"\n[{idx}/{len(eval_data)}] Question: {question[:80]}...")
        print(f"   Expected type: {expected_type.get('primary_type', 'unknown')}")
        
        # Run workflow
        try:
            result = await run_agent_on_question(
                workflow=workflow,
                question=question,
                owner=owner,
                doc_ids=doc_ids,
                qdrant_client=qdrant_client,
                neo4j_driver=neo4j_driver
            )
            
            prediction = result["answer"]
            contexts = result["contexts"]
            tools = result["tools_used"]
            intent = result["query_intent"]
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            contexts_list.append(contexts)
            tools_usage.append(tools)
            query_intents.append(intent)
            
            print(f"   ✅ Answer generated ({len(prediction['answer_text'])} chars)")
            print(f"      Citations: {len(prediction['citations'])}")
            print(f"      Figures: {len(prediction['figures'])}")
            print(f"      Tables: {len(prediction['tables'])}")
            print(f"      Tools used: {', '.join(tools) if tools else 'none'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            # Add empty prediction to maintain alignment
            predictions.append({
                "answer_text": "",
                "citations": [],
                "figures": [],
                "tables": []
            })
            ground_truths.append(ground_truth)
            contexts_list.append([])
            tools_usage.append([])
            query_intents.append("")
    
    # Calculate custom metrics
    print("\n" + "=" * 80)
    print("📊 CALCULATING CUSTOM METRICS")
    print("=" * 80)
    
    schema_metric = SchemaTableMetric("schema")
    table_metric = SchemaTableMetric("table")
    citation_metric = CitationAccuracyMetric()
    
    schema_scores = []
    table_scores = []
    citation_scores = []
    
    for gt, pred in zip(ground_truths, predictions):
        schema_scores.append(schema_metric.score(gt, pred))
        table_scores.append(table_metric.score(gt, pred))
        citation_scores.append(citation_metric.score(gt, pred))
    
    # Aggregate custom metrics
    custom_results = {
        "schema_inclusion": {
            "precision": sum(s["precision"] for s in schema_scores) / len(schema_scores),
            "recall": sum(s["recall"] for s in schema_scores) / len(schema_scores),
            "f1": sum(s["f1"] for s in schema_scores) / len(schema_scores),
        },
        "table_inclusion": {
            "precision": sum(t["precision"] for t in table_scores) / len(table_scores),
            "recall": sum(t["recall"] for t in table_scores) / len(table_scores),
            "f1": sum(t["f1"] for t in table_scores) / len(table_scores),
        },
        "citation_accuracy": {
            "precision": sum(c["precision"] for c in citation_scores) / len(citation_scores),
            "recall": sum(c["recall"] for c in citation_scores) / len(citation_scores),
            "f1": sum(c["f1"] for c in citation_scores) / len(citation_scores),
        }
    }
    
    print("\n📈 Custom Metrics Results:")
    print(f"   Schema Inclusion F1: {custom_results['schema_inclusion']['f1']:.3f}")
    print(f"   Table Inclusion F1:  {custom_results['table_inclusion']['f1']:.3f}")
    print(f"   Citation Accuracy:   P={custom_results['citation_accuracy']['precision']:.3f}, R={custom_results['citation_accuracy']['recall']:.3f}, F1={custom_results['citation_accuracy']['f1']:.3f}")
    
    # Analyze by question type
    print("\n" + "=" * 80)
    print("📊 ANALYSIS BY QUESTION TYPE")
    print("=" * 80)
    
    type_analysis = analyze_by_question_type(
        eval_data, predictions, schema_scores, table_scores, citation_scores
    )
    
    print_type_analysis(type_analysis)
    
    # Analyze tool usage
    print("\n" + "=" * 80)
    print("🔧 TOOL USAGE ANALYSIS")
    print("=" * 80)
    
    tool_analysis = analyze_tool_usage(eval_data, tools_usage, predictions)
    
    print_tool_analysis(tool_analysis)
    
    # Prepare data for RAGAS
    print("\n" + "=" * 80)
    print("📊 CALCULATING RAGAS METRICS")
    print("=" * 80)
    
    questions = [ex["question"] for ex in eval_data]
    gt_texts = [gt["answer_text"] for gt in ground_truths]
    pred_texts = [pred["answer_text"] for pred in predictions]
    
    # Extract ground truth contexts from citations
    gt_contexts_list = [extract_ground_truth_contexts(gt) for gt in ground_truths]
    
    ragas_dataset = prepare_ragas_dataset(
        questions=questions,
        ground_truths=gt_texts,
        answers=pred_texts,
        contexts=contexts_list,
        ground_truth_contexts=gt_contexts_list
    )
    
    # Run RAGAS evaluation
    print("\n⏳ Running RAGAS evaluation (this may take a few minutes)...")
    print("   Using model: gpt-4o-mini (cost-effective)")
    print("   Note: RAGAS warnings about '1 generations instead of 3' are expected and non-critical")
    
    # Configure LLM and Embeddings for RAGAS (modern API)
    from langchain_openai import ChatOpenAI
    
    # Use LangChain's ChatOpenAI for better RAGAS compatibility
    ragas_llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=settings.openai_api_key,
        temperature=0
    )
    
    # Create embeddings using modern RAGAS OpenAIEmbeddings (direct import)
    try:
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        ragas_embeddings = RagasOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key
        )
    except Exception as e:
        # Fallback to LangchainEmbeddingsWrapper if modern API not available
        print(f"   ⚠️ Modern RAGAS embeddings failed ({e}), using LangChain wrapper")
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings
        langchain_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    try:
        ragas_results = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,  # Now enabled with ground_truth contexts
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        # Convert to dict - use correct method based on RAGAS version
        if hasattr(ragas_results, 'to_pandas'):
            try:
                # Convert to pandas then to dict
                df = ragas_results.to_pandas()
                # Filter only numeric columns for mean calculation
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                ragas_scores = {col: float(df[col].mean()) for col in numeric_cols}
            except Exception as e:
                print(f"   ⚠️  Error converting to pandas: {e}")
                # Fallback: try to extract scores directly
                ragas_scores = {}
                for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    if hasattr(ragas_results, metric):
                        ragas_scores[metric] = getattr(ragas_results, metric)
        elif hasattr(ragas_results, 'scores'):
            ragas_scores = ragas_results.scores
        else:
            # Fallback - try direct dict conversion
            ragas_scores = dict(ragas_results) if isinstance(ragas_results, dict) else {}
        
        if ragas_scores:
            print("\n📈 RAGAS Metrics Results:")
            print(f"   Faithfulness:       {ragas_scores.get('faithfulness', 0):.3f}")
            print(f"   Answer Relevancy:   {ragas_scores.get('answer_relevancy', 0):.3f}")
            print(f"   Context Precision:  {ragas_scores.get('context_precision', 0):.3f}")
            print(f"   Context Recall:     {ragas_scores.get('context_recall', 0):.3f}")
        else:
            print("\n⚠️  RAGAS metrics could not be extracted")
        
    except Exception as e:
        print(f"\n⚠️  RAGAS evaluation failed: {e}")
        import traceback
        print(traceback.format_exc())
        ragas_scores = {}
    
    # Compile results
    results = {
        "metadata": {
            "eval_dataset": eval_data_path,
            "num_examples": len(eval_data),
            "owner": owner,
            "doc_ids": doc_ids,
        },
        "custom_metrics": custom_results,
        "ragas_metrics": ragas_scores,
        "type_analysis": type_analysis,
        "tool_analysis": tool_analysis,
        "per_question_results": [
            {
                "question": eval_data[i]["question"],
                "expected_type": eval_data[i].get("expected_type", {}),
                "ground_truth": ground_truths[i],
                "prediction": predictions[i],
                "schema_score": schema_scores[i],
                "table_score": table_scores[i],
                "citation_score": citation_scores[i],
                "tools_used": tools_usage[i],
                "query_intent": query_intents[i],
            }
            for i in range(len(eval_data))
        ]
    }
    
    # Save results
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    try:
        visualize_results(results)
        print("✅ Visualizations saved to: evaluation_plots/")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Cleanup connections
    print("\n🔌 Closing connections...")
    try:
        await neo4j_driver.close()
        qdrant_client.close()
        print("✅ Connections closed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    
    return results



# CLI Interface

if __name__ == "__main__":
    import sys
    import os
    
    # Parse arguments with smart detection
    eval_file = "evaluation.json"
    output_file = "evaluation_results.json"
    num_questions = None
    
    # Smart argument parsing:
    # - If single arg that's a number or "all" -> num_questions
    # - If single arg that's a file path -> eval_file
    # - Standard: [eval_file] [output_file] [num_questions]
    
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        # Check if it's a number or "all"
        if arg.lower() == "all" or arg.isdigit():
            num_questions = None if arg.lower() == "all" else int(arg)
        else:
            # Assume it's an eval file path
            eval_file = arg
    elif len(sys.argv) == 3:
        arg1, arg2 = sys.argv[1], sys.argv[2]
        # Check if arg2 is num_questions
        if arg2.lower() == "all" or arg2.isdigit():
            eval_file = arg1
            num_questions = None if arg2.lower() == "all" else int(arg2)
        else:
            # Standard: eval_file, output_file
            eval_file = arg1
            output_file = arg2
    elif len(sys.argv) >= 4:
        eval_file = sys.argv[1]
        output_file = sys.argv[2]
        num_questions_arg = sys.argv[3]
        if num_questions_arg.lower() == "all":
            num_questions = None
        else:
            try:
                num_questions = int(num_questions_arg)
            except ValueError:
                print(f"⚠️  Invalid num_questions argument: {num_questions_arg}")
                print("   Usage: python evaluate_rag.py [num_questions|all]")
                print("          python evaluate_rag.py [eval_file] [num_questions|all]")
                print("          python evaluate_rag.py [eval_file] [output_file] [num_questions|all]")
                sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Evaluation file: {eval_file}")
    print(f"   Output file: {output_file}")
    print(f"   Number of questions: {num_questions if num_questions else 'all'}")
    
    # Run evaluation
    results = asyncio.run(
        evaluate_rag_system(
            eval_data_path=eval_file,
            output_path=output_file,
            owner=None,  # Search all owners
            doc_ids=None,  # Evaluate on all documents
            num_questions=num_questions
        )
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n✅ Evaluated {results['metadata']['num_examples']} questions")
    print(f"\n🎯 Custom Metrics:")
    print(f"   Schema Inclusion F1: {results['custom_metrics']['schema_inclusion']['f1']:.3f}")
    print(f"   Table Inclusion F1:  {results['custom_metrics']['table_inclusion']['f1']:.3f}")
    citation_acc = results['custom_metrics']['citation_accuracy']
    print(f"   Citation Accuracy:   P={citation_acc['precision']:.3f}, R={citation_acc['recall']:.3f}, F1={citation_acc['f1']:.3f}")
    
    if results['ragas_metrics']:
        print(f"\n📈 RAGAS Metrics:")
        for metric, score in results['ragas_metrics'].items():
            print(f"   {metric}: {score:.3f}")
    
    print(f"\n💾 Results saved to: {output_file}")

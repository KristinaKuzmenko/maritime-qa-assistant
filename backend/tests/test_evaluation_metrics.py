"""
Tests for evaluation metrics and dataset preparation functions.
Tests real function behavior with proper signatures.
"""

import pytest
from unittest.mock import Mock, patch


class TestEvaluationImports:
    """Test importing evaluation modules."""
    
    def test_import_evaluation_module(self):
        """Test importing main evaluation module."""
        import evaluate_rag
        assert evaluate_rag is not None
    
    def test_import_helper_functions(self):
        """Test importing specific functions."""
        from evaluate_rag import (
            SchemaTableMetric,
            CitationAccuracyMetric,
            prepare_ragas_dataset,
            calculate_latency_stats,
            analyze_by_question_type,
            analyze_tool_usage
        )
        
        assert SchemaTableMetric is not None
        assert CitationAccuracyMetric is not None
        assert prepare_ragas_dataset is not None
        assert calculate_latency_stats is not None
        assert analyze_by_question_type is not None
        assert analyze_tool_usage is not None


class TestSchemaTableMetric:
    """Test custom schema/table inclusion metric."""
    
    def test_create_schema_metric(self):
        """Test creating schema metric."""
        from evaluate_rag import SchemaTableMetric
        
        metric = SchemaTableMetric(metric_type="schema")
        
        assert metric.metric_type == "schema"
        assert metric.name == "schema_inclusion_score"
    
    def test_create_table_metric(self):
        """Test creating table metric."""
        from evaluate_rag import SchemaTableMetric
        
        metric = SchemaTableMetric(metric_type="table")
        
        assert metric.metric_type == "table"
        assert metric.name == "table_inclusion_score"
    
    def test_score_perfect_match(self):
        """Test scoring with perfect schema match."""
        from evaluate_rag import SchemaTableMetric
        
        metric = SchemaTableMetric(metric_type="schema")
        
        ground_truth = {
            "figures": [
                {"url": "/schemas/schema1.png", "page": 15},
                {"url": "/schemas/schema2.png", "page": 20},
            ]
        }
        
        prediction = {
            "figures": [
                {"url": "/schemas/schema1.png", "page": 15},
                {"url": "/schemas/schema2.png", "page": 20},
            ]
        }
        
        score = metric.score(ground_truth, prediction)
        
        assert isinstance(score, dict)
        assert "precision" in score
        assert "recall" in score
        assert "f1" in score
        assert score["f1"] == 1.0  # Perfect match
    
    def test_score_partial_match(self):
        """Test scoring with partial match."""
        from evaluate_rag import SchemaTableMetric
        
        metric = SchemaTableMetric(metric_type="table")
        
        ground_truth = {
            "tables": [
                {"url": "/tables/table1.csv", "page": 10},
                {"url": "/tables/table2.csv", "page": 12},
            ]
        }
        
        prediction = {
            "tables": [
                {"url": "/tables/table1.csv", "page": 10},  # Match
                {"url": "/tables/table3.csv", "page": 15},  # Extra (false positive)
            ]
        }
        
        score = metric.score(ground_truth, prediction)
        
        assert isinstance(score, dict)
        assert score["precision"] == 0.5  # 1 correct out of 2 predicted
        assert score["recall"] == 0.5  # 1 correct out of 2 expected
    
    def test_score_empty_both(self):
        """Test scoring when both ground truth and prediction are empty."""
        from evaluate_rag import SchemaTableMetric
        
        metric = SchemaTableMetric(metric_type="schema")
        
        score = metric.score({"figures": []}, {"figures": []})
        
        assert score["f1"] == 1.0  # Empty match is perfect


class TestCitationAccuracyMetric:
    """Test citation accuracy metric."""
    
    def test_create_citation_metric(self):
        """Test creating citation accuracy metric."""
        from evaluate_rag import CitationAccuracyMetric
        
        metric = CitationAccuracyMetric()
        
        # Test that metric can score citations
        assert hasattr(metric, 'score')
        assert callable(metric.score)
        
        # Test scoring with empty data
        result = metric.score({}, {})
        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
    
    def test_score_perfect_citations(self):
        """Test scoring with perfect citations."""
        from evaluate_rag import CitationAccuracyMetric
        
        metric = CitationAccuracyMetric()
        
        ground_truth = {
            "text_chunks": [
                {"doc_id": "doc1", "page": 15},
                {"doc_id": "doc1", "page": 16},
            ]
        }
        
        prediction = {
            "text_chunks": [
                {"doc_id": "doc1", "page": 15},
                {"doc_id": "doc1", "page": 16},
            ]
        }
        
        score = metric.score(ground_truth, prediction)
        
        assert isinstance(score, dict)
        assert "precision" in score
        assert "recall" in score
        assert score["f1"] == 1.0


class TestDatasetLoading:
    """Test dataset loading functions."""
    
    def test_load_evaluation_dataset(self):
        """Test loading evaluation dataset from JSON."""
        from evaluate_rag import load_evaluation_dataset
        import tempfile
        import json
        
        # Create temporary test dataset
        test_data = [
            {
                "id": 1,
                "question": "What is the fuel pump maintenance interval?",
                "answer": {"text": "6 months", "text_chunks": []},
                "question_type": "text"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = load_evaluation_dataset(temp_path)
            
            assert isinstance(dataset, list)
            assert len(dataset) == 1
            assert dataset[0]["question"] == "What is the fuel pump maintenance interval?"
        finally:
            import os
            os.unlink(temp_path)
    
    def test_prepare_ragas_dataset(self):
        """Test preparing dataset for RAGAS evaluation."""
        from evaluate_rag import prepare_ragas_dataset
        
        questions = ["How does the fuel pump work?"]
        ground_truths = ["The fuel pump transfers fuel from tank to engine."]
        answers = ["The fuel pump transfers fuel."]
        contexts = [["Fuel pump operation details..."]]
        
        # Real signature: prepare_ragas_dataset(questions, ground_truths, answers, contexts)
        dataset = prepare_ragas_dataset(
            questions=questions,
            ground_truths=ground_truths,
            answers=answers,
            contexts=contexts
        )
        
        # Returns HuggingFace Dataset object
        assert dataset is not None
        assert hasattr(dataset, '__getitem__')  # Dataset is iterable
        assert len(dataset) == 1
        
        # Check first item
        item = dataset[0]
        assert item["question"] == questions[0]
        assert item["answer"] == answers[0]
        assert item["contexts"] == contexts[0]
        assert item["ground_truth"] == ground_truths[0]


class TestLatencyStats:
    """Test latency statistics calculation."""
    
    def test_calculate_latency_stats(self):
        """Test calculating latency statistics."""
        from evaluate_rag import calculate_latency_stats
        
        latencies = [1.5, 2.0, 1.8, 2.2, 1.9]
        
        stats = calculate_latency_stats(latencies)
        
        assert isinstance(stats, dict)
        # Returns: avg, median, p50, p95, p99, min, max
        assert "avg" in stats
        assert "median" in stats
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "min" in stats
        assert "max" in stats
        
        # Check values are reasonable
        assert stats["avg"] == sum(latencies) / len(latencies)
        assert stats["median"] == 1.9
        assert stats["min"] == 1.5
        assert stats["max"] == 2.2
        assert stats["p95"] > 0
        assert stats["p99"] > 0


class TestQuestionTypeAnalysis:
    """Test question type analysis."""
    
    def test_analyze_by_question_type(self):
        """Test analyzing results by question type."""
        from evaluate_rag import analyze_by_question_type
        
        # Real signature: analyze_by_question_type(eval_data, predictions, schema_scores, table_scores, citation_scores)
        eval_data = [
            {"expected_type": {"primary_type": "text"}, "ground_truth": {"text_chunks": []}},
            {"expected_type": {"primary_type": "table"}, "ground_truth": {"tables": [{"url": "/t1.csv"}]}}
        ]
        predictions = [
            {"answer_text": "Answer 1"},
            {"answer_text": "Answer 2"}
        ]
        schema_scores = [{"f1": 0.8}, {"f1": 0.9}]
        table_scores = [{"f1": 0.7}, {"f1": 0.85}]
        citation_scores = [{"f1": 0.9}, {"f1": 0.95}]
        
        result = analyze_by_question_type(
            eval_data=eval_data,
            predictions=predictions,
            schema_scores=schema_scores,
            table_scores=table_scores,
            citation_scores=citation_scores
        )
        
        assert isinstance(result, dict)
        # Should have metrics for each question type
        assert "text" in result
        assert "table" in result
        # Each type should have aggregated metrics
        assert "count" in result["text"]
        assert result["text"]["count"] == 1
        assert result["table"]["count"] == 1


class TestToolUsageAnalysis:
    """Test tool usage analysis."""
    
    def test_analyze_tool_usage(self):
        """Test analyzing which tools were used."""
        from evaluate_rag import analyze_tool_usage
        
        # Real signature: analyze_tool_usage(eval_data, tools_usage, predictions)
        eval_data = [
            {"expected_type": {"expected_tools": ["search_text"]}},
            {"expected_type": {"expected_tools": ["search_schemas", "search_tables"]}}
        ]
        tools_usage = [
            ["search_text"],
            ["search_schemas", "search_tables"]
        ]
        predictions = [
            {"answer_text": "Answer 1"},
            {"answer_text": "Answer 2"}
        ]
        
        result = analyze_tool_usage(
            eval_data=eval_data,
            tools_usage=tools_usage,
            predictions=predictions
        )
        
        assert isinstance(result, dict)
        # Should have tool usage statistics
        assert len(result) > 0  # Has some analysis data

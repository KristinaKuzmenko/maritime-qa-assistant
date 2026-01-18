"""
Q&A Chat endpoints using LangGraph workflow.
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging

from middleware.rate_limiter import role_rate_limit
from core.prompt_injection_filter import PromptInjectionFilter
from core.dependencies import QueryServices
from core.exceptions import PromptInjectionError, ProcessingError, ServiceUnavailableError
from helpers.response_transformer import transform_response_urls_sync

logger = logging.getLogger(__name__)

# Initialize prompt injection filter
injection_filter = PromptInjectionFilter(strict_mode=True)

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question")
    user_id: str = Field(default="global", description="User identifier")
    chat_history: List[Dict[str, str]] = Field(default=[], description="Conversation history")
    owner: Optional[str] = Field(default=None, description="Filter by owner (e.g., user ID)")
    doc_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs")

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    metadata: Dict[str, Any]



# Chat Endpoints

@router.post("/answer", response_model=AnswerResponse)
@role_rate_limit("qa")
async def answer_question(
    request: Request, 
    question_req: QuestionRequest,
    services: QueryServices = Depends()
):
    """
    Answer a question using agentic LangGraph workflow.
    Chat history is passed from frontend (Streamlit session_state) and used
    to provide context for the current question.
    
    Agentic workflow features:
    1. Query intent detection
    2. Router Agent with tools (Qdrant + Neo4j)
    3. Anchor-based context filtering
    4. Neighbor chunk expansion
    5. Hard limits (3 sections, 3 tables, 3 schemas)
    """
    
    # SECURITY: Check for prompt injection attempts
    injection_check = injection_filter.check_query(question_req.question)
    
    if not injection_check.is_safe:
        logger.warning(
            f"🚨 Prompt injection detected: {injection_check.explanation}",
            extra={
                "user_id": question_req.user_id,
                "risk_level": injection_check.risk_level.value,  
                "patterns": injection_check.detected_patterns,
                "query_preview": question_req.question[:100]
            }
        )
        raise PromptInjectionError(
            message="Please ask a question about maritime technical documentation. "
                   "Your query contains content that cannot be processed.",
            risk_level=injection_check.risk_level.value,  
            detected_patterns=injection_check.detected_patterns,
            query_preview=question_req.question[:100]
        )
    
    # Use sanitized query for processing
    safe_question = injection_check.sanitized_query
    logger.info(
        f"✅ Query passed injection filter (risk: {injection_check.risk_level.value})"
    )
    
    try:
        # Prepare initial state for agentic workflow
        state = {
            "user_id": question_req.user_id,
            "question": question_req.question,
            "chat_history": question_req.chat_history,
            "owner": question_req.owner,
            "doc_ids": question_req.doc_ids,
            
            # These will be filled by workflow
            "query_intent": "text",  # Default
            "tool_names": [],
            "anchor_sections": [],
            "messages": [],
            "search_results": {"text": [], "tables": [], "schemas": []},
            "neo4j_results": [],
            "entity_results": None,
            "enriched_context": [],
            "retrieval_attempt": 0,  # Adaptive retry: 0 = first attempt, max 1 retry
            "answer": {},
        }
        
        logger.info(
            f"Processing question: {safe_question} "
            f"(owner={question_req.owner}, doc_ids={question_req.doc_ids})"
        )
        
        # Run agentic LangGraph workflow
        result = await services.qa_graph.ainvoke(state)
        
        # Extract answer data
        answer_data = result.get("answer", {})
        
        # Enhanced metadata with agentic workflow insights
        metadata = {
            # Query analysis
            "query_intent": result.get("query_intent", "text"),
            
            # Agent tool usage
            "tools_used": _extract_tools_used(result.get("messages", [])),
            
            # Anchor sections
            "anchor_sections": len(result.get("anchor_sections", [])),
            "anchor_details": [
                {
                    "doc_id": a.get("doc_id"),
                    "section_id": a.get("section_id"),
                    "score": round(a.get("score", 0), 3)
                }
                for a in result.get("anchor_sections", [])[:3]
            ],
            
            # Retrieval statistics
            "search_results": {
                "text": len(result.get("search_results", {}).get("text", [])),
                "tables": len(result.get("search_results", {}).get("tables", [])),
                "schemas": len(result.get("search_results", {}).get("schemas", [])),
            },
            "neo4j_results": len(result.get("neo4j_results", [])),
            
            # Final context (after filtering + limits)
            "final_context": {
                "chunks": len([c for c in result.get("enriched_context", []) if c.get("type") == "text_chunk"]),
                "tables": len([c for c in result.get("enriched_context", []) if c.get("type") == "table_chunk"]),
                "schemas": len([c for c in result.get("enriched_context", []) if c.get("type") == "schema"]),
            },
            
            # Quality indicators
            "neighbor_expansion": any(
                c.get("expanded") 
                for c in result.get("enriched_context", []) 
                if c.get("type") == "text_chunk"
            ),
            "anchor_filtering_applied": len(result.get("anchor_sections", [])) > 0,
        }
        
        response = AnswerResponse(
            answer=answer_data.get("answer_text", "No answer generated"),
            citations=answer_data.get("citations", []),
            tables=answer_data.get("tables", []),
            figures=answer_data.get("figures", []),
            metadata=metadata
        )
        
        # 🔄 Transform file paths to accessible URLs for S3 storage
        if services.storage:
            response_dict = response.dict()
            
            # Log BEFORE transformation
            if response_dict.get("figures"):
                logger.info(f"📥 BEFORE transform: First figure URL = {response_dict['figures'][0].get('url')}")
            
            response_dict = transform_response_urls_sync(
                response_dict,
                services.storage,
                expiration=3600  # 1 hour
            )
            
            # Log AFTER transformation
            if response_dict.get("figures"):
                logger.info(f"📤 AFTER transform: First figure URL = {response_dict['figures'][0].get('url')[:150]}...")
            
            response = AnswerResponse(**response_dict)
        
        logger.info(
            f"✅ Answer generated: intent={metadata['query_intent']}, "
            f"tools={metadata['tools_used']}, "
            f"anchors={metadata['anchor_sections']}, "
            f"context={metadata['final_context']}"
        )
        
        # Log what will be sent to client
        if hasattr(response, 'figures') and response.figures:
            logger.info(f"📡 Sending to client: {len(response.figures)} figures (type={type(response.figures)})")
            if isinstance(response.figures, list) and len(response.figures) > 0:
                first_fig = response.figures[0]
                if isinstance(first_fig, dict):
                    logger.info(f"   First figure URL: {first_fig.get('url', 'N/A')[:120]}...")
                else:
                    logger.info(f"   First figure type: {type(first_fig)}, has url: {hasattr(first_fig, 'url')}")
        
        return response
        
    except PromptInjectionError:
        # Re-raise already typed exceptions
        raise
    except Exception as e:
        logger.error(f"Q&A workflow error: {e}", exc_info=True)
        
        # Check if it's a rate limit error
        error_msg = str(e)
        if "429" in error_msg or "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
            raise ProcessingError(
                message=f"LLM service is experiencing high traffic. Please try again in a few moments. (Question: {question_req.question[:80]}...)"
            )
        
        raise ProcessingError(
            message=f"Failed to process query: {question_req.question[:100]}"
        )


def _extract_tools_used(messages: List) -> List[str]:
    """Extract list of tools used by agent from messages"""
    tools_used = []
    
    for msg in messages:
        # Safely get tool_calls
        tool_calls = None
        if hasattr(msg, 'tool_calls') and msg.tool_calls is not None:
            tool_calls = msg.tool_calls
        elif hasattr(msg, 'additional_kwargs'):
            tool_calls = msg.additional_kwargs.get('tool_calls', [])
        
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call["name"]
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
    
    return tools_used


@router.post("/debug")
@role_rate_limit("qa")
async def debug_workflow(
    request: Request, 
    question_req: QuestionRequest,
    services: QueryServices = Depends()
):
    """
    UPDATED debug endpoint for agentic workflow.
    
    ⚠️  Admin only - shows:
    - Query intent detection
    - Agent tool selection
    - Anchor section selection
    - Context filtering steps
    - Final context composition
    """
    
    # Check admin access
    user_role = getattr(request.state, "user_role", "guest")
    if user_role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Debug endpoint is only available for administrators"
        )
    
    try:
        state = {
            "user_id": question_req.user_id,
            "question": question_req.question,
            "chat_history": question_req.chat_history,
            "owner": question_req.owner,
            "doc_ids": question_req.doc_ids,
            "query_intent": "text",
            "anchor_sections": [],
            "messages": [],
            "tool_names": [],
            "search_results": {"text": [], "tables": [], "schemas": []},
            "neo4j_results": [],
            "entity_results": None,
            "enriched_context": [],
            "retrieval_attempt": 0,
            "answer": {},
        }
        
        result = await services.qa_graph.ainvoke(state)
        
        # Comprehensive debug output for agentic workflow
        return {
            "question": result.get("question"),
            
            # Step 1: Query Analysis
            "step_1_query_analysis": {
                "intent": result.get("query_intent", "text"),
            },
            
            # Step 2: Router Agent
            "step_2_router_agent": {
                "tools_called": _extract_tools_used(result.get("messages", [])),
                "agent_messages_count": len(result.get("messages", [])),
            },
            
            # Step 3: Tool Execution
            "step_3_tool_execution": {
                "search_text": len(result.get("search_results", {}).get("text", [])),
                "search_tables": len(result.get("search_results", {}).get("tables", [])),
                "search_schemas": len(result.get("search_results", {}).get("schemas", [])),
                "neo4j_records": len(result.get("neo4j_results", [])),
                "samples": {
                    "search_text": result.get("search_results", {}).get("text", [])[:2],
                    "neo4j": result.get("neo4j_results", [])[:2],
                }
            },
            
            # Step 4: Anchor Selection
            "step_4_anchor_selection": {
                "anchors_selected": len(result.get("anchor_sections", [])),
                "anchor_details": result.get("anchor_sections", []),
            },
            
            # Step 5: Context Building
            "step_5_context_building": {
                "total_enriched": len(result.get("enriched_context", [])),
                "by_type": {
                    "text_chunks": len([c for c in result.get("enriched_context", []) if c.get("type") == "text_chunk"]),
                    "tables": len([c for c in result.get("enriched_context", []) if c.get("type") == "table_chunk"]),
                    "schemas": len([c for c in result.get("enriched_context", []) if c.get("type") == "schema"]),
                },
                "expansion_applied": any(
                    c.get("expanded") 
                    for c in result.get("enriched_context", []) 
                    if c.get("type") == "text_chunk"
                ),
                "samples": result.get("enriched_context", [])[:2],
            },
            
            # Step 6: Answer Generation
            "step_6_answer": {
                "answer_text": result.get("answer", {}).get("answer_text", "")[:200] + "...",
                "citations_count": len(result.get("answer", {}).get("citations", [])),
                "tables_count": len(result.get("answer", {}).get("tables", [])),
                "figures_count": len(result.get("answer", {}).get("figures", [])),
            },
            
            # Full answer for display (compatible with normal response)
            "answer": result.get("answer", {}),
        }
        
    except Exception as e:
        logger.error(f"Debug workflow error: {e}", exc_info=True)
        raise ProcessingError(
            message=f"Debug workflow failed: {question_req.question[:100]}"
        )


@router.post("/analyze")
@role_rate_limit("qa")
async def analyze_query(
    request: Request, 
    question_req: QuestionRequest,
    services: QueryServices = Depends()
):
    """
    Analyze query intent without running full workflow.
    
    Useful for:
    - Understanding query classification
    - Testing intent detection
    - Preview what the workflow will do
    """
    
    try:
        # Import analysis function from workflow
        from workflow import node_analyze_and_route
        
        state = {
            "question": question_req.question,
            "user_id": question_req.user_id,
            "chat_history": question_req.chat_history,
            "owner": question_req.owner,
            "doc_ids": question_req.doc_ids,
            "query_intent": "text",
            "anchor_sections": [],
            "messages": [],
            "tool_names": [],
            "search_results": {"text": [], "tables": [], "schemas": []},
            "neo4j_results": [],
            "entity_results": None,
            "enriched_context": [],
            "retrieval_attempt": 0,
            "answer": {},
        }
        
        # Run only analysis step
        analyzed = node_analyze_and_route(state)
        
        return {
            "question": question_req.question,
            "analysis": {
                "intent": analyzed.get("query_intent"),
            },
            "explanation": {
                "intent": _explain_intent(analyzed.get("query_intent")),
            },
            "workflow_preview": {
                "text": "Will use Qdrant text search" if analyzed.get("query_intent") in ["text", "mixed"] else "Skipped",
                "tables": "Will use Qdrant table search" if analyzed.get("query_intent") in ["table", "mixed"] else "Skipped",
                "schemas": "Will use Qdrant schema search" if analyzed.get("query_intent") in ["schema", "mixed"] else "Skipped",
                "neo4j": "Agent will decide if Neo4j is needed",
            }
        }
        
    except Exception as e:
        logger.error(f"Query analysis error: {e}", exc_info=True)
        raise ProcessingError(
            message=f"Query analysis failed: {question_req.question[:100]}"
        )


def _explain_intent(intent: str) -> str:
    """Explain query intent for users."""
    explanations = {
        "text": "Question focuses on textual information and procedures",
        "table": "Question requires data from tables or specifications",
        "schema": "Question needs diagrams or visual schematics",
        "mixed": "Question may need multiple types of information + graph structure",
    }
    return explanations.get(intent, "Unknown intent")


@router.get("/stats")
async def get_qa_stats(services: QueryServices = Depends()):
    """
    Get Q&A system statistics.
    
    Shows agentic workflow features.
    """
    
    return {
        "status": "available",
        "workflow_type": "agentic",
        "features": {
            "query_routing": "✅ Active",
            "agent_with_tools": "✅ Active (Qdrant + Neo4j)",
            "neo4j_as_tool": "✅ Active (agent-controlled)",
            "anchor_sections": "✅ Active (max 3)",
            "neighbor_expansion": "✅ Active",
            "hard_limits": "✅ Active (3+3+3)",
        },
        "tools_available": [
            "qdrant_search_text",
            "qdrant_search_tables",
            "qdrant_search_schemas",
            "neo4j_query (read-only)"
        ],
    }


@router.get("/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """
    Get user's chat history.
    
    TODO: Implement chat history storage
    """
    return {
        "user_id": user_id,
        "chats": [],
        "message": "Chat history not yet implemented"
    }


@router.delete("/history/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Delete specific chat session.
    
    TODO: Implement chat deletion
    """
    return {
        "chat_id": chat_id,
        "message": "Chat deletion not yet implemented"
    }
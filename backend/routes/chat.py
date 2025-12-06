"""
Q&A Chat endpoints using LangGraph workflow.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging

from middleware.rate_limiter import role_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()

qa_graph = None
graph_client = None
qdrant_client = None
neo4j_driver = None


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
async def answer_question(request: Request, question_req: QuestionRequest):
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
    
    # Check if Q&A workflow is available
    if not qa_graph:
        raise HTTPException(
            status_code=503,
            detail="Q&A service not available. Check Neo4j and Qdrant connections."
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
            "anchor_sections": [],
            "messages": [],
            "qdrant_results": {"text": [], "tables": [], "schemas": []},
            "neo4j_results": [],
            "enriched_context": [],
            "answer": {},
        }
        
        logger.info(
            f"Processing question: {question_req.question} "
            f"(owner={question_req.owner}, doc_ids={question_req.doc_ids})"
        )
        
        # Run agentic LangGraph workflow
        result = await qa_graph.ainvoke(state)
        
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
            "qdrant_results": {
                "text": len(result.get("qdrant_results", {}).get("text", [])),
                "tables": len(result.get("qdrant_results", {}).get("tables", [])),
                "schemas": len(result.get("qdrant_results", {}).get("schemas", [])),
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
        
        logger.info(
            f"✅ Answer generated: intent={metadata['query_intent']}, "
            f"tools={metadata['tools_used']}, "
            f"anchors={metadata['anchor_sections']}, "
            f"context={metadata['final_context']}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Q&A workflow error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
@role_rate_limit("chat")
async def debug_workflow(request: Request, question_req: QuestionRequest):
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
    
    if not qa_graph:
        raise HTTPException(
            status_code=503,
            detail="Q&A service not available"
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
            "qdrant_results": {"text": [], "tables": [], "schemas": []},
            "neo4j_results": [],
            "enriched_context": [],
            "answer": {},
        }
        
        result = await qa_graph.ainvoke(state)
        
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
                "qdrant_text": len(result.get("qdrant_results", {}).get("text", [])),
                "qdrant_tables": len(result.get("qdrant_results", {}).get("tables", [])),
                "qdrant_schemas": len(result.get("qdrant_results", {}).get("schemas", [])),
                "neo4j_records": len(result.get("neo4j_results", [])),
                "samples": {
                    "qdrant_text": result.get("qdrant_results", {}).get("text", [])[:2],
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
        }
        
    except Exception as e:
        logger.error(f"Debug workflow error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
@role_rate_limit("chat")
async def analyze_query(request: Request, question_req: QuestionRequest):
    """
    Analyze query intent without running full workflow.
    
    Useful for:
    - Understanding query classification
    - Testing intent detection
    - Preview what the workflow will do
    """
    
    try:
        # Import analysis function from agentic workflow
        from workflow_agentic import node_analyze_question
        
        state = {
            "question": question_req.question,
            "user_id": question_req.user_id,
            "chat_history": question_req.chat_history,
            "owner": question_req.owner,
            "doc_ids": question_req.doc_ids,
            "query_intent": "text",
            "anchor_sections": [],
            "messages": [],
            "qdrant_results": {"text": [], "tables": [], "schemas": []},
            "neo4j_results": [],
            "enriched_context": [],
            "answer": {},
        }
        
        # Run only analysis step
        analyzed = node_analyze_question(state)
        
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
        raise HTTPException(status_code=500, detail=str(e))


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
async def get_qa_stats():
    """
    Get Q&A system statistics.
    
    Shows agentic workflow features.
    """
    
    if not qa_graph:
        raise HTTPException(
            status_code=503,
            detail="Q&A service not available"
        )
    
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
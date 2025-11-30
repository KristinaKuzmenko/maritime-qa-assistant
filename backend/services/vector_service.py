"""
Qdrant vector service for semantic search.
Manages embeddings and similarity operations with chunking support.
UPDATED: Added schema embedding support for layout-aware extraction.
"""

from typing import List, Dict, Any, Optional
import uuid
import logging
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    FilterSelector,
)

from core.config import settings
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorService:
    """Service for Qdrant vector database operations with chunking and schema support."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        client: Optional[QdrantClient] = None,
    ) -> None:
        """
        Initialize Qdrant vector service.
        
        :param embedding_service: Shared EmbeddingService instance
        :param client: Optional QdrantClient (for tests or custom config)
        """
        self.embeddings = embedding_service

        # Initialize Qdrant client
        if client is not None:
            self.client = client
        else:
            if getattr(settings, "qdrant_api_key", None):
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key,
                )
            else:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )

        # Collection names from settings
        self.text_collection = settings.text_chunks_collection
        self.schemas_collection = settings.schemas_collection  # NEW
        self.tables_text_collection = settings.tables_text_collection
        
        # Deprecated collections (kept for backward compatibility)
        self.figures_text_collection = getattr(settings, "figures_text_collection", "figures_text")
        self.figures_image_collection = getattr(settings, "figures_image_collection", "figures_image")

    # -------------------------------------------------------------------------
    # Collections Management
    # -------------------------------------------------------------------------

    def initialize_collections(self) -> None:
        """Create Qdrant collections if they do not exist."""
        existing = {c.name for c in self.client.get_collections().collections}

        dim = settings.vector_dimension
        distance = Distance.COSINE

        # Text chunks collection
        if self.text_collection not in existing:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=VectorParams(size=dim, distance=distance),
            )
            logger.info(f"Created collection: {self.text_collection}")

        # NEW: Schemas collection (replaces figures_text)
        if self.schemas_collection not in existing:
            self.client.create_collection(
                collection_name=self.schemas_collection,
                vectors_config=VectorParams(size=dim, distance=distance),
            )
            logger.info(f"Created collection: {self.schemas_collection}")

        # Tables collection for table chunks
        if self.tables_text_collection not in existing:
            self.client.create_collection(
                collection_name=self.tables_text_collection,
                vectors_config=VectorParams(size=dim, distance=distance),
            )
            logger.info(f"Created collection: {self.tables_text_collection}")
        
        # Backward compatibility: Keep old collections if they exist
        if self.figures_text_collection not in existing:
            self.client.create_collection(
                collection_name=self.figures_text_collection,
                vectors_config=VectorParams(size=dim, distance=distance),
            )
            logger.info(f"Created collection: {self.figures_text_collection} (legacy)")

    # -------------------------------------------------------------------------
    # Embeddings Helpers
    # -------------------------------------------------------------------------

    async def _embed(self, text: str) -> List[float]:
        """Get embedding via shared embedding service."""
        return await self.embeddings.create_embedding(text)

    # -------------------------------------------------------------------------
    # Upsert Operations - Text Chunks
    # -------------------------------------------------------------------------

    async def add_text_chunk(
        self,
        section_id: str,
        chunk_index: int,
        text: str,
        doc_id: str,
        doc_title: str,
        page_start: int,
        page_end: int,
        chunk_char_start: int,
        chunk_char_end: int,
        section_number: Optional[str] = None,
        section_title: Optional[str] = None,
        system_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
        owner: str = "global",
    ) -> None:
        """
        Add a text chunk (section fragment) to the vector database.
        Supports multiple chunks per section with overlap.
        
        :param section_id: Section ID
        :param chunk_index: Chunk index within section
        :param text: Chunk text content
        :param doc_id: Document ID
        :param doc_title: Document title
        :param page_start: Starting page
        :param page_end: Ending page
        :param chunk_char_start: Character start position
        :param chunk_char_end: Character end position
        :param section_number: Section number
        :param section_title: Section title
        :param system_ids: Related system IDs
        :param entity_ids: Related entity IDs
        :param owner: Owner identifier
        """
        # Validation BEFORE embedding
        token_estimate = len(text) // 4
        if token_estimate > 8000:
            logger.warning(
                f"Chunk too long ({token_estimate} tokens est.) for section {section_id}, "
                f"chunk {chunk_index}. Truncating."
            )
            text = text[:32000]  # ~8000 tokens fallback
        
        embedding = await self._embed(text)

        # Deterministic id per (section_id, chunk_index)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{section_id}_chunk_{chunk_index}"))

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "type": "text_chunk",
                "section_id": section_id,
                "chunk_index": chunk_index,
                "chunk_char_start": chunk_char_start,
                "chunk_char_end": chunk_char_end,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "section_number": section_number or "",
                "section_title": section_title or "",
                "page_start": page_start,
                "page_end": page_end,
                "system_ids": system_ids or [],
                "entity_ids": entity_ids or [],
                "owner": owner,
                "text": text,  # Store FULL chunk text in Qdrant
                "text_preview": text[:500],  # Keep preview for backward compatibility
                "char_count": len(text),
            },
        )

        # Non-blocking upsert
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.text_collection,
            points=[point],
            wait=True,
        )
        
        logger.debug(
            f"Added chunk {chunk_index} for section {section_id} "
            f"({chunk_char_start}-{chunk_char_end}, {len(text)} chars)"
        )

    # -------------------------------------------------------------------------
    # Upsert Operations - Schemas 
    # -------------------------------------------------------------------------

    async def add_schema_embedding(
        self,
        schema_id: str,
        text: str,
        doc_id: str,
        page: int,
        caption: Optional[str] = None,
        system_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
        owner: str = "global",
    ) -> None:
        """
        NEW: Add schema text embedding (caption + context) to vector database.
        
        Schemas are treated as atomic visual elements with text context for retrieval.
        No internal geometry parsing - just caption + surrounding text.
        
        :param schema_id: Schema ID from Neo4j
        :param text: Combined text (caption + context)
        :param doc_id: Document ID
        :param page: Page number
        :param caption: Schema caption
        :param system_ids: Related system IDs
        :param entity_ids: Related entity IDs
        :param owner: Owner identifier
        """
        # Validation
        token_estimate = len(text) // 4
        if token_estimate > 8000:
            logger.warning(
                f"Schema text too long ({token_estimate} tokens) for schema {schema_id}. "
                f"Truncating."
            )
            text = text[:32000]
        
        embedding = await self._embed(text)

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, schema_id))

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "type": "schema",
                "schema_id": schema_id,
                "doc_id": doc_id,
                "page": page,
                "caption": caption or "",
                "system_ids": system_ids or [],
                "entity_ids": entity_ids or [],
                "owner": owner,
                "text_preview": text[:500],
                "char_count": len(text),
            },
        )

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.schemas_collection,
            points=[point],
            wait=True,
        )
        
        logger.debug(f"Added schema embedding for {schema_id} ({len(text)} chars)")

    # -------------------------------------------------------------------------
    # Upsert Operations - Tables
    # -------------------------------------------------------------------------

    async def add_table_chunk(
        self,
        chunk_id: str,
        table_id: str,
        chunk_index: int,
        text: str,
        doc_id: str,
        page: int,
        table_title: Optional[str] = None,
        table_caption: Optional[str] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        total_chunks: Optional[int] = None,
        system_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
        owner: str = "global",
    ) -> None:
        """
        Add a single table chunk to Qdrant for granular semantic search.
        Each chunk represents a portion of table rows/content.
        
        :param chunk_id: TableChunk node ID from Neo4j
        :param table_id: Parent Table node ID
        :param chunk_index: Index of this chunk (0-based)
        :param text: Text content of this chunk (linearized table rows)
        :param doc_id: Document ID
        :param page: Page number
        :param table_title: Parent table title
        :param table_caption: Parent table caption
        :param rows: Total rows in parent table
        :param cols: Total columns in parent table
        :param total_chunks: Total number of chunks for this table
        :param system_ids: Linked system entities
        :param entity_ids: Linked component entities
        :param owner: Owner identifier for multi-tenancy
        """
        # Validation BEFORE embedding
        token_estimate = len(text) // 4
        if token_estimate > 8000:
            logger.error(
                f"Table chunk {chunk_id} exceeds token limit ({token_estimate} tokens). "
                f"This should not happen - table_extractor should have chunked properly!"
            )
            # Last resort truncation
            text = text[:32000]  # ~8000 tokens
        
        embedding = await self._embed(text)

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

        payload = {
            "type": "table_chunk",
            "chunk_id": chunk_id,
            "table_id": table_id,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks or 1,
            "doc_id": doc_id,
            "page": page,
            "table_title": table_title or "",
            "table_caption": table_caption or "",
            "rows": rows,
            "cols": cols,
            "system_ids": system_ids or [],
            "entity_ids": entity_ids or [],
            "owner": owner,
            "text_preview": text[:500],
            "char_count": len(text),
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.tables_text_collection,
            points=[point],
            wait=True,
        )
        
        logger.debug(
            f"Added table chunk {chunk_index+1}/{total_chunks} "
            f"for table {table_id} ({len(text)} chars)"
        )

    # -------------------------------------------------------------------------
    # Search Operations - Text
    # -------------------------------------------------------------------------

    async def search_text(
        self,
        query: str,
        limit: int = 10,
        doc_id: Optional[str] = None,
        owner: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search text chunks (section content) using semantic similarity.
        
        :param query: Search query text
        :param limit: Number of results
        :param doc_id: Filter by document
        :param owner: Filter by owner
        :param score_threshold: Minimum similarity score
        :return: List of matching chunks with metadata
        """
        logger.error("🚀🚀🚀 search_text called - CODE UPDATED! 🚀🚀🚀")
        query_embedding = await self._embed(query)

        # Build filter
        filters = []
        if doc_id:
            filters.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if owner:
            filters.append(FieldCondition(key="owner", match=MatchValue(value=owner)))

        search_filter = Filter(must=filters) if filters else None

        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.text_collection,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        # # DEBUG: Check first result payload
        # if results:
        #     first_hit = results[0]
        #     text_in_payload = 'text' in first_hit.payload
        #     text_value = first_hit.payload.get('text', '')
        #     text_len = len(text_value) if text_value else 0
        #     logger.error(
        #         f"🔍🔍🔍 search_text QDRANT RESPONSE | "
        #         f"payload_keys={list(first_hit.payload.keys())} | "
        #         f"has_text={text_in_payload} | "
        #         f"text_len={text_len} | "
        #         f"text_type={type(text_value)} | "
        #         f"text_is_empty={not bool(text_value)}"
        #     )

        formatted_results = [
            {
                "section_id": hit.payload["section_id"],
                "chunk_index": hit.payload["chunk_index"],
                "doc_id": hit.payload["doc_id"],
                "section_number": hit.payload.get("section_number", ""),
                "section_title": hit.payload.get("section_title", ""),
                "page_start": hit.payload["page_start"],
                "page_end": hit.payload["page_end"],
                "score": hit.score,
                "text": hit.payload.get("text", ""),  # Include full text
                "text_preview": hit.payload.get("text_preview", ""),
                "chunk_char_start": hit.payload.get("chunk_char_start", 0),
                "chunk_char_end": hit.payload.get("chunk_char_end", 0),
                "char_count": hit.payload.get("char_count", 0),
                "system_ids": hit.payload.get("system_ids", []),
                "entity_ids": hit.payload.get("entity_ids", []),
            }
            for hit in results
        ]
        
        # # DEBUG: Verify formatted results have text
        # if formatted_results:
        #     first_result = formatted_results[0]
        #     logger.error(
        #         f"🔍🔍🔍 search_text FORMATTED RESULT | "
        #         f"result_keys={list(first_result.keys())} | "
        #         f"has_text={'text' in first_result} | "
        #         f"text_len={len(first_result.get('text', ''))} | "
        #         f"text_empty={not bool(first_result.get('text'))}"
        #     )
        
        return formatted_results

    async def get_neighbor_chunks(
        self,
        section_id: str,
        chunk_index: int,
        neighbor_range: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get neighbor chunks from same section by chunk_index.
        
        :param section_id: Section ID
        :param chunk_index: Center chunk index
        :param neighbor_range: How many chunks before/after (default: ±1)
        :return: List of chunks sorted by chunk_index
        """
        # Calculate chunk index range
        min_index = max(0, chunk_index - neighbor_range)
        max_index = chunk_index + neighbor_range
        
        # Build filter for section_id and chunk_index range
        search_filter = Filter(
            must=[
                FieldCondition(key="section_id", match=MatchValue(value=section_id)),
            ]
        )
        
        # Scroll through all chunks of this section (should be small number)
        results = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.text_collection,
            scroll_filter=search_filter,
            limit=100,  # Max chunks per section
            with_payload=True,
            with_vectors=False,
        )
        
        # Filter by chunk_index range and sort
        chunks = []
        for point in results[0]:  # results is tuple (points, next_page_offset)
            idx = point.payload.get("chunk_index", 0)
            if min_index <= idx <= max_index:
                chunks.append({
                    "section_id": point.payload["section_id"],
                    "chunk_index": idx,
                    "doc_id": point.payload["doc_id"],
                    "section_title": point.payload.get("section_title", ""),
                    "page_start": point.payload["page_start"],
                    "page_end": point.payload["page_end"],
                    "text": point.payload.get("text", ""),
                    "chunk_char_start": point.payload.get("chunk_char_start", 0),
                    "chunk_char_end": point.payload.get("chunk_char_end", 0),
                })
        
        # Sort by chunk_index
        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

    # -------------------------------------------------------------------------
    # Search Operations - Schemas 
    # -------------------------------------------------------------------------

    async def search_schemas(
        self,
        query: str,
        limit: int = 10,
        doc_id: Optional[str] = None,
        owner: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        NEW: Search schemas by caption + context using semantic similarity.
        
        :param query: Search query text
        :param limit: Number of results
        :param doc_id: Filter by document
        :param owner: Filter by owner
        :param score_threshold: Minimum similarity score
        :return: List of matching schemas with metadata
        """
        query_embedding = await self._embed(query)

        # Build filter
        filters = []
        if doc_id:
            filters.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if owner:
            filters.append(FieldCondition(key="owner", match=MatchValue(value=owner)))

        search_filter = Filter(must=filters) if filters else None

        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.schemas_collection,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        return [
            {
                "schema_id": hit.payload["schema_id"],
                "doc_id": hit.payload["doc_id"],
                "page": hit.payload["page"],
                "caption": hit.payload.get("caption", ""),
                "score": hit.score,
                "text_preview": hit.payload.get("text_preview", ""),
                "char_count": hit.payload.get("char_count", 0),
                "system_ids": hit.payload.get("system_ids", []),
                "entity_ids": hit.payload.get("entity_ids", []),
            }
            for hit in results
        ]

    # -------------------------------------------------------------------------
    # Search Operations - Tables
    # -------------------------------------------------------------------------

    async def search_tables(
        self,
        query: str,
        limit: int = 10,
        doc_id: Optional[str] = None,
        owner: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search table chunks using semantic similarity.
        Returns individual chunks - caller can group by table_id if needed.
        
        :param query: Search query text
        :param limit: Number of chunk results
        :param doc_id: Filter by document
        :param owner: Filter by owner
        :param score_threshold: Minimum similarity score
        :return: List of matching table chunks with metadata
        """
        query_embedding = await self._embed(query)

        # Build filter
        filters = []
        if doc_id:
            filters.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if owner:
            filters.append(FieldCondition(key="owner", match=MatchValue(value=owner)))

        search_filter = Filter(must=filters) if filters else None

        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.tables_text_collection,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        return [
            {
                "chunk_id": hit.payload["chunk_id"],
                "table_id": hit.payload["table_id"],
                "chunk_index": hit.payload["chunk_index"],
                "total_chunks": hit.payload.get("total_chunks", 1),
                "doc_id": hit.payload["doc_id"],
                "page": hit.payload["page"],
                "table_title": hit.payload.get("table_title", ""),
                "table_caption": hit.payload.get("table_caption", ""),
                "rows": hit.payload.get("rows"),
                "cols": hit.payload.get("cols"),
                "score": hit.score,
                "text_preview": hit.payload.get("text_preview", ""),
                "char_count": hit.payload.get("char_count", 0),
                "system_ids": hit.payload.get("system_ids", []),
                "entity_ids": hit.payload.get("entity_ids", []),
            }
            for hit in results
        ]

    # -------------------------------------------------------------------------
    # Combined Search
    # -------------------------------------------------------------------------

    async def search_combined(
        self,
        query: str,
        doc_id: Optional[str] = None,
        owner: Optional[str] = None,
        limit_per_type: int = 5,
        score_threshold: float = 0.5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        NEW: Combined search across text, schemas, and tables.
        
        :param query: Search query text
        :param doc_id: Filter by document
        :param owner: Filter by owner
        :param limit_per_type: Results per type
        :param score_threshold: Minimum similarity score
        :return: Dict with results by type
        """
        # Run searches in parallel
        text_task = self.search_text(
            query=query,
            limit=limit_per_type,
            doc_id=doc_id,
            owner=owner,
            score_threshold=score_threshold,
        )
        
        schemas_task = self.search_schemas(
            query=query,
            limit=limit_per_type,
            doc_id=doc_id,
            owner=owner,
            score_threshold=score_threshold,
        )
        
        tables_task = self.search_tables(
            query=query,
            limit=limit_per_type,
            doc_id=doc_id,
            owner=owner,
            score_threshold=score_threshold,
        )
        
        text_results, schema_results, table_results = await asyncio.gather(
            text_task,
            schemas_task,
            tables_task,
        )
        
        return {
            "sections": text_results,
            "schemas": schema_results,
            "table_chunks": table_results,
        }

    # -------------------------------------------------------------------------
    # Search by Entities (filtering)
    # -------------------------------------------------------------------------

    async def search_by_entities(
        self,
        entity_ids: List[str],
        content_types: Optional[List[str]] = None,
        limit: int = 20,
        doc_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all collections for content mentioning specific entities.
        
        :param entity_ids: List of entity IDs to search for
        :param content_types: Filter by content type (text_chunk, schema, table_chunk)
        :param limit: Results per collection
        :param doc_id: Filter by document
        :return: Dictionary with results grouped by collection
        """
        if content_types is None:
            content_types = ["text_chunk", "schema", "table_chunk"]

        results = {}

        # Build filter
        filters = [FieldCondition(key="entity_ids", match=MatchAny(any=entity_ids))]
        if doc_id:
            filters.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

        search_filter = Filter(must=filters)

        # Search text chunks
        if "text_chunk" in content_types:
            text_results = self.client.scroll(
                collection_name=self.text_collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results["text_chunks"] = [
                {
                    "section_id": point.payload["section_id"],
                    "chunk_index": point.payload["chunk_index"],
                    "doc_id": point.payload["doc_id"],
                    "section_title": point.payload.get("section_title", ""),
                    "text_preview": point.payload.get("text_preview", ""),
                }
                for point in text_results[0]
            ]

        # Search schemas
        if "schema" in content_types:
            schema_results = self.client.scroll(
                collection_name=self.schemas_collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results["schemas"] = [
                {
                    "schema_id": point.payload["schema_id"],
                    "doc_id": point.payload["doc_id"],
                    "page": point.payload["page"],
                    "caption": point.payload.get("caption", ""),
                    "text_preview": point.payload.get("text_preview", ""),
                }
                for point in schema_results[0]
            ]

        # Search table chunks
        if "table_chunk" in content_types:
            table_results = self.client.scroll(
                collection_name=self.tables_text_collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results["table_chunks"] = [
                {
                    "chunk_id": point.payload["chunk_id"],
                    "table_id": point.payload["table_id"],
                    "chunk_index": point.payload["chunk_index"],
                    "doc_id": point.payload["doc_id"],
                    "page": point.payload["page"],
                    "table_title": point.payload.get("table_title", ""),
                    "text_preview": point.payload.get("text_preview", ""),
                }
                for point in table_results[0]
            ]

        return results

    # -------------------------------------------------------------------------
    # Similarity Computation (for SIMILAR_TO relationships)
    # -------------------------------------------------------------------------

    async def compute_section_similarities(
        self,
        doc_id: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Compute pairwise similarities between all sections in a document.
        Returns list of similarity relationships for Neo4j.

        NOTE: This operates on section-level, averaging all chunks per section.
        
        :param doc_id: Document ID
        :param top_k: Number of similar sections per section
        :param score_threshold: Minimum similarity score
        :param batch_size: Batch size for scrolling
        :return: List of similarity dicts
        """
        logger.info(f"Computing section similarities for document {doc_id}")
        
        # Scroll through all chunks for this document
        all_points = []
        offset = None
        
        while True:
            scroll_result = self.client.scroll(
                collection_name=self.text_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            
            points, next_offset = scroll_result
            all_points.extend(points)
            
            if next_offset is None:
                break
            offset = next_offset
        
        if not all_points:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        logger.info(f"Found {len(all_points)} chunks for document {doc_id}")
        
        # Group chunks by section_id and compute average embeddings
        section_vectors: Dict[str, List[List[float]]] = {}
        section_metadata: Dict[str, Dict] = {}
        
        for point in all_points:
            section_id = point.payload.get("section_id")
            if not section_id:
                continue
            
            if section_id not in section_vectors:
                section_vectors[section_id] = []
                section_metadata[section_id] = {
                    "doc_id": point.payload.get("doc_id"),
                    "page_start": point.payload.get("page_start"),
                    "page_end": point.payload.get("page_end"),
                }
            
            section_vectors[section_id].append(point.vector)
        
        # Compute average embedding per section
        section_embeddings: Dict[str, List[float]] = {}
        
        for section_id, vectors in section_vectors.items():
            if vectors:
                avg_vector = [
                    sum(v[i] for v in vectors) / len(vectors)
                    for i in range(len(vectors[0]))
                ]
                section_embeddings[section_id] = avg_vector
        
        logger.info(
            f"Computed embeddings for {len(section_embeddings)} sections "
            f"in document {doc_id}"
        )
        
        # Find top-K similar sections for each section
        similarities = []
        section_ids = list(section_embeddings.keys())
        
        for i, source_section_id in enumerate(section_ids):
            source_vector = section_embeddings[source_section_id]
            
            search_results = self.client.search(
                collection_name=self.text_collection,
                query_vector=source_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
                    must_not=[FieldCondition(key="section_id", match=MatchValue(value=source_section_id))],
                ),
                limit=top_k * 3,
                with_payload=True,
                with_vectors=False,
            )
            
            seen_sections = set()
            for result in search_results:
                target_section_id = result.payload.get("section_id")
                
                if (
                    target_section_id 
                    and target_section_id not in seen_sections
                    and result.score >= score_threshold
                ):
                    seen_sections.add(target_section_id)
                    
                    similarities.append({
                        "from_id": source_section_id,
                        "to_id": target_section_id,
                        "score": float(result.score),
                    })
                    
                    if len(seen_sections) >= top_k:
                        break
            
            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1}/{len(section_ids)} sections")
        
        logger.info(
            f"Found {len(similarities)} similarity relationships "
            f"for document {doc_id}"
        )
        
        return similarities

    # -------------------------------------------------------------------------
    # Maintenance Operations
    # -------------------------------------------------------------------------

    def delete_document_vectors(self, doc_id: str) -> None:
        """
        Delete all vectors associated with a document from all collections.
        
        :param doc_id: Document ID
        """
        selector = FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )

        deleted_counts = {}
        
        # Delete from text chunks
        try:
            result = self.client.delete(
                collection_name=self.text_collection,
                points_selector=selector,
                wait=True,
            )
            deleted_counts["text_chunks"] = result.status if hasattr(result, 'status') else "deleted"
            logger.info(f"Deleted text vectors for document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete text vectors: {e}")
            deleted_counts["text_chunks"] = f"error: {e}"

        # Delete from schemas
        try:
            result = self.client.delete(
                collection_name=self.schemas_collection,
                points_selector=selector,
                wait=True,
            )
            deleted_counts["schemas"] = result.status if hasattr(result, 'status') else "deleted"
            logger.info(f"Deleted schema vectors for document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete schema vectors: {e}")
            deleted_counts["schemas"] = f"error: {e}"
        
        # Delete from table chunks
        try:
            result = self.client.delete(
                collection_name=self.tables_text_collection,
                points_selector=selector,
                wait=True,
            )
            deleted_counts["tables"] = result.status if hasattr(result, 'status') else "deleted"
            logger.info(f"Deleted table vectors for document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete table vectors: {e}")
            deleted_counts["tables"] = f"error: {e}"
        
        logger.info(f"✅ Deleted all vectors for document {doc_id}: {deleted_counts}")

    def delete_section_vectors(self, section_id: str) -> None:
        """
        Delete all vectors (chunks) for a specific section.
        
        :param section_id: Section ID
        """
        selector = FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="section_id", match=MatchValue(value=section_id))]
            )
        )
        self.client.delete(
            collection_name=self.text_collection,
            points_selector=selector,
        )
        logger.debug(f"Deleted vectors for section {section_id}")

    def delete_schema_vector(self, schema_id: str) -> None:
        """
        NEW: Delete vector for a specific schema.
        
        :param schema_id: Schema ID
        """
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, schema_id))
        
        self.client.delete(
            collection_name=self.schemas_collection,
            points_selector=[point_id],
        )
        
        logger.debug(f"Deleted vector for schema {schema_id}")

    def delete_figure_vector(self, figure_id: str) -> None:
        """
        DEPRECATED: Use delete_schema_vector() instead.
        """
        logger.warning("delete_figure_vector() is deprecated. Use delete_schema_vector().")
        self.delete_schema_vector(figure_id)

    def delete_table_vectors(self, table_id: str) -> None:
        """
        Delete all chunk vectors for a specific table.
        
        :param table_id: Table ID
        """
        selector = FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="table_id", match=MatchValue(value=table_id))]
            )
        )
        self.client.delete(
            collection_name=self.tables_text_collection,
            points_selector=selector,
        )
        logger.debug(f"Deleted all chunk vectors for table {table_id}")

    async def count_text_chunks(self, doc_id: str) -> int:
        """
        Count text chunks for a specific document.
        
        :param doc_id: Document ID
        :return: Number of text chunks
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Count points with matching doc_id
            count_result = await asyncio.to_thread(
                self.client.count,
                collection_name=self.text_collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                ),
                exact=True,
            )
            
            return count_result.count
        
        except Exception as e:
            logger.error(f"Error counting text chunks for {doc_id}: {e}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get basic statistics about collections.
        Returns point counts for each collection.
        
        :return: Dictionary with collection statistics
        """
        info: Dict[str, Any] = {}
        
        collections = {
            "text_chunks": self.text_collection,
            "schemas": self.schemas_collection,
            "tables": self.tables_text_collection,
        }
        
        for label, collection_name in collections.items():
            try:
                # Get exact count
                count_result = self.client.count(
                    collection_name=collection_name,
                    exact=True,
                )
                
                info[label] = {
                    "collection_name": collection_name,
                    "points_count": count_result.count,
                    "status": "active",
                }
            
            except Exception as e:
                logger.error(f"Error getting info for {collection_name}: {e}")
                info[label] = {
                    "collection_name": collection_name,
                    "points_count": 0,
                    "status": "error",
                    "error": str(e),
                }
        
        # Add summary
        info["summary"] = {
            "total_points": sum(
                coll.get("points_count", 0) 
                for coll in info.values() 
                if isinstance(coll, dict) and "points_count" in coll
            ),
            "active_collections": sum(
                1 for coll in info.values()
                if isinstance(coll, dict) and coll.get("status") == "active"
            ),
        }
        
        return info
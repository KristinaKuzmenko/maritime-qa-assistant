"""
Neo4j database client for graph operations.
Manages document structure, schemas, tables, and relationships.
NO EMBEDDINGS - vectors stored only in Qdrant.

GRAPH SCHEMA:
- Document → Chapter → Section (document hierarchy)
- Section → Schema (diagrams/drawings)
- Section → Table (specifications/data)
- Schema ↔ Table (RELATED_TO: co-located on same page)
- Section ↔ Section (SIMILAR_TO: semantic similarity, REFERENCES: cross-refs)
- Entity nodes and relationships 

"""

from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any, Optional
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async Neo4j client for graph database operations"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[AsyncDriver] = None
        
    async def connect(self):
        """Initialize connection to Neo4j"""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        await self.verify_connection()
        await self.create_constraints_and_indexes()
        
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            
    async def verify_connection(self):
        """Verify Neo4j connection"""
        async with self.driver.session(database=self.database) as session:
            result = await session.run("RETURN 1 as n")
            record = await result.single()
            if record["n"] != 1:
                raise ConnectionError("Failed to connect to Neo4j")
            logger.info("Successfully connected to Neo4j")
            
    async def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS ON (d:Document) ASSERT d.id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS ON (c:Chapter) ASSERT c.id IS UNIQUE",
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS ON (s:Section) ASSERT s.id IS UNIQUE",
            "CREATE CONSTRAINT schema_id_unique IF NOT EXISTS ON (sc:Schema) ASSERT sc.id IS UNIQUE",
            "CREATE CONSTRAINT table_id_unique IF NOT EXISTS ON (t:Table) ASSERT t.id IS UNIQUE",
            "CREATE CONSTRAINT table_chunk_id_unique IF NOT EXISTS ON (tc:TableChunk) ASSERT tc.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS ON (e:Entity) ASSERT e.id IS UNIQUE",
            "CREATE CONSTRAINT term_id_unique IF NOT EXISTS ON (t:Term) ASSERT t.id IS UNIQUE",
        ]
        
        indexes = [
            "CREATE INDEX doc_type_status IF NOT EXISTS FOR (d:Document) ON (d.doc_type, d.status)",
            "CREATE INDEX section_doc_page IF NOT EXISTS FOR (s:Section) ON (s.doc_id, s.page_number)",
            "CREATE INDEX schema_doc_page IF NOT EXISTS FOR (sc:Schema) ON (sc.doc_id, sc.page_number)",
            "CREATE INDEX chapter_doc_number IF NOT EXISTS FOR (c:Chapter) ON (c.doc_id, c.number)",
            "CREATE INDEX entity_code_type IF NOT EXISTS FOR (e:Entity) ON (e.code, e.entity_type)",
            "CREATE INDEX table_doc_page IF NOT EXISTS FOR (t:Table) ON (t.doc_id, t.page_number)",
            "CREATE INDEX table_chunk_parent IF NOT EXISTS FOR (tc:TableChunk) ON (tc.parent_table_id, tc.chunk_index)",
            "CREATE INDEX similar_to_score IF NOT EXISTS FOR ()-[r:SIMILAR_TO]-() ON (r.score)",
        ]
        
        async with self.driver.session(database=self.database) as session:
            # Constraints
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint might already exist: {e}")
            
            # Indexes
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index might already exist: {e}")
                    
            # Fulltext indexes
            fulltext_indexes = [
                ("documentSearch", ["Document"], ["title", "tags"]),
                ("sectionSearch", ["Section", "Chapter"], ["title", "content"]),
                ("schemaSearch", ["Schema"], ["title", "caption", "text_context"]),
                ("tableSearch", ["Table"], ["title", "caption", "text_preview"]),
                ("tableChunkSearch", ["TableChunk"], ["text_preview"]),
                ("entitySearch", ["Entity"], ["name", "code", "entity_type", "system", "tags"]),
            ]
            
            for index_name, labels, properties in fulltext_indexes:
                try:
                    query = f"""
                    CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                    FOR (n:{'|'.join(labels)})
                    ON EACH [{', '.join([f"n.{prop}" for prop in properties])}]
                    """
                    await session.run(query)
                except Exception as e:
                    logger.debug(f"Fulltext index might already exist: {e}")
                    
            logger.info("Successfully created constraints and indexes")
    
    async def run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return result data"""
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records
            
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        :param query: Cypher query string
        :param parameters: Query parameters
        :return: List of result records as dictionaries
        """
        if parameters is None:
            parameters = {}
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters)
                records = await result.data()
                
                logger.debug(f"Query executed: {query[:100]}...")
                logger.debug(f"Parameters: {parameters}")
                logger.debug(f"Result count: {len(records)}")
                
                return records
        
        except Exception as e:
            logger.error(f"❌ Neo4j query error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            
            # Check if Neo4j is accessible
            try:
                await self.verify_connection()
                logger.error("Neo4j is connected but query failed - likely a Cypher syntax error")
            except Exception:
                logger.error("❌ Neo4j connection lost!")
            
            raise
    
    # -------------------------------------------------------------------------
    # Document operations
    # -------------------------------------------------------------------------
    
    async def create_document(self, document_data: Dict[str, Any]) -> str:
        """Create a new document node"""
        doc_id = document_data.get("id", str(uuid.uuid4()))
        
        query = """
        CREATE (d:Document {
            id: $id,
            title: $title,
            doc_type: $doc_type,
            version: $version,
            language: $language,
            file_path: $file_path,
            file_hash: $file_hash,
            total_pages: $total_pages,
            upload_date: datetime(),
            status: 'processing',
            metadata: $metadata,
            tags: $tags,
            owner: $owner
        })
        RETURN d.id as doc_id
        """
        
        result = await self.run_query(
            query,
            {
                "id": doc_id,
                "title": document_data.get("title", "Untitled"),
                "doc_type": document_data.get("doc_type", "manual"),
                "version": document_data.get("version", "1.0"),
                "language": document_data.get("language", "en"),
                "file_path": document_data.get("file_path"),
                "file_hash": document_data.get("file_hash"),
                "total_pages": document_data.get("total_pages", 0),
                "metadata": json.dumps(document_data.get("metadata", {})),
                "tags": document_data.get("tags", []),
                "owner": document_data.get("owner", "global"),
            },
        )
        
        return result[0]["doc_id"]
    
    async def create_chapter(self, chapter_data: Dict[str, Any], doc_id: str) -> str:
        """
        Create a Chapter node and link it to a Document.
        
        :param chapter_data: Chapter properties (must include 'id')
        :param doc_id: ID of parent document
        :return: chapter_id: Created chapter node ID
        """
        chapter_id = chapter_data.get("id")
        if not chapter_id:
            chapter_id = str(uuid.uuid4())
            logger.warning(f"No chapter ID provided, generated: {chapter_id}")
        
        query = """
        MATCH (doc:Document {id: $doc_id})
        CREATE (chapter:Chapter {
            id: $chapter_id,
            number: $number,
            title: $title,
            start_page: $start_page,
            end_page: $end_page,
            doc_id: $doc_id,
            created_at: datetime()
        })
        CREATE (doc)-[:HAS_CHAPTER]->(chapter)
        RETURN chapter.id AS chapter_id
        """
        
        params = {
            "doc_id": doc_id,
            "chapter_id": chapter_id,
            "number": chapter_data.get("number", ""),
            "title": chapter_data.get("title", "Untitled Chapter"),
            "start_page": chapter_data.get("start_page"),
            "end_page": chapter_data.get("end_page"),
        }
        
        try:
            result = await self.run_query(query, params)
            
            if not result:
                logger.error(f"❌ create_chapter returned empty result")
                logger.error(f"Params: {params}")
                raise ValueError(f"Failed to create chapter")
            
            created_id = result[0]["chapter_id"]
            
            if created_id != chapter_id:
                logger.error(f"❌ ID mismatch! Expected {chapter_id}, got {created_id}")
                raise ValueError(f"Chapter ID mismatch")
            
            logger.info(f"✅ Created chapter {created_id} in document {doc_id}")
            
            return created_id
            
        except Exception as e:
            logger.error(f"❌ Error creating chapter: {e}")
            logger.error(f"Doc ID: {doc_id}")
            logger.error(f"Chapter data: {chapter_data}")
            raise
    
    async def create_section(self, section_data: Dict[str, Any], chapter_id: str) -> str:
        """
        Create a Section node and link it to a Chapter.
        
        :param section_data: Section properties (must include 'id' if you want specific ID)
        :param chapter_id: ID of parent chapter
        :return: section_id: Created section node ID
        """
        logger.info(f"create_section called with chapter_id={chapter_id}")
        logger.debug(f"Section data keys: {list(section_data.keys())}")
        
        # Verify chapter exists
        check_query = """
        MATCH (chapter:Chapter {id: $chapter_id})
        RETURN chapter.id AS id, chapter.title AS title, chapter.number AS number
        """
        
        check_result = await self.run_query(check_query, {"chapter_id": chapter_id})
        
        if not check_result:
            logger.error(f"❌ Chapter {chapter_id} does not exist!")
            
            # Debug: Show all chapters
            all_chapters_query = """
            MATCH (c:Chapter)
            RETURN c.id, c.number, c.title, c.doc_id, labels(c) as labels
            ORDER BY c.created_at DESC
            LIMIT 10
            """
            all_chapters = await self.run_query(all_chapters_query)
            logger.error(f"All chapters in DB ({len(all_chapters)}):")
            for ch in all_chapters:
                logger.error(f"  - {ch['c.id']}: {ch['c.title']} (number: {ch['c.number']})")
            
            raise ValueError(f"Chapter {chapter_id} not found in database")
        
        logger.info(f"✅ Found chapter: {check_result[0]['title']} (number: {check_result[0]['number']})")
        
        # Use provided section ID or generate one
        section_id = section_data.get("id")
        if not section_id:
            section_id = str(uuid.uuid4())
            logger.warning(f"No section ID provided, generated: {section_id}")
        
        # Create section with optional merged section fields
        base_props = """
            id: $section_id,
            section_number: $section_number,
            title: $title,
            content: $content,
            section_type: $section_type,
            importance_score: $importance_score,
            page_start: $page_start,
            page_end: $page_end,
            doc_id: chapter.doc_id,
            created_at: datetime()
        """
        
        # Add merged section fields if present
        if section_data.get("is_merged"):
            base_props += """,
            is_merged: $is_merged,
            original_count: $original_count,
            merged_sections: $merged_sections
        """
        
        query = f"""
        MATCH (chapter:Chapter {{id: $chapter_id}})
        CREATE (section:Section {{
            {base_props}
        }})
        CREATE (chapter)-[:HAS_SECTION]->(section)
        RETURN section.id AS section_id
        """
        
        params = {
            "chapter_id": chapter_id,
            "section_id": section_id,
            "section_number": section_data.get("section_number", ""),
            "title": section_data.get("title", "Untitled Section"),
            "content": section_data.get("content", ""),
            "section_type": section_data.get("section_type", ""),
            "importance_score": section_data.get("importance_score", 0.0),
            "page_start": section_data.get("page_start"),
            "page_end": section_data.get("page_end"),
        }
        
        # Add merged section params if present
        if section_data.get("is_merged"):
            params["is_merged"] = section_data["is_merged"]
            params["original_count"] = section_data.get("original_count", 0)
            params["merged_sections"] = section_data.get("merged_sections", [])
        
        logger.info(f"Creating section {section_id}")
        result = await self.run_query(query, params)
        
        if not result:
            raise ValueError(f"Failed to create section under chapter {chapter_id}")
        
        created_id = result[0]["section_id"]
        
        if created_id != section_id:
            logger.error(f"❌ ID mismatch! Expected {section_id}, got {created_id}")
            raise ValueError(f"Section ID mismatch")
        
        logger.info(f"✅ Created section {created_id}: {params['title']}")
        
        return created_id

    # -------------------------------------------------------------------------
    # Schema (Diagram/Figure) operations 
    # -------------------------------------------------------------------------
    
    async def create_schema(self, schema_data: Dict[str, Any], section_id: Optional[str] = None) -> str:
        """
        Create a schema (diagram/figure) node with guaranteed link to Document.
        UPDATED: Simplified for layout-aware extraction (no internal parsing).
        
        :param schema_data: Schema properties
        :param section_id: Optional parent section ID
        :return: schema_id
        """
        schema_id = schema_data.get("id", str(uuid.uuid4()))
        doc_id = schema_data.get("doc_id")
        
        if not doc_id:
            raise ValueError("doc_id is required for schema creation")
        
        if section_id:
            # Link through Section only (Document connection via Chapter→Section path)
            query = """
            MATCH (s:Section {id: $section_id})
            CREATE (sc:Schema {
                id: $schema_id,
                doc_id: $doc_id,
                page_number: $page_number,
                title: $title,
                caption: $caption,
                file_path: $file_path,
                thumbnail_path: $thumbnail_path,
                bbox: $bbox,
                text_context: $text_context,
                llm_summary: $llm_summary,
                confidence: $confidence,
                tags: $tags
            })
            CREATE (s)-[:CONTAINS_SCHEMA]->(sc)
            RETURN sc.id as schema_id
            """
        else:
            # Link directly to Document (fallback)
            query = """
            MATCH (d:Document {id: $doc_id})
            CREATE (sc:Schema {
                id: $schema_id,
                doc_id: $doc_id,
                page_number: $page_number,
                title: $title,
                caption: $caption,
                file_path: $file_path,
                thumbnail_path: $thumbnail_path,
                bbox: $bbox,
                text_context: $text_context,
                llm_summary: $llm_summary,
                confidence: $confidence,
                tags: $tags
            })
            CREATE (d)-[:HAS_SCHEMA]->(sc)
            RETURN sc.id as schema_id
            """
            logger.warning(f"Schema {schema_id} created without section link, linking directly to document")
    
        result = await self.run_query(
            query,
            {
                "schema_id": schema_id,
                "section_id": section_id,
                "doc_id": doc_id,
                "page_number": schema_data.get("page_number"),
                "title": schema_data.get("title", ""),
                "caption": schema_data.get("caption", ""),
                "file_path": schema_data.get("file_path"),
                "thumbnail_path": schema_data.get("thumbnail_path"),
                "bbox": json.dumps(schema_data.get("bbox", {})),
                "text_context": schema_data.get("text_context", ""),
                "llm_summary": schema_data.get("llm_summary", ""),
                "confidence": schema_data.get("confidence", 1.0),
                "tags": schema_data.get("tags", []),
            },
        )
        
        return result[0]["schema_id"]
    
    async def link_schema_to_table(
        self,
        schema_id: str,
        table_id: str,
        relationship_type: str = "HAS_TABLE",
    ) -> None:
        """
        NEW: Create relationship between schema and table (e.g., diagram and BOM).
        
        :param schema_id: Schema node ID
        :param table_id: Table node ID
        :param relationship_type: Relationship type (default: HAS_TABLE)
        """
        query = f"""
        MATCH (sc:Schema {{id: $schema_id}})
        MATCH (t:Table {{id: $table_id}})
        MERGE (sc)-[r:{relationship_type}]->(t)
        RETURN r
        """
        
        await self.run_query(query, {
            "schema_id": schema_id,
            "table_id": table_id,
        })
        
        logger.debug(f"Linked schema {schema_id} to table {table_id}")
    
    async def link_section_to_schema(
        self,
        section_id: str,
        schema_id: str,
        reference: Optional[str] = None
    ):
        """Link a section to a schema it references"""
        query = """
        MATCH (s:Section {id: $section_id})
        MATCH (sc:Schema {id: $schema_id})
        MERGE (s)-[r:REFERENCES]->(sc)
        SET r.reference = $reference
        """
        await self.run_query(query, {
            "section_id": section_id,
            "schema_id": schema_id,
            "reference": reference
        })
    
    async def get_schema_with_context(
        self,
        schema_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        NEW: Get schema with related entities (sections, tables).
        
        :param schema_id: Schema ID
        :return: Schema dict with relations
        """
        query = """
        MATCH (sc:Schema {id: $schema_id})
        OPTIONAL MATCH (s:Section)-[:REFERENCES]->(sc)
        OPTIONAL MATCH (sc)-[:HAS_TABLE]->(t:Table)
        RETURN sc,
               collect(DISTINCT s.id) as section_ids,
               collect(DISTINCT t.id) as table_ids
        """
        
        result = await self.run_query(query, {"schema_id": schema_id})
        
        if not result:
            return None
        
        record = result[0]
        schema = dict(record["sc"])
        schema["section_ids"] = record["section_ids"]
        schema["table_ids"] = record["table_ids"]
        
        return schema
    
    async def get_schemas_by_doc(
        self,
        doc_id: str,
        page_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        NEW: Get all schemas for a document, optionally filtered by page.
        
        :param doc_id: Document ID
        :param page_number: Optional page number filter
        :return: List of schema dicts
        """
        if page_number is not None:
            query = """
            MATCH (sc:Schema {doc_id: $doc_id, page_number: $page_number})
            RETURN sc
            ORDER BY sc.page_number
            """
            params = {"doc_id": doc_id, "page_number": page_number}
        else:
            query = """
            MATCH (sc:Schema {doc_id: $doc_id})
            RETURN sc
            ORDER BY sc.page_number
            """
            params = {"doc_id": doc_id}
        
        result = await self.run_query(query, params)
        return [dict(r["sc"]) for r in result]
    
    async def get_schema_details(
        self,
        schema_ids: List[str],
        include_entities: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get schema details with document and section context, including embedded tables."""
        query = """
        UNWIND $schema_ids AS sid
        MATCH (sc:Schema {id: sid})
        OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:CONTAINS_SCHEMA]->(sc)
        OPTIONAL MATCH (sc)-[:DEPICTS]->(e:Entity)
        WHERE $include_entities = true
        OPTIONAL MATCH (sc)-[:HAS_LEGEND]->(legend_table:Table)
        RETURN sc {
                   .*,
                   doc_title: d.title,
                   chapter_title: c.title,
                   section_id: s.id,
                   has_embedded_table: legend_table IS NOT NULL
               } as schema,
               collect(DISTINCT e {.*}) as depicted_entities,
               collect(DISTINCT legend_table {.id, .title, .rows, .cols}) as embedded_tables
        """
        return await self.run_query(query, {
            "schema_ids": schema_ids,
            "include_entities": include_entities
        })
    
    # -------------------------------------------------------------------------
    # Table operations (parent Table + TableChunk nodes)
    # -------------------------------------------------------------------------
    
    async def create_table(self, table_data: Dict[str, Any], section_id: Optional[str] = None) -> str:
        """Create a parent Table node with guaranteed link to Document"""
        table_id = table_data.get("id", str(uuid.uuid4()))
        doc_id = table_data.get("doc_id")
        
        if not doc_id:
            raise ValueError("doc_id is required for table creation")
        
        if section_id:
            # Link through Section only (Document connection via Chapter→Section path)
            query = """
            MATCH (s:Section {id: $section_id})
            CREATE (t:Table {
                id: $table_id,
                doc_id: $doc_id,
                page_number: $page_number,
                title: $title,
                caption: $caption,
                rows: $rows,
                cols: $cols,
                file_path: $file_path,
                thumbnail_path: $thumbnail_path,
                csv_path: $csv_path,
                bbox: $bbox,
                text_preview: $text_preview,
                normalized_text: $normalized_text,
                tags: $tags
            })
            CREATE (s)-[:CONTAINS_TABLE]->(t)
            RETURN t.id as table_id
            """
        else:
            # Link directly to Document (fallback)
            query = """
            MATCH (d:Document {id: $doc_id})
            CREATE (t:Table {
                id: $table_id,
                doc_id: $doc_id,
                page_number: $page_number,
                title: $title,
                caption: $caption,
                rows: $rows,
                cols: $cols,
                file_path: $file_path,
                thumbnail_path: $thumbnail_path,
                csv_path: $csv_path,
                bbox: $bbox,
                text_preview: $text_preview,
                normalized_text: $normalized_text,
                tags: $tags
            })
            CREATE (d)-[:HAS_TABLE]->(t)
            RETURN t.id as table_id
            """
            logger.warning(f"Table {table_id} created without section link, linking directly to document")

        result = await self.run_query(
            query,
            {
                "table_id": table_id,
                "section_id": section_id,
                "doc_id": doc_id,
                "page_number": table_data.get("page_number"),
                "title": table_data.get("title", ""),
                "caption": table_data.get("caption", ""),
                "rows": table_data.get("rows", 0),
                "cols": table_data.get("cols", 0),
                "file_path": table_data.get("file_path"),
                "thumbnail_path": table_data.get("thumbnail_path"),
                "csv_path": table_data.get("csv_path", ""),
                "bbox": json.dumps(table_data.get("bbox", {})),
                "text_preview": table_data.get("text_preview", ""),
                "normalized_text": table_data.get("normalized_text", ""),
                "tags": table_data.get("tags", []),
            },
        )

        return result[0]["table_id"]

    async def create_table_chunk(
        self,
        parent_table_id: str,
        chunk_index: int,
        chunk_text: str,
        total_chunks: int,
        doc_id: str,
        page_number: int,
    ) -> str:
        """
        Create a TableChunk node linked to parent Table.
        Each chunk represents a searchable portion of the table content.
        
        :param parent_table_id: ID of parent Table node
        :param chunk_index: Index of this chunk (0-based)
        :param chunk_text: Text content of this chunk
        :param total_chunks: Total number of chunks for this table
        :param doc_id: Document ID
        :param page_number: Page number
        :return: TableChunk ID
        """
        chunk_id = f"{parent_table_id}_chunk_{chunk_index}"
        
        query = """
        MATCH (t:Table {id: $parent_table_id})
        CREATE (tc:TableChunk {
            id: $chunk_id,
            parent_table_id: $parent_table_id,
            chunk_index: $chunk_index,
            text_preview: $text_preview,
            text_length: $text_length,
            total_chunks: $total_chunks,
            doc_id: $doc_id,
            page_number: $page_number
        })
        CREATE (tc)-[:PART_OF]->(t)
        RETURN tc.id as chunk_id
        """
        
        text_preview = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
        
        result = await self.run_query(
            query,
            {
                "chunk_id": chunk_id,
                "parent_table_id": parent_table_id,
                "chunk_index": chunk_index,
                "text_preview": text_preview,
                "text_length": len(chunk_text),
                "total_chunks": total_chunks,
                "doc_id": doc_id,
                "page_number": page_number,
            },
        )
        
        if not result or len(result) == 0:
            logger.error(
                f"❌ Failed to create TableChunk {chunk_id}! "
                f"Parent table {parent_table_id} not found in Neo4j"
            )
            raise ValueError(f"Parent Table {parent_table_id} not found in Neo4j")
        
        logger.debug(
            f"✅ Created TableChunk {chunk_id} "
            f"(index {chunk_index}/{total_chunks-1}, {len(chunk_text)} chars)"
        )
        
        return result[0]["chunk_id"]

    async def get_table_chunks(self, table_id: str) -> List[Dict[str, Any]]:
        """
        NEW: Get all chunks for a table, ordered by chunk_index.
        
        :param table_id: Table ID
        :return: List of chunk dicts
        """
        query = """
        MATCH (t:Table {id: $table_id})<-[:PART_OF]-(tc:TableChunk)
        RETURN tc {.*} as chunk
        ORDER BY tc.chunk_index
        """
        result = await self.run_query(query, {"table_id": table_id})
        return [r["chunk"] for r in result]

    async def link_section_to_table(self, section_id: str, table_id: str):
        """Link a Section to an existing Table via CONTAINS_TABLE."""
        query = """
        MATCH (s:Section {id: $section_id})
        MATCH (t:Table  {id: $table_id})
        MERGE (s)-[:CONTAINS_TABLE]->(t)
        """
        await self.run_query(query, {"section_id": section_id, "table_id": table_id})

    async def link_table_to_entity(self, table_id: str, entity_id: str):
        """Link table to a domain entity it lists/mentions."""
        query = """
        MATCH (t:Table {id: $table_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (t)-[:MENTIONS]->(e)
        """
        await self.run_query(query, {"table_id": table_id, "entity_id": entity_id})
    
    async def link_schema_to_table(self, schema_id: str, table_id: str):
        """
        Link a Schema to an embedded Table (legend, specs, etc.).
        Used for hybrid schema+table cases where table is part of the schema.
        """
        query = """
        MATCH (sc:Schema {id: $schema_id})
        MATCH (t:Table {id: $table_id})
        MERGE (sc)-[:HAS_LEGEND]->(t)
        """
        await self.run_query(query, {"schema_id": schema_id, "table_id": table_id})
        logger.info(f"Linked Schema {schema_id} to embedded Table {table_id}")

    async def get_table_details(
        self,
        table_ids: List[str],
        include_entities: bool = True,
        include_chunks: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get table details with document and section context.
        Optionally include TableChunk information.
        """
        query = """
        UNWIND $table_ids AS tid
        MATCH (t:Table {id: tid})
        OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:CONTAINS_TABLE]->(t)
        OPTIONAL MATCH (t)-[:MENTIONS]->(e:Entity)
        WHERE $include_entities = true
        OPTIONAL MATCH (t)<-[:PART_OF]-(tc:TableChunk)
        WHERE $include_chunks = true
        WITH t, d, c, s, 
             collect(DISTINCT e {.*}) as mentioned_entities,
             tc
        ORDER BY tc.chunk_index
        WITH t, d, c, s, mentioned_entities,
             collect(tc {.*}) as chunks
        RETURN t {
                   .*,
                   doc_title: d.title,
                   chapter_title: c.title,
                   section_id: s.id
               } as table,
               mentioned_entities,
               chunks
        """
        return await self.run_query(
            query, 
            {
                "table_ids": table_ids, 
                "include_entities": include_entities,
                "include_chunks": include_chunks,
            }
        )
            
    # -------------------------------------------------------------------------
    # Term operations
    # -------------------------------------------------------------------------

    async def create_term(self, term_data: Dict[str, Any]) -> str:
        """Create or update a technical term"""
        term_id = term_data.get("id", str(uuid.uuid4()))
        
        # Validate required fields
        term = term_data.get("term") or term_data.get("code") or term_data.get("name")
        if not term:
            logger.warning("Skipping term creation - no valid term identifier provided")
            return term_id
        
        query = """
        MERGE (t:Term {term: $term})
        ON CREATE SET
            t.id = $term_id,
            t.abbreviation = $abbreviation,
            t.definition = $definition,
            t.context = $context,
            t.synonyms = $synonyms,
            t.translations = $translations
        ON MATCH SET
            t.definition = COALESCE($definition, t.definition),
            t.context = COALESCE($context, t.context)
        RETURN t.id as term_id
        """
        
        result = await self.run_query(
            query,
            {
                "term_id": term_id,
                "term": term,
                "abbreviation": term_data.get("abbreviation", ""),
                "definition": term_data.get("definition", ""),
                "context": term_data.get("context", ""),
                "synonyms": term_data.get("synonyms", []),
                "translations": json.dumps(term_data.get("translations", {})),
            },
        )
        
        return result[0]["term_id"]
    
    async def link_section_to_term(self, section_id: str, term_id: str):
        """Link section to a term it uses"""
        query = """
        MATCH (s:Section {id: $section_id})
        MATCH (t:Term {id: $term_id})
        MERGE (s)-[:USES_TERM]->(t)
        """
        await self.run_query(query, {"section_id": section_id, "term_id": term_id})
    
    # -------------------------------------------------------------------------
    # Entity operations (domain model)
    # -------------------------------------------------------------------------
    
    async def create_entity(self, entity_data: Dict[str, Any]) -> str:
        """Create or update a physical/logical entity (pump, valve, tank, system, etc.)."""
        entity_id = entity_data.get("id", str(uuid.uuid4()))
        
        query = """
        MERGE (e:Entity {code: $code})
        ON CREATE SET
            e.id = $id,
            e.name = $name,
            e.entity_type = $entity_type,
            e.system = $system,
            e.tags = $tags,
            e.metadata = $metadata
        ON MATCH SET
            e.name = COALESCE($name, e.name),
            e.entity_type = COALESCE($entity_type, e.entity_type),
            e.system = COALESCE($system, e.system)
        RETURN e.id AS entity_id
        """
        
        result = await self.run_query(
            query,
            {
                "id": entity_id,
                "code": entity_data.get("code"),
                "name": entity_data.get("name"),
                "entity_type": entity_data.get("entity_type"),
                "system": entity_data.get("system"),
                "tags": entity_data.get("tags", []),
                "metadata": json.dumps(entity_data.get("metadata", {})),
            },
        )
        
        return result[0]["entity_id"]
    
    async def link_term_to_entity(self, term_id: str, entity_id: str):
        """Link technical term to the domain entity it refers to."""
        query = """
        MATCH (t:Term {id: $term_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (t)-[:REFERS_TO]->(e)
        """
        await self.run_query(query, {"term_id": term_id, "entity_id": entity_id})
    
    async def link_section_to_entity(self, section_id: str, entity_id: str):
        """Link section that describes or mentions an entity."""
        query = """
        MATCH (s:Section {id: $section_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (s)-[:DESCRIBES]->(e)
        """
        await self.run_query(query, {"section_id": section_id, "entity_id": entity_id})
    
    async def link_schema_to_entity(self, schema_id: str, entity_id: str):
        """Link schema that visually depicts an entity."""
        query = """
        MATCH (sc:Schema {id: $schema_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (sc)-[:DEPICTS]->(e)
        """
        await self.run_query(query, {"schema_id": schema_id, "entity_id": entity_id})
    
    async def create_entity_relation(
        self,
        from_entity_id: str,
        to_entity_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Create typed relationship between two entities (PART_OF, CONNECTED_TO, etc.).
        rel_type must be controlled by application code.
        """
        if properties is None:
            properties = {}
        
        allowed_types = ["PART_OF", "CONNECTED_TO", "LOCATED_IN", "FEEDS_INTO", "CONTROLS"]
        if rel_type not in allowed_types:
            raise ValueError(f"Invalid relationship type: {rel_type}")
        
        query = f"""
        MATCH (a:Entity {{id: $from_id}})
        MATCH (b:Entity {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """
        
        await self.run_query(
            query,
            {
                "from_id": from_entity_id,
                "to_id": to_entity_id,
                "props": properties,
            },
        )
    
    async def get_entity_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Get entity by code."""
        query = """
        MATCH (e:Entity {code: $code})
        RETURN e {.*} as entity
        """
        result = await self.run_query(query, {"code": code})
        return result[0]["entity"] if result else None
    
    async def find_entities_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """Find entities by their names (case-insensitive partial match)."""
        query = """
        UNWIND $names AS name
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower(name) OR toLower(e.system) CONTAINS toLower(name)
        RETURN DISTINCT e.id AS id, e.name AS name, e.code AS code, 
               e.entity_type AS type, e.system AS system
        """
        return await self.run_query(query, {"names": names})
    
    async def get_entity_hierarchy(self, entity_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get entity hierarchy (systems, subsystems, components)."""
        query = """
        MATCH path = (e:Entity {id: $entity_id})-[:PART_OF*0..]->(parent:Entity)
        WHERE length(path) <= $max_depth
        RETURN e {.*} as entity,
               [node IN nodes(path) | node {.*}] as hierarchy,
               length(path) as depth
        ORDER BY depth DESC
        LIMIT 1
        """
        result = await self.run_query(query, {"entity_id": entity_id, "max_depth": max_depth})
        return result[0] if result else {}
    
    async def get_entity_context(
        self,
        entity_ids: List[str],
        include_hierarchy: bool = True,
        include_connections: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get entity context including hierarchy and connections."""
        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:Entity {id: entity_id})
        OPTIONAL MATCH path = (e)-[:PART_OF*1..3]->(parent:Entity)
        WHERE $include_hierarchy = true
        OPTIONAL MATCH (e)-[conn:CONNECTED_TO]-(connected:Entity)
        WHERE $include_connections = true
        RETURN e {.*} as entity,
               [node IN nodes(path) | node {.*}] as hierarchy,
               collect(DISTINCT {
                   entity: connected {.*},
                   medium: conn.medium,
                   direction: conn.direction
               }) as connections
        """
        return await self.run_query(query, {
            "entity_ids": entity_ids,
            "include_hierarchy": include_hierarchy,
            "include_connections": include_connections
        })
    
    async def search_sections_fulltext(
        self,
        query: str,
        limit: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search in sections (fallback)."""
        cypher_query = """
        CALL db.index.fulltext.queryNodes('sectionSearch', $query) 
        YIELD node, score
        WHERE node:Section AND ($doc_id IS NULL OR node.doc_id = $doc_id)
        MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(node)
        RETURN node {
                   .*,
                   doc_title: d.title,
                   chapter_title: c.title,
                   search_score: score
               } as section
        ORDER BY score DESC
        LIMIT $limit
        """
        return await self.run_query(cypher_query, {"query": query, "doc_id": doc_id, "limit": limit})
    
    # -------------------------------------------------------------------------
    # Section similarity operations
    # -------------------------------------------------------------------------
    
    async def create_section_similarities(self, similarities: List[Dict[str, Any]]) -> int:
        """
        Batch create SIMILAR_TO relationships between sections.
        
        :param similarities: List of dicts with 'from_id', 'to_id', 'score'
        :return: Number of relationships created
        """
        query = """
        UNWIND $similarities AS sim
        MATCH (s1:Section {id: sim.from_id})
        MATCH (s2:Section {id: sim.to_id})
        MERGE (s1)-[r:SIMILAR_TO]->(s2)
        SET r.score = sim.score,
            r.computed_at = datetime()
        """
        await self.run_query(query, {"similarities": similarities})
        return len(similarities)
    
    async def delete_section_similarities(self, doc_id: str) -> int:
        """Delete all SIMILAR_TO relationships for a document."""
        query = """
        MATCH (s:Section {doc_id: $doc_id})-[r:SIMILAR_TO]->()
        DELETE r
        RETURN count(r) as deleted
        """
        result = await self.run_query(query, {"doc_id": doc_id})
        return result[0]["deleted"] if result else 0
    
    async def get_similarity_stats(self, doc_id: str) -> Dict[str, Any]:
        """Get statistics about section similarities in a document."""
        query = """
        MATCH (s:Section {doc_id: $doc_id})
        OPTIONAL MATCH (s)-[r:SIMILAR_TO]->()
        WITH count(DISTINCT s) as total_sections,
             count(r) as total_similarities,
             avg(r.score) as avg_score
        RETURN total_sections,
               total_similarities,
               avg_score,
               CASE WHEN total_sections > 0 
                    THEN toFloat(total_similarities) / total_sections 
                    ELSE 0.0 
               END as avg_similarities_per_section
        """
        result = await self.run_query(query, {"doc_id": doc_id})
        return result[0] if result else {}
    
    # -------------------------------------------------------------------------
    # Document management
    # -------------------------------------------------------------------------
    
    async def update_document_status(self, doc_id: str, status: str, metadata: Optional[Dict] = None):
        """Update document processing status"""
        query = """
        MATCH (d:Document {id: $doc_id})
        SET d.status = $status,
            d.last_modified = datetime()
        """
        params = {"doc_id": doc_id, "status": status}
        if metadata:
            query += ", d.metadata = $metadata"
            params["metadata"] = json.dumps(metadata)
        await self.run_query(query, params)
    
    async def get_document_stats(self, doc_id: Optional[str] = None) -> Dict:
        """Get statistics about documents in the knowledge base"""
        if doc_id:
            # Use CALL subqueries to count each type separately, avoiding cartesian products
            query = """
            MATCH (d:Document {id: $doc_id})
            
            // Optimized: count nodes without loading paths
            WITH d, 
                size([(d)-[:HAS_CHAPTER]->(c:Chapter) | c]) as chapters,
                size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->(s:Section) | s]) as sections,
                size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:PART_OF]->(chunk:TextChunk) | chunk]) as text_chunks,
                size([(d)-[:HAS_SCHEMA]->(sc:Schema) | sc]) + 
                    size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:CONTAINS_SCHEMA]->(sc:Schema) | sc]) as schemas,
                size([(d)-[:HAS_TABLE]->(tb:Table) | tb]) + 
                    size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:CONTAINS_TABLE]->(tb:Table) | tb]) as tables,
                size([(d)-[:HAS_TABLE]->(tb:Table)<-[:PART_OF]-(tc:TableChunk) | tc]) + 
                    size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:CONTAINS_TABLE]->(tb:Table)<-[:PART_OF]-(tc:TableChunk) | tc]) as table_chunks,
                size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:DESCRIBES]->(e:Entity) | e]) +
                    size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:CONTAINS_SCHEMA]->()-[:DEPICTS]->(e:Entity) | e]) +
                    size([(d)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->()-[:CONTAINS_TABLE]->()-[:MENTIONS]->(e:Entity) | e]) as entities
            
            RETURN 
                d.id as doc_id,
                d.title as document_title,
                d.total_pages as total_pages,
                chapters,
                sections,
                text_chunks,
                schemas,
                tables,
                table_chunks,
                entities
            """
            result = await self.run_query(query, {"doc_id": doc_id})
        else:
            # Global stats
            query = """
            MATCH (d:Document)
            WITH count(d) as doc_count
            MATCH (c:Chapter)
            WITH doc_count, count(c) as chapter_count
            MATCH (s:Section)
            WITH doc_count, chapter_count, count(s) as section_count
            MATCH (sc:Schema)
            WITH doc_count, chapter_count, section_count, count(sc) as schema_count
            MATCH (tb:Table)
            WITH doc_count, chapter_count, section_count, schema_count, count(tb) as table_count
            MATCH (tc:TableChunk)
            WITH doc_count, chapter_count, section_count, schema_count, table_count, count(tc) as table_chunk_count
            MATCH (t:Term)
            WITH doc_count, chapter_count, section_count, schema_count, table_count, table_chunk_count, count(t) as term_count
            MATCH (e:Entity)
            RETURN doc_count as documents,
                   chapter_count as chapters,
                   section_count as sections,
                   schema_count as schemas,
                   table_count as tables,
                   table_chunk_count as table_chunks,
                   term_count as terms,
                   count(e) as entities
            """
            result = await self.run_query(query)
    
        return result[0] if result else {}
    
    async def get_all_documents(self, owner: Optional[str] = None) -> List[Dict]:
        """Get all documents, optionally filtered by owner"""
        if owner:
            query = """
            MATCH (d:Document {owner: $owner})
            RETURN d.id AS id, d.title AS title, d.doc_type AS type,
                   d.status AS status, d.total_pages AS total_pages,
                   d.upload_date AS created_at, d.owner AS owner
            ORDER BY d.upload_date DESC
            """
            return await self.run_query(query, {"owner": owner})
        else:
            query = """
            MATCH (d:Document)
            RETURN d.id AS id, d.title AS title, d.doc_type AS type,
                   d.status AS status, d.total_pages AS total_pages,
                   d.upload_date AS created_at, d.owner AS owner
            ORDER BY d.upload_date DESC
            """
            return await self.run_query(query)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete document and ALL related nodes.
        Handles BOTH relationship paths and orphaned nodes cleanup.
        """
        # Step 1: Delete by relationships
        query_by_relationships = """
        MATCH (d:Document {id: $doc_id})
        
        // Delete content via section relationships
        OPTIONAL MATCH (d)-[:HAS_CHAPTER]->(c:Chapter)
        OPTIONAL MATCH (c)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:CONTAINS_SCHEMA]->(sc_section:Schema)
        OPTIONAL MATCH (s)-[:CONTAINS_TABLE]->(tb_section:Table)
        OPTIONAL MATCH (tb_section)<-[:PART_OF]-(tc_section:TableChunk)
        
        // Delete content via direct document relationships
        OPTIONAL MATCH (d)-[:HAS_SCHEMA]->(sc_direct:Schema)
        OPTIONAL MATCH (d)-[:HAS_TABLE]->(tb_direct:Table)
        OPTIONAL MATCH (tb_direct)<-[:PART_OF]-(tc_direct:TableChunk)
        
        // Collect all
        WITH d, c, s, 
            collect(DISTINCT sc_section) + collect(DISTINCT sc_direct) as all_schemas,
            collect(DISTINCT tb_section) + collect(DISTINCT tb_direct) as all_tables,
            collect(DISTINCT tc_section) + collect(DISTINCT tc_direct) as all_chunks
        
        // Delete all nodes
        FOREACH (sc IN all_schemas | DETACH DELETE sc)
        FOREACH (tb IN all_tables | DETACH DELETE tb)
        FOREACH (tc IN all_chunks | DETACH DELETE tc)
        
        DETACH DELETE d, c, s
        
        RETURN count(d) as doc_deleted,
            count(DISTINCT c) as chapters_deleted,
            count(DISTINCT s) as sections_deleted,
            size(all_schemas) as schemas_deleted,
            size(all_tables) as tables_deleted,
            size(all_chunks) as table_chunks_deleted
        """
        
        result = await self.run_query(query_by_relationships, {"doc_id": doc_id})
        
        if not result or result[0]["doc_deleted"] == 0:
            logger.warning(f"Document {doc_id} not found in Neo4j")
            return False
        
        stats = result[0]
        logger.info(
            f"Deleted via batched operations: "
            f"doc=1, chapters={stats['chapters_deleted']}, "
            f"sections={stats['sections_deleted']}, "
            f"schemas={stats['schemas_deleted']}, "
            f"tables={stats['tables_deleted']}, "
            f"table_chunks={stats['table_chunks_deleted']}"
        )
        
        # Step 2: Delete orphaned nodes by doc_id property (batched)
        query_orphan_schemas = """
        MATCH (sc:Schema {doc_id: $doc_id})
        WHERE NOT (sc)<-[:CONTAINS_SCHEMA]-() AND NOT (sc)<-[:HAS_SCHEMA]-()
        WITH sc LIMIT 100
        DETACH DELETE sc
        RETURN count(sc) as deleted
        """
        schemas_orphan = 0
        while True:
            result = await self.run_query(query_orphan_schemas, {"doc_id": doc_id})
            deleted = result[0]["deleted"] if result else 0
            schemas_orphan += deleted
            if deleted == 0:
                break
        
        query_orphan_tables = """
        MATCH (tb:Table {doc_id: $doc_id})
        WHERE NOT (tb)<-[:CONTAINS_TABLE]-() AND NOT (tb)<-[:HAS_TABLE]-()
        WITH tb LIMIT 100
        DETACH DELETE tb
        RETURN count(tb) as deleted
        """
        tables_orphan = 0
        while True:
            result = await self.run_query(query_orphan_tables, {"doc_id": doc_id})
            deleted = result[0]["deleted"] if result else 0
            tables_orphan += deleted
            if deleted == 0:
                break
        
        query_orphan_chunks = """
        MATCH (tc:TableChunk {doc_id: $doc_id})
        WHERE NOT (tc)-[:PART_OF]->()
        WITH tc LIMIT 100
        DETACH DELETE tc
        RETURN count(tc) as deleted
        """
        chunks_orphan = 0
        while True:
            result = await self.run_query(query_orphan_chunks, {"doc_id": doc_id})
            deleted = result[0]["deleted"] if result else 0
            chunks_orphan += deleted
            if deleted == 0:
                break
        
        orphan_stats = {
            "schemas_orphan": schemas_orphan,
            "tables_orphan": tables_orphan,
            "chunks_orphan": chunks_orphan
        }
        
        if schemas_orphan > 0 or tables_orphan > 0:
            logger.warning(
                f"⚠️ Deleted ORPHANED nodes: "
                f"schemas={orphan_stats.get('schemas_orphan', 0)}, "
                f"tables={orphan_stats.get('tables_orphan', 0)}, "
                f"chunks={orphan_stats.get('chunks_orphan', 0)}"
            )

        # Step 3: Clean up ALL orphaned Entities (batched to avoid memory issues)
        query_orphaned_entities = """
        MATCH (e:Entity)
        WHERE NOT (e)<-[:DESCRIBES|DEPICTS|MENTIONS]-()
        WITH e LIMIT 200
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        deleted_entities = 0
        while True:
            result = await self.run_query(query_orphaned_entities)
            deleted = result[0]["deleted"] if result else 0
            deleted_entities += deleted
            if deleted == 0:
                break
        
        if deleted_entities > 0:
            logger.info(f"🧹 Deleted {deleted_entities} orphaned entities")
        
        logger.info(f"✅ Document {doc_id} fully deleted from Neo4j")
        return True

    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        query = """
        MATCH (d:Document {id: $doc_id})
        RETURN d {.*} as document
        """
        
        result = await self.run_query(query, {"doc_id": doc_id})
        
        if not result:
            logger.warning(f"Document {doc_id} not found")
            return None
        
        # Parse metadata JSON if present
        doc = result[0]["document"]
        if "metadata" in doc and isinstance(doc["metadata"], str):
            try:
                doc["metadata"] = json.loads(doc["metadata"])
            except json.JSONDecodeError:
                doc["metadata"] = {}
        
        return doc
   
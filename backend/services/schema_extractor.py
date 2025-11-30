"""
Schema (diagram/figure) extractor for maritime technical documentation.
Uses layout analyzer to detect schema regions and extracts:
- Full-resolution PNG images
- Thumbnails
- Captions from surrounding text
- Text context from page/section
Does NOT parse internal geometry - schemas are treated as atomic visual elements.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import io
import logging
import hashlib
import re
import base64

import fitz  # PyMuPDF
from PIL import Image, ImageOps
from openai import AsyncOpenAI

from services.layout_analyzer import LayoutAnalyzer, RegionType, Region, BBox
from core.config import Settings

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """
    Extract schemas (diagrams, figures, schematics) from PDF documents.
    Schemas are treated as atomic visual elements with text context for retrieval.
    """
    
    def __init__(
        self,
        storage_service,
        layout_analyzer: LayoutAnalyzer,
        llm_service: Optional[AsyncOpenAI] = None,
        zoom: float = 3.0,  # High resolution for technical diagrams
        thumbnail_size: Tuple[int, int] = (600, 600),
        caption_search_distance: int = 250,  # pixels above/below schema (increased)
        surrounding_text_radius: int = 300,  # pixels for surrounding text
        max_nearby_paragraphs: int = 3,  # max paragraphs to extract
        enable_llm_summary: bool = True,  # Enable LLM-based schema description
    ) -> None:
        """
        Initialize schema extractor with enhanced context extraction.
        
        :param storage_service: Storage service for saving images
        :param layout_analyzer: Layout analyzer for region detection
        :param llm_service: OpenAI client for LLM-based schema description
        :param zoom: Zoom factor for rendering (higher = better quality)
        :param thumbnail_size: Thumbnail dimensions
        :param caption_search_distance: Distance to search for captions (pixels)
        :param surrounding_text_radius: Radius for extracting surrounding text
        :param max_nearby_paragraphs: Maximum nearby paragraphs to extract
        :param enable_llm_summary: Enable LLM-based schema description generation
        """
        self.storage = storage_service
        self.layout_analyzer = layout_analyzer
        self.llm_service = llm_service
        self.zoom = zoom
        self.thumbnail_size = thumbnail_size
        self.caption_search_distance = caption_search_distance
        self.surrounding_text_radius = surrounding_text_radius
        self.max_nearby_paragraphs = max_nearby_paragraphs
        self.enable_llm_summary = enable_llm_summary and llm_service is not None
        
        # Initialize settings for LLM
        if self.enable_llm_summary:
            self.settings = Settings()
        
        # Caption patterns - expanded for multilingual support
        self.caption_patterns = [
            re.compile(r'(?i)(?:figure|fig\.?|diagram|schema|drawing|dwg\.?)\s+([0-9]+(?:[.-][0-9]+)*)\s*[:-]?\s*(.+?)(?:\n|$)', re.MULTILINE),
            re.compile(r'(?i)(?:рис(?:унок)?\.?|схема|чертеж)\s+([0-9]+(?:[.-][0-9]+)*)\s*[:-]?\s*(.+?)(?:\n|$)', re.MULTILINE),  # Russian
        ]
        
        # Reference patterns for finding mentions of schema in text
        self.reference_patterns = [
            re.compile(r'(?i)(?:see|refer to|shown in|depicted in|illustrated in|as per)\s+(?:figure|fig\.?|diagram|drawing)\s+([0-9]+(?:[.-][0-9]+)*)', re.IGNORECASE),
            re.compile(r'(?i)(?:figure|fig\.?|diagram|drawing)\s+([0-9]+(?:[.-][0-9]+)*)\s+(?:shows|illustrates|depicts|represents)', re.IGNORECASE),
            re.compile(r'(?i)\((?:see\s+)?(?:figure|fig\.?|diagram|drawing)\s+([0-9]+(?:[.-][0-9]+)*)\)', re.IGNORECASE),
        ]
    
    async def extract_from_page(
        self,
        page: fitz.Page,
        doc_id: str,
        page_num: int,  # zero-based
        text_context: Optional[str] = None,
        section_id: Optional[str] = None,
        doc_id_sanitized: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract schemas from page using layout analyzer.
        
        :param page: PyMuPDF page object
        :param doc_id: Document ID
        :param page_num: Page number (zero-based)
        :param text_context: Text content from page/section for context
        :param section_id: Associated section ID
        :param doc_id_sanitized: Sanitized doc ID for file paths
        :return: List of schema metadata dicts
        """
        schemas = []
        
        try:
            # Detect regions with layout analyzer
            regions = self.layout_analyzer.analyze_page(page, page_num)
            
            # Filter schema regions
            schema_regions = self.layout_analyzer.filter_regions_by_type(
                regions,
                RegionType.SCHEMA,
            )
            
            if not schema_regions:
                logger.debug(f"No schemas detected on page {page_num + 1}")
                return schemas
            
            logger.info(f"Page {page_num + 1}: found {len(schema_regions)} schema(s)")
            
            safe_doc_id = doc_id_sanitized or self._sanitize(doc_id)
            
            # Extract each schema
            for idx, region in enumerate(schema_regions):
                try:
                    schema_data = await self._extract_schema_region(
                        page=page,
                        region=region,
                        doc_id=doc_id,
                        safe_doc_id=safe_doc_id,
                        page_num=page_num,
                        idx=idx,
                        text_context=text_context,
                        section_id=section_id,
                    )
                    
                    if schema_data:
                        schemas.append(schema_data)
                        
                except Exception as e:
                    logger.error(
                        f"Failed to extract schema {idx} from page {page_num + 1}: {e}",
                        exc_info=True,
                    )
            
        except Exception as e:
            logger.error(
                f"Schema extraction failed on page {page_num + 1}: {e}",
                exc_info=True,
            )
        
        return schemas
    
    async def _extract_schema_region(
        self,
        page: fitz.Page,
        region: Region,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        idx: int,
        text_context: Optional[str],
        section_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract single schema region with enhanced context.
        
        :param page: PyMuPDF page object
        :param region: Detected schema region
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param idx: Schema index on page
        :param text_context: Text context from page/section
        :param section_id: Associated section ID
        :return: Schema metadata dict with enhanced context
        """
        bbox = region.bbox
        
        # Filter out tiny schemas (logos, icons, small links)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        bbox_area = bbox.area()
        area_ratio = bbox_area / page_area

        # Minimum thresholds
        min_area_ratio = 0.003  # 0.3% of page area
        min_width = 80          # pixels in page coordinates
        min_height = 80         # pixels in page coordinates

        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0

        if (
            area_ratio < min_area_ratio
            or width < min_width
            or height < min_height
        ):
            logger.debug(
                f"Skipping tiny schema on page {page_num + 1}: "
                f"area_ratio={area_ratio:.5f}, size={width:.1f}x{height:.1f}px, "
                f"bbox={bbox.to_dict()}"
            )
            return None

        logger.debug(
            f"Accepted schema on page {page_num + 1}: "
            f"area_ratio={area_ratio:.5f}, size={width:.1f}x{height:.1f}px"
        )
        
        # Generate unique schema ID
        schema_id = self._generate_schema_id(doc_id, page_num, bbox, idx)
        
        # Render high-resolution PNG
        try:
            crop_png = self._render_bbox_as_png(page, bbox, self.zoom)
        except Exception as e:
            logger.error(f"Failed to render schema crop: {e}")
            return None
        
        # Create thumbnail
        thumb_png = self._make_thumbnail(crop_png, self.thumbnail_size)
        
        # Storage paths
        base_name = f"page_{page_num + 1}_schema_{idx}"
        img_rel = f"schemas/original/{safe_doc_id}/{base_name}.png"
        thumb_rel = f"schemas/thumbnail/{safe_doc_id}/{base_name}.png"
        
        # Save to storage
        img_path = await self.storage.save_file(
            content=crop_png,
            file_path=img_rel,
            content_type="image/png",
        )
        thumb_path = await self.storage.save_file(
            thumb_png,
            thumb_rel,
            content_type="image/png",
        )
        
        # Extract enhanced context
        context = self._extract_enhanced_context(page, bbox, page_num, text_context or "")
        
        # Add page metadata for fallback context
        context['page_number'] = page_num + 1
        context['doc_title'] = doc_id  # Will be replaced with actual title if available
        
        # Generate LLM summary if enabled
        llm_summary = None
        if self.enable_llm_summary:
            llm_summary = await self._generate_llm_summary(crop_png, context)
            if llm_summary:
                context['llm_summary'] = llm_summary
        
        # Build schema metadata with enhanced context
        schema_data = {
            "id": schema_id,
            "doc_id": doc_id,
            "page_number": page_num + 1,
            "title": f"Schema - Page {page_num + 1}.{idx}",
            "caption": context['caption'],
            "bbox": bbox.to_dict(),
            "file_path": img_path,
            "thumbnail_path": thumb_path,
            "confidence": region.confidence,
            "text_context": self._build_rich_context(context),  # Enhanced context with LLM summary
            "llm_summary": llm_summary or "",  # Store separately for Neo4j
            "section_id": section_id,
            "metadata": {
                "nearby_paragraphs": context['nearby_paragraphs'],
                "references": context['references'],
                "surrounding_text": context['surrounding_text'][:500] if context['surrounding_text'] else "",
            }
        }
        
        logger.debug(
            f"Extracted schema {schema_id} from page {page_num + 1}, "
            f"bbox: {bbox.to_dict()}, "
            f"context_length: {len(schema_data['text_context'])} chars"
        )
        
        return schema_data
    
    def _render_bbox_as_png(
        self,
        page: fitz.Page,
        bbox: BBox,
        zoom: float,
    ) -> bytes:
        """
        Render page region as high-quality PNG.
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box to render
        :param zoom: Zoom factor for resolution
        :return: PNG image bytes
        """
        # Create transformation matrix
        mat = fitz.Matrix(zoom, zoom)
        
        # Render clipped region
        clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        
        # Convert to PNG bytes
        png_bytes = pix.tobytes("png")
        
        # Clean up
        pix = None
        
        return png_bytes
    
    def _make_thumbnail(
        self,
        png_bytes: bytes,
        size: Tuple[int, int],
    ) -> bytes:
        """
        Create thumbnail from PNG bytes.
        
        :param png_bytes: Original PNG bytes
        :param size: Thumbnail size (width, height)
        :return: Thumbnail PNG bytes
        """
        img = Image.open(io.BytesIO(png_bytes))
        
        # Preserve aspect ratio
        img = ImageOps.contain(img, size)
        
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        
        return out.getvalue()
    
    def _extract_caption_near_bbox(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> str:
        """
        Extract caption text near schema bbox.
        Searches in regions above and below the schema.
        
        :param page: PyMuPDF page object
        :param bbox: Schema bounding box
        :return: Extracted caption text (empty string if not found)
        """
        try:
            # Define search regions (above and below schema)
            search_regions = [
                # Above schema
                fitz.Rect(
                    bbox.x0,
                    max(0, bbox.y0 - self.caption_search_distance),
                    bbox.x1,
                    bbox.y0,
                ),
                # Below schema
                fitz.Rect(
                    bbox.x0,
                    bbox.y1,
                    bbox.x1,
                    min(page.rect.height, bbox.y1 + self.caption_search_distance),
                ),
            ]
            
            for search_rect in search_regions:
                # Extract text from search region
                text = page.get_text("text", clip=search_rect)
                
                if not text:
                    continue
                
                # Try to match caption patterns
                for pattern in self.caption_patterns:
                    match = pattern.search(text)
                    if match:
                        # Extract figure number and caption text
                        fig_number = match.group(1)
                        caption_text = match.group(2).strip()
                        
                        # Clean up caption
                        caption = f"Figure {fig_number}: {caption_text}"
                        caption = self._clean_text(caption)
                        
                        logger.debug(f"Found caption: {caption}")
                        return caption
            
            # If no pattern match, return first line of text as fallback
            for search_rect in search_regions:
                text = page.get_text("text", clip=search_rect).strip()
                if text:
                    first_line = text.split('\n')[0].strip()
                    if first_line and len(first_line) > 10:  # Reasonable caption length
                        return self._clean_text(first_line)
            
        except Exception as e:
            logger.debug(f"Caption extraction failed: {e}")
        
        return ""
    
    def _extract_enhanced_context(
        self,
        page: fitz.Page,
        bbox: BBox,
        page_num: int,
        full_page_text: str,
    ) -> Dict[str, Any]:
        """
        Extract rich context for schema with nearby paragraphs and references.
        
        :param page: PyMuPDF page object
        :param bbox: Schema bounding box
        :param page_num: Page number (zero-based)
        :param full_page_text: Full page text for reference finding
        :return: Context dict with caption, paragraphs, references, surrounding text
        """
        context = {
            'caption': '',
            'nearby_paragraphs': [],
            'references': [],
            'surrounding_text': '',
        }
        
        # 1. Caption extraction (existing method)
        context['caption'] = self._extract_caption_near_bbox(page, bbox)
        
        # 2. Neighboring paragraphs
        context['nearby_paragraphs'] = self._extract_neighboring_paragraphs(page, bbox)
        
        # 3. References to this schema in page text
        if context['caption']:
            fig_number = self._extract_figure_number(context['caption'])
            if fig_number:
                context['references'] = self._find_references_in_text(
                    full_page_text,
                    fig_number
                )
        
        # 4. Surrounding text (fallback if no paragraphs)
        if not context['nearby_paragraphs']:
            context['surrounding_text'] = self._extract_surrounding_text(page, bbox)
        
        return context
    
    def _extract_neighboring_paragraphs(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> List[str]:
        """
        Extract paragraphs near schema (above and below).
        
        :param page: PyMuPDF page object
        :param bbox: Schema bounding box
        :return: List of paragraph texts
        """
        paragraphs = []
        
        # Expanded search areas
        above_rect = fitz.Rect(
            0,
            max(0, bbox.y0 - 400),
            page.rect.width,
            bbox.y0,
        )
        
        below_rect = fitz.Rect(
            0,
            bbox.y1,
            page.rect.width,
            min(page.rect.height, bbox.y1 + 400),
        )
        
        # Extract text blocks (pre-grouped by paragraph)
        for rect in [above_rect, below_rect]:
            blocks = page.get_text("blocks", clip=rect)
            
            for block in blocks:
                if len(block) >= 5:  # block format: (x0,y0,x1,y1,text,...)
                    text = block[4].strip()
                    
                    # Filter noise
                    if len(text) > 50 and not self._is_noise_text(text):
                        paragraphs.append(text)
                        
                        if len(paragraphs) >= self.max_nearby_paragraphs * 2:
                            break
        
        # Sort by proximity to bbox
        paragraphs_with_distance = []
        for para in paragraphs:
            # Find paragraph position
            para_rects = page.search_for(para[:50])  # first 50 chars
            if para_rects:
                para_rect = para_rects[0]
                distance = self._distance_to_bbox(para_rect, bbox)
                paragraphs_with_distance.append((distance, para))
        
        # Sort by distance, take closest
        paragraphs_with_distance.sort(key=lambda x: x[0])
        
        return [p[1] for p in paragraphs_with_distance[:self.max_nearby_paragraphs]]
    
    def _find_references_in_text(
        self,
        full_text: str,
        figure_number: str,
    ) -> List[str]:
        """
        Find mentions of schema in text (e.g., "see Figure 4-3").
        
        :param full_text: Full page text
        :param figure_number: Figure number to search for
        :return: List of reference contexts
        """
        references = []
        
        # Escape figure number for regex
        fig_num_escaped = re.escape(figure_number)
        
        # Build patterns with escaped figure number
        patterns = [
            re.compile(
                rf'(?i)(?:see|refer to|shown in|depicted in|illustrated in|as per)\s+'
                rf'(?:figure|fig\.?|diagram|drawing)\s+{fig_num_escaped}',
                re.IGNORECASE
            ),
            re.compile(
                rf'(?i)(?:figure|fig\.?|diagram|drawing)\s+{fig_num_escaped}\s+'
                rf'(?:shows|illustrates|depicts|represents)',
                re.IGNORECASE
            ),
            re.compile(
                rf'(?i)\((?:see\s+)?(?:figure|fig\.?|diagram|drawing)\s+{fig_num_escaped}\)',
                re.IGNORECASE
            ),
        ]
        
        for pattern in patterns:
            matches = pattern.finditer(full_text)
            for match in matches:
                # Extract context around match (sentence)
                start = max(0, match.start() - 100)
                end = min(len(full_text), match.end() + 100)
                
                context = full_text[start:end].strip()
                context = self._clean_text(context)
                
                if context and context not in references:
                    references.append(context)
        
        return references
    
    def _extract_surrounding_text(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> str:
        """
        Extract all text around schema in specified radius.
        
        :param page: PyMuPDF page object
        :param bbox: Schema bounding box
        :return: Surrounding text
        """
        # Expanded area around bbox
        surrounding_rect = fitz.Rect(
            max(0, bbox.x0 - self.surrounding_text_radius),
            max(0, bbox.y0 - self.surrounding_text_radius),
            min(page.rect.width, bbox.x1 + self.surrounding_text_radius),
            min(page.rect.height, bbox.y1 + self.surrounding_text_radius),
        )
        
        text = page.get_text("text", clip=surrounding_rect)
        return self._clean_text(text)
    
    def _extract_figure_number(self, caption: str) -> Optional[str]:
        """
        Extract figure number from caption.
        Example: "Figure 4-3: System diagram" → "4-3"
        
        :param caption: Caption text
        :return: Figure number or None
        """
        pattern = re.compile(r'(?i)(?:figure|fig\.?|diagram|drawing)\s+([0-9]+(?:[.-][0-9]+)*)')
        match = pattern.search(caption)
        
        if match:
            return match.group(1)
        
        return None
    
    def _distance_to_bbox(self, rect: fitz.Rect, bbox: BBox) -> float:
        """
        Calculate distance from rect to bbox (center-to-center).
        
        :param rect: fitz.Rect object
        :param bbox: BBox object
        :return: Distance in pixels
        """
        rect_center_x = (rect.x0 + rect.x1) / 2
        rect_center_y = (rect.y0 + rect.y1) / 2
        
        bbox_center_x = (bbox.x0 + bbox.x1) / 2
        bbox_center_y = (bbox.y0 + bbox.y1) / 2
        
        return ((rect_center_x - bbox_center_x)**2 +
                (rect_center_y - bbox_center_y)**2)**0.5
    
    def _is_noise_text(self, text: str) -> bool:
        """
        Filter noisy text (headers, footers, page numbers).
        
        :param text: Text to check
        :return: True if noise
        """
        text_lower = text.lower()
        
        # Page numbers
        if re.match(r'^\s*\d+\s*$', text):
            return True
        
        # Headers/footers keywords
        noise_keywords = ['page', 'chapter', 'section', 'copyright', '©', 'reserved']
        if any(kw in text_lower for kw in noise_keywords) and len(text) < 100:
            return True
        
        return False
    
    async def _generate_llm_summary(self, image_bytes: bytes, context: Dict[str, Any]) -> Optional[str]:
        """
        Generate LLM-based description of schema using vision model.
        
        :param image_bytes: PNG image bytes
        :param context: Existing context (caption, paragraphs, etc.)
        :return: LLM-generated description or None
        """
        if not self.enable_llm_summary:
            return None
        
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Build prompt with available context
            prompt_parts = [
                "Analyze this technical schematic diagram and provide a concise description (2-3 sentences) of:",
                "1. What type of system/component it shows",
                "2. Key elements visible in the diagram",
                "3. The diagram's purpose or function",
            ]
            
            # Add existing context if available
            if context.get('caption'):
                prompt_parts.append(f"\nCaption: {context['caption']}")
            if context.get('nearby_paragraphs'):
                prompt_parts.append(f"\nContext: {context['nearby_paragraphs'][0][:200]}...")
            
            prompt = "\n".join(prompt_parts)
            
            # Call GPT-4 Vision API
            response = await self.llm_service.chat.completions.create(
                model="gpt-4o-mini",  # Vision-capable model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "low"  # Faster, cheaper
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.3,
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated LLM summary for schema ({len(summary)} chars)")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            return None
    
    def _build_rich_context(self, context: Dict[str, Any]) -> str:
        """
        Build rich text context from extracted components.
        
        :param context: Context dict from _extract_enhanced_context
        :return: Combined context string for embedding
        """
        text_parts = []
        
        # 0. LLM Summary (highest priority if available)
        if context.get('llm_summary'):
            text_parts.append(f"Description: {context['llm_summary']}")
        
        # 1. Caption
        if context['caption']:
            text_parts.append(f"Caption: {context['caption']}")
        
        # 2. Nearby paragraphs
        if context['nearby_paragraphs']:
            para_text = "\n\n".join(context['nearby_paragraphs'])
            text_parts.append(f"Context:\n{para_text}")
        
        # 3. References
        if context['references']:
            ref_text = " ".join(context['references'])
            text_parts.append(f"References: {ref_text}")
        
        # 4. Surrounding text (fallback)
        if not context['nearby_paragraphs'] and context['surrounding_text']:
            text_parts.append(f"Surrounding text: {context['surrounding_text'][:500]}")
        
        # Combine all parts
        combined = "\n\n".join(text_parts)
        
        # FALLBACK: If no context found, use OCR text from schema or minimal context
        if not combined.strip():
            # Try OCR text embedded in context (if available)
            ocr_text = context.get('ocr_text', '')
            if ocr_text:
                text_parts.append(f"Schema text (OCR): {ocr_text[:500]}")
                combined = "\n\n".join(text_parts)
            else:
                # Absolute fallback: use page-level metadata
                page_num = context.get('page_number', 'unknown')
                doc_title = context.get('doc_title', 'Technical diagram')
                combined = f"Technical schematic diagram from page {page_num}. Document: {doc_title}"
                
                logger.debug(
                    f"Schema has no text context, using minimal fallback: page {page_num}"
                )
        
        # Truncate if too long (2000 chars for embeddings)
        if len(combined) > 2000:
            combined = combined[:2000] + "..."
        
        return combined
    
    def _generate_schema_id(
        self,
        doc_id: str,
        page_num: int,
        bbox: BBox,
        idx: int,
    ) -> str:
        """Generate stable hash-based ID for schema"""
        basis = f"{doc_id}:{page_num}:{bbox.x0}:{bbox.y0}:{bbox.x1}:{bbox.y1}:{idx}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
    
    def _sanitize(self, s: str) -> str:
        """Sanitize string for use in filenames"""
        return re.sub(r"[^\w\-.]", "_", s)[:100]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        return text.strip()
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length at word boundary"""
        if len(text) <= max_length:
            return text
        
        # Truncate at word boundary
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + "..."
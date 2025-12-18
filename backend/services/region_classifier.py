"""
Region classifier with LLM-based verification for ambiguous cases.
Uses caption detection + LLM for low-confidence YOLO predictions.

SIMPLIFIED APPROACH:
1. Caption detection (fast, free) - "Table X" or "Figure Y"
2. High YOLO confidence (>=0.8) - trust YOLO
3. Low YOLO confidence (<0.8) - use LLM verification
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import base64

import fitz  # PyMuPDF

from services.layout_analyzer import RegionType, Region, BBox

logger = logging.getLogger(__name__)


class RegionClassifier:
    """
    Lightweight region classifier with LLM verification.
    
    Strategy:
    - Caption detection (always) - strong signal
    - YOLO confidence >= 0.8 - trust YOLO
    - YOLO confidence < 0.8 - verify with LLM
    """
    
    # Caption patterns for tables
    TABLE_CAPTION_PATTERNS = [
        re.compile(r'(?i)\btable\s+([0-9]+(?:[.-][0-9]+)*)', re.MULTILINE),
        re.compile(r'(?i)\btab\.?\s+([0-9]+(?:[.-][0-9]+)*)', re.MULTILINE),
        re.compile(r'(?i)^\s*table\s*[:\-]', re.MULTILINE),  # "Table:" or "Table-"
        re.compile(r'(?i)\btable\s+\w+', re.MULTILINE),  # "Table Name" without number
    ]
    
    # Caption patterns for figures/schemas
    FIGURE_CAPTION_PATTERNS = [
        re.compile(r'(?i)\b(?:figure|fig\.?|diagram|drawing|dwg\.?|schema|schematic)\s+([0-9]+(?:[.-][0-9]+)*)', re.MULTILINE),
        re.compile(r'(?i)^\s*(?:figure|fig|diagram|drawing|schema)\s*[:\-]', re.MULTILINE),  # "Figure:" or "Diagram-"
        re.compile(r'<<\s*FIG', re.MULTILINE),  # << FIG. S-03 >> - specific maritime notation
        # NOTE: Removed "Type :" and "Model:" - too generic, catches page headers
        # NOTE: Removed ALL CAPS pattern - catches table headers, not captions
    ]
    
    def __init__(
        self,
        llm_service: Optional[Any] = None,
        caption_search_distance: int = 600,
        yolo_confidence_threshold: float = 0.8,
        enable_llm_verification: bool = True,
    ) -> None:
        """
        Initialize region classifier.
        
        :param llm_service: LLM service for verification (optional)
        :param caption_search_distance: Distance in pixels to search for captions
        :param yolo_confidence_threshold: Threshold for trusting YOLO (default 0.8)
        :param enable_llm_verification: Enable LLM verification for low confidence
        """
        self.llm_service = llm_service
        self.caption_search_distance = caption_search_distance
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.enable_llm_verification = enable_llm_verification and llm_service is not None
        
        # Statistics for logging
        self.stats = {
            'total_classified': 0,
            'caption_detected': 0,
            'high_confidence_yolo': 0,
            'llm_verified': 0,
            'llm_changed_decision': 0,
        }
        
        logger.info(
            f"RegionClassifier initialized: "
            f"LLM={'enabled' if self.enable_llm_verification else 'disabled'}, "
            f"confidence_threshold={yolo_confidence_threshold}"
        )
    
    async def reclassify_region(
        self,
        page: fitz.Page,
        region: Region,
    ) -> RegionType:
        """
        Reclassify region with simplified logic:
        1. TOC detection (override YOLO if it's a table of contents)
        2. Caption detection (strong signal)
        3. High YOLO confidence - trust it
        4. Low YOLO confidence - verify with LLM
        
        :param page: PyMuPDF page object
        :param region: Detected region from YOLO
        :return: Reclassified region type (TABLE or SCHEMA)
        """
        self.stats['total_classified'] += 1
        
        # Only reclassify TABLE and SCHEMA regions
        if region.region_type not in [RegionType.TABLE, RegionType.SCHEMA]:
            logger.debug(f"Skipping non-TABLE/SCHEMA region: {region.region_type.value}")
            return region.region_type
        
        page_num = region.page_number + 1
        yolo_type = region.region_type.value
        yolo_conf = region.confidence
        
        logger.info(
            f"🔍 Page {page_num}: Classifying YOLO={yolo_type} "
            f"(confidence={yolo_conf:.3f}, bbox=[{region.bbox.x0:.0f},{region.bbox.y0:.0f},"
            f"{region.bbox.x1:.0f},{region.bbox.y1:.0f}])"
        )
        
        # STEP 0: TOC detection (if YOLO detected table, check if it's actually TOC)
        if region.region_type == RegionType.TABLE:
            if self._is_toc_region(page, region.bbox):
                logger.info(
                    f"📋 Page {page_num}: Detected TOC content → Treating as TEXT "
                    f"(YOLO misclassified TOC as table)"
                )
                return RegionType.TEXT  # TOC should be processed as text
        
        # STEP 1: Caption detection (search CLOSE to region - max 50px above/below)
        caption_type, caption_text = self._detect_caption_type(page, region.bbox)
        logger.debug(f"Caption detection result: type={caption_type}, text={caption_text[:50] if caption_text else 'None'}")
        
        # Save caption to region regardless of whether it affects classification
        if caption_text:
            region.caption_text = caption_text
            logger.debug(f"💾 Saved caption to region: {caption_text[:50]}")
        
        # Caption provides strong signal for type determination
        # BUT: If YOLO confident it's TABLE (conf >= 0.8) and we found figure caption,
        # might be misdetection (table column headers mistaken for caption) → keep TABLE
        
        if caption_type == 'table':
            self.stats['caption_detected'] += 1
            logger.info(
                f"📊 Page {page_num}: Table caption detected → TABLE "
                f"(found '{caption_text[:50]}...' near region)"
            )
            return RegionType.TABLE
        
        elif caption_type == 'figure':
            # Check if YOLO was very confident about TABLE type
            if yolo_type == RegionType.TABLE and yolo_conf >= 0.8:
                self.stats['caption_detected'] += 1
                logger.info(
                    f"📊 Page {page_num}: Figure caption found but YOLO confident TABLE "
                    f"(conf={yolo_conf:.3f}) → Keep TABLE type"
                )
                # Keep TABLE type, caption saved for reference
                return RegionType.TABLE
            else:
                # Trust figure caption (YOLO not confident or not TABLE)
                self.stats['caption_detected'] += 1
                logger.info(
                    f"🖼️ Page {page_num}: Figure caption detected → SCHEMA "
                    f"(found '{caption_text[:50]}...' near region)"
                )
                return RegionType.SCHEMA
        
        # STEP 2: High YOLO confidence - trust it (caption already saved above if found)
        if yolo_conf >= self.yolo_confidence_threshold:
            self.stats['high_confidence_yolo'] += 1
            logger.info(
                f"✅ Page {page_num}: High YOLO confidence → {yolo_type.upper()} "
                f"(conf={yolo_conf:.3f} >= {self.yolo_confidence_threshold})"
            )
            return region.region_type
        
        # STEP 3: Low YOLO confidence - use LLM verification
        if self.enable_llm_verification:
            logger.info(
                f"🤖 Page {page_num}: Low YOLO confidence ({yolo_conf:.3f}), "
                f"using LLM verification..."
            )
            
            try:
                llm_type = await self._llm_verify_type(page, region, page_num)
                self.stats['llm_verified'] += 1
                
                if llm_type != region.region_type:
                    self.stats['llm_changed_decision'] += 1
                    logger.warning(
                        f"🔄 Page {page_num}: LLM OVERRIDE - "
                        f"YOLO said {yolo_type}, LLM says {llm_type.value} "
                        f"(YOLO conf={yolo_conf:.3f})"
                    )
                else:
                    logger.info(
                        f"✅ Page {page_num}: LLM confirms YOLO → {llm_type.value}"
                    )
                
                return llm_type
                
            except Exception as e:
                logger.error(
                    f"❌ Page {page_num}: LLM verification failed: {e}, "
                    f"falling back to YOLO prediction ({yolo_type})"
                )
                return region.region_type
        
        # Fallback: trust YOLO even with low confidence
        logger.info(
            f"⚠️ Page {page_num}: LLM disabled, trusting YOLO → {yolo_type} "
            f"(conf={yolo_conf:.3f})"
        )
        return region.region_type
    
    async def _llm_verify_type(
        self,
        page: fitz.Page,
        region: Region,
        page_num: int = 0,
        use_high_detail: bool = False,
    ) -> RegionType:
        """
        Use LLM to verify region type with retry logic for rate limits.
        
        :param page: PyMuPDF page object
        :param region: Region to verify
        :param page_num: Page number for logging
        :param use_high_detail: If True, uses "high" detail for better schema detection (more expensive)
        :return: Verified region type
        """
        import asyncio
        from openai import RateLimitError
        
        # Retry configuration
        max_retries = 3
        retry_delay = 0.5  # Start with 0.5 seconds
        
        # Render region as image
        image_bytes = self._render_region_as_png(page, region.bbox)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Simple, focused prompt with TEXT option
        prompt = """Analyze this image from a technical maritime document.

What is this?

- TABLE: structured data grid with rows and columns (specifications, lists, data tables)
- SCHEMA: technical diagram (P&ID, electrical schematic, equipment layout, flowchart, drawing)
- TEXT: regular text paragraph, heading, or text block (NOT a table or diagram)

Answer with ONLY one word: TABLE, SCHEMA, or TEXT"""

        # Call LLM with retry logic
        for attempt in range(max_retries):
            try:
                response = await self.llm_service.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": "high" if use_high_detail else "low"  # High detail for re-verification
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
                break  # Success, exit retry loop
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⏳ Page {page_num}: Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Page {page_num}: Rate limit exceeded after {max_retries} retries")
                    raise  # Re-raise to be caught by outer try/except
        
        # Parse response
        answer = response.choices[0].message.content.strip().upper()
        
        if "TABLE" in answer:
            return RegionType.TABLE
        elif "SCHEMA" in answer:
            return RegionType.SCHEMA
        elif "TEXT" in answer:
            # Return TEXT type - will be handled as regular text extraction
            logger.info(f"📝 LLM detected TEXT block (not table/schema), will process as text")
            return RegionType.TEXT
        else:
            # Fallback to YOLO if unclear
            logger.warning(f"LLM unclear response: '{answer}', using YOLO prediction")
            return region.region_type
    
    def _render_region_as_png(
        self,
        page: fitz.Page,
        bbox: BBox,
        zoom: float = 2.0,
    ) -> bytes:
        """
        Render page region as PNG for LLM analysis.
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box to render
        :param zoom: Zoom factor for resolution
        :return: PNG image bytes
        """
        mat = fitz.Matrix(zoom, zoom)
        clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        png_bytes = pix.tobytes("png")
        pix = None  # Release memory
        return png_bytes
    
    def _detect_caption_type(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect caption type and extract caption text.
        
        :param page: PyMuPDF page object
        :param bbox: Region bbox
        :return: Tuple of (caption_type, caption_text) - both can be None
        """
        # Search regions above and below bbox (real captions are OUTSIDE the region)
        # NOTE: Text INSIDE bbox for tables is usually headers, not captions
        # Search CLOSE to the region (max 50px above or below)
        # This prevents matching page headers or distant text as captions
        max_caption_distance = 50  # pixels
        
        search_regions = [
            # Above the region (but not too far)
            fitz.Rect(
                max(0, bbox.x0 - 50),
                max(0, bbox.y0 - max_caption_distance),
                min(page.rect.width, bbox.x1 + 50),
                bbox.y0,
            ),
            # Below the region (but not too far)
            fitz.Rect(
                max(0, bbox.x0 - 50),
                bbox.y1,
                min(page.rect.width, bbox.x1 + 50),
                min(page.rect.height, bbox.y1 + max_caption_distance),
            ),
        ]
        
        for i, search_rect in enumerate(search_regions):
            text = page.get_text("text", clip=search_rect)
            
            if not text:
                continue
            
            location = "above" if i == 0 else "below"
            
            # Check table patterns first (higher priority)
            for pattern in self.TABLE_CAPTION_PATTERNS:
                match = pattern.search(text)
                if match:
                    caption_text = match.group(0).strip()
                    logger.info(f"✅ Found table caption {location} bbox (within {max_caption_distance}px): {caption_text[:50]}")
                    return ('table', caption_text)
            
            # Check figure patterns
            for pattern in self.FIGURE_CAPTION_PATTERNS:
                match = pattern.search(text)
                if match:
                    caption_text = match.group(0).strip()
                    logger.info(f"✅ Found figure caption {location} bbox (within {max_caption_distance}px): {caption_text[:50]}")
                    return ('figure', caption_text)
        
        return (None, None)
    
    def _is_toc_region(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> bool:
        """
        Detect if region is actually a Table of Contents (TOC).
        
        TOC characteristics:
        - Contains numbered lines like "1.0 TITLE .... 5" or "3.0 TITLE    12"
        - Has multiple dotted lines with page numbers
        - Contains hierarchical numbering (1.0, 1.1, 2.0, etc.)
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box to check
        :return: True if region appears to be TOC
        """
        # Extract text from region
        clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        text = page.get_text("text", clip=clip_rect)
        
        if not text or len(text) < 50:
            return False
        
        lines = text.split('\n')
        
        # Count TOC-like patterns
        toc_indicators = 0
        total_substantial_lines = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip very short lines
            if len(line_stripped) < 5:
                continue
            
            total_substantial_lines += 1
            
            # Pattern 1: "1.0 TITLE .... 5" (dotted leader with page number)
            if re.search(r'^\d+\.?\d*\s+.+[\.…·\-_]{3,}\s*\d+\s*$', line_stripped):
                toc_indicators += 1
                continue
            
            # Pattern 2: "1.0    TITLE    5" (multiple spaces with page number at end)
            if re.search(r'^\d+\.?\d*\s+.+\s{3,}\d+\s*$', line_stripped):
                toc_indicators += 1
                continue
            
            # Pattern 3: Hierarchical numbering with title (1.0, 1.1, 2.0, etc.)
            if re.search(r'^\d+\.\d+\s+[A-Z]', line_stripped):
                toc_indicators += 1
                continue
        
        # If > 50% of lines match TOC patterns, it's likely a TOC
        if total_substantial_lines > 0:
            toc_ratio = toc_indicators / total_substantial_lines
            if toc_ratio > 0.5 and toc_indicators >= 3:
                return True
        
        # Additional check: look for TOC header nearby
        # Check text above the region
        header_region = fitz.Rect(
            max(0, bbox.x0 - 50),
            max(0, bbox.y0 - 100),
            min(page.rect.width, bbox.x1 + 50),
            bbox.y0
        )
        header_text = page.get_text("text", clip=header_region)
        
        if re.search(r'(?:TABLE\s+OF\s+)?CONTENTS?|INDEX', header_text, re.IGNORECASE):
            return True
        
        return False
    
    def log_statistics(self) -> None:
        """Log classification statistics"""
        total = self.stats['total_classified']
        if total == 0:
            logger.info("No regions classified yet")
            return
        
        logger.info("="*80)
        logger.info("REGION CLASSIFICATION STATISTICS")
        logger.info("="*80)
        logger.info(f"Total regions classified: {total}")
        logger.info(f"Caption detected: {self.stats['caption_detected']} "
                   f"({100*self.stats['caption_detected']/total:.1f}%)")
        logger.info(f"High confidence YOLO: {self.stats['high_confidence_yolo']} "
                   f"({100*self.stats['high_confidence_yolo']/total:.1f}%)")
        logger.info(f"LLM verified: {self.stats['llm_verified']} "
                   f"({100*self.stats['llm_verified']/total:.1f}%)")
        if self.stats['llm_verified'] > 0:
            logger.info(f"LLM changed decision: {self.stats['llm_changed_decision']} "
                       f"({100*self.stats['llm_changed_decision']/self.stats['llm_verified']:.1f}% of LLM calls)")
        logger.info("="*80)

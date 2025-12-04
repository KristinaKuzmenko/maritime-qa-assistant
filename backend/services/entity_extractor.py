"""
Entity extraction and normalization service.

Uses structured dictionary for:
1. Consistent entity codes (normalized IDs)
2. Alias resolution (FO pump → fuel_oil_pump)
3. Hierarchy inference (fuel oil pump → part of fuel oil system)
4. Question entity extraction (for entity-based search)
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Maritime domain entity extractor with normalization.
    
    Features:
    - Dictionary-based extraction (not just regex)
    - Alias and abbreviation resolution
    - System-component hierarchy linking
    - Optimized for both ingestion and query-time extraction
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize extractor with entity dictionary.
        
        :param dictionary_path: Path to entity_dictionary.json
        """
        if dictionary_path is None:
            # Default path relative to this file
            dictionary_path = Path(__file__).parent.parent / "data" / "entity_dictionary.json"
        
        self.dictionary = self._load_dictionary(dictionary_path)
        
        # Build lookup indexes for fast extraction
        self._system_lookup: Dict[str, str] = {}  # keyword/alias → system_code
        self._component_patterns: List[Tuple[re.Pattern, str]] = []  # (pattern, component_type)
        self._abbreviation_lookup: Dict[str, str] = {}  # abbreviation → system_code
        
        self._build_indexes()
        
        logger.info(
            f"EntityExtractor initialized: "
            f"{len(self._system_lookup)} system keywords, "
            f"{len(self._component_patterns)} component patterns, "
            f"{len(self._abbreviation_lookup)} abbreviations"
        )
    
    def _load_dictionary(self, path: Path) -> Dict[str, Any]:
        """Load entity dictionary from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Entity dictionary not found at {path}, using empty dictionary")
            return {"systems": {}, "component_types": {}, "qualifiers": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in entity dictionary: {e}")
            return {"systems": {}, "component_types": {}, "qualifiers": {}}
    
    def _build_indexes(self):
        """Build lookup indexes from dictionary for fast extraction."""
        systems = self.dictionary.get("systems", {})
        components = self.dictionary.get("component_types", {})
        
        # Build system lookup (keyword → code)
        for sys_key, sys_data in systems.items():
            code = sys_data.get("code", f"sys_{sys_key}")
            
            # Add canonical name
            canonical = sys_data.get("canonical", "").lower()
            if canonical:
                self._system_lookup[canonical] = code
            
            # Add aliases
            for alias in sys_data.get("aliases", []):
                self._system_lookup[alias.lower()] = code
            
            # Add keywords
            for keyword in sys_data.get("keywords", []):
                self._system_lookup[keyword.lower()] = code
            
            # Add abbreviations (separate lookup for context-aware matching)
            for abbr in sys_data.get("abbreviations", []):
                self._abbreviation_lookup[abbr.upper()] = code
        
        # Build component patterns with STRICT qualifier matching
        # Only allow known qualifiers before component
        self._valid_qualifiers = {
            'main', 'auxiliary', 'aux', 'standby', 'emergency', 'backup',
            'primary', 'secondary', 'supply', 'transfer', 'booster', 'feed',
            'fuel', 'oil', 'water', 'air', 'steam', 'gas', 'exhaust',
            'cooling', 'heating', 'lubricating', 'lube', 'service',
            'hot', 'cold', 'fresh', 'sea', 'bilge', 'ballast', 'cargo',
            'inlet', 'outlet', 'suction', 'discharge', 'return',
            'high', 'low', 'pressure', 'temperature', 'temp',
            'electric', 'electrical', 'manual', 'automatic', 'auto',
            'safety', 'relief', 'control', 'isolation', 'check', 'shut',
            'storage', 'settling', 'day', 'overflow', 'drain',
        }
        
        # Stop words that should NEVER be part of component name
        self._stop_words = {
            'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'this', 'that', 'these', 'those', 'all', 'each', 'every', 'any',
            'no', 'not', 'and', 'or', 'but', 'if', 'when', 'while', 'as',
            'from', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'out', 'off',
            'can', 'could', 'will', 'would', 'should', 'must', 'may', 'might',
            'push', 'pull', 'check', 'ensure', 'verify', 'operate', 'operating',
            'damage', 'damaged', 'activate', 'higher', 'lower', 'than',
            'general', 'operation', 'include', 'within', 'between',
        }
        
        for comp_key, comp_data in components.items():
            patterns = comp_data.get("patterns", [comp_key])
            
            for pattern in patterns:
                # Simpler pattern - just capture component with optional 1-2 word qualifier
                regex = re.compile(
                    rf'\b((?:\w+\s+)?(?:\w+\s+)?){re.escape(pattern)}s?\b',
                    re.IGNORECASE
                )
                self._component_patterns.append((regex, comp_key))
    
    def extract_from_text(
        self,
        text: str,
        extract_systems: bool = True,
        extract_components: bool = True,
        link_hierarchy: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract entities from text with normalization.
        
        :param text: Text to extract from
        :param extract_systems: Whether to extract system entities
        :param extract_components: Whether to extract component entities  
        :param link_hierarchy: Whether to infer system from component context
        :return: Dict with 'systems', 'components', 'entity_ids'
        """
        text_lower = text.lower()
        
        found_systems: Set[str] = set()
        found_components: List[Dict[str, Any]] = []
        
        # Extract systems
        if extract_systems:
            found_systems = self._extract_systems(text_lower)
        
        # Extract components
        if extract_components:
            found_components = self._extract_components(text, text_lower)
        
        # Link components to systems if not explicitly found
        if link_hierarchy and found_components:
            found_systems = self._infer_system_hierarchy(
                found_systems, found_components, text_lower
            )
        
        # Build entity_ids list (for Qdrant payload)
        entity_ids = list(found_systems)
        for comp in found_components:
            entity_ids.append(comp["code"])
        
        return {
            "systems": list(found_systems),
            "components": found_components,
            "entity_ids": entity_ids,
        }
    
    def extract_from_question(self, question: str) -> Dict[str, Any]:
        """
        Extract entities from user question for search.
        More aggressive extraction - includes abbreviations in isolation.
        
        :param question: User question
        :return: Dict with entity info for search
        """
        question_lower = question.lower()
        question_upper = question.upper()
        
        found_systems: Set[str] = set()
        found_components: List[Dict[str, Any]] = []
        
        # Extract systems (including abbreviation matching)
        found_systems = self._extract_systems(question_lower)
        
        # Check for standalone abbreviations (common in questions)
        words = re.findall(r'\b[A-Z]{2,5}\b', question_upper)
        for word in words:
            if word in self._abbreviation_lookup:
                found_systems.add(self._abbreviation_lookup[word])
        
        # Extract components
        found_components = self._extract_components(question, question_lower)
        
        # Infer hierarchy
        found_systems = self._infer_system_hierarchy(
            found_systems, found_components, question_lower
        )
        
        # Get human-readable names for logging/debugging
        system_names = []
        for sys_code in found_systems:
            for sys_key, sys_data in self.dictionary.get("systems", {}).items():
                if sys_data.get("code") == sys_code:
                    system_names.append(sys_data.get("canonical", sys_key))
                    break
        
        component_names = [c["name"] for c in found_components]
        
        return {
            "systems": list(found_systems),
            "system_names": system_names,
            "components": found_components,
            "component_names": component_names,
            "entity_ids": list(found_systems) + [c["code"] for c in found_components],
        }
    
    def _extract_systems(self, text_lower: str) -> Set[str]:
        """Extract system codes from lowercased text."""
        found = set()
        
        # 1. Dictionary-based extraction (sorted by length, longest first)
        sorted_keywords = sorted(
            self._system_lookup.keys(),
            key=len,
            reverse=True
        )
        
        for keyword in sorted_keywords:
            if keyword in text_lower:
                found.add(self._system_lookup[keyword])
        
        # 2. Fallback systems - DISABLED for ingestion (too much noise)
        # Only enable for search queries if needed
        # fallback_systems = self._extract_fallback_systems(text_lower)
        # found.update(fallback_systems)
        
        return found
    
    def _extract_fallback_systems(self, text_lower: str) -> Set[str]:
        """
        Extract systems not in dictionary using generic patterns.
        Creates normalized codes for unknown systems.
        
        NOTE: This is DISABLED by default due to high false positive rate.
        Only use for explicit system discovery tasks.
        """
        found = set()
        
        # Pattern: "X system" or "X oil system" etc.
        system_pattern = re.compile(
            r'\b(\w+(?:\s+\w+)?)\s+system\b',
            re.IGNORECASE
        )
        
        # Additional blocklist for fallback systems
        fallback_blocklist = {
            'the', 'this', 'that', 'each', 'any', 'all', 'a', 'an',
            'of', 'to', 'in', 'on', 'by', 'for', 'with', 'from',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'ensuring', 'control', 'entire', 'whole', 'general',
            'following', 'respective', 'particular', 'specific',
            'same', 'other', 'such', 'own', 'operating',
        }
        
        for match in system_pattern.finditer(text_lower):
            system_name = match.group(1).strip()
            
            # Skip very short names
            if len(system_name) < 4:
                continue
                
            # Skip if starts with blocklisted word
            first_word = system_name.split()[0] if ' ' in system_name else system_name
            if first_word in fallback_blocklist:
                continue
            
            # Skip generic phrases
            if system_name in fallback_blocklist:
                continue
            
            # Check if not already in dictionary
            full_phrase = f"{system_name} system"
            if full_phrase not in self._system_lookup:
                # Generate normalized code (same format as dictionary)
                normalized = system_name.replace(' ', '_').replace('-', '_')
                code = f"sys_{normalized}"
                found.add(code)
                logger.debug(f"Fallback system extracted: '{full_phrase}' → {code}")
        
        return found
    
    def _extract_components(
        self,
        text: str,
        text_lower: str,
    ) -> List[Dict[str, Any]]:
        """Extract component entities with STRICT qualifier validation."""
        found = []
        seen_codes = set()  # Deduplicate by normalized code
        seen_positions = set()  # Avoid duplicate extractions
        
        # 1. Dictionary-based extraction (known patterns)
        for pattern, comp_type in self._component_patterns:
            for match in pattern.finditer(text_lower):
                start, end = match.span()
                
                # Skip if overlapping with already extracted
                if any(start < seen_end and end > seen_start 
                       for seen_start, seen_end in seen_positions):
                    continue
                
                # Get full match from original text (preserve case)
                full_name = text[start:end].strip()
                
                # STRICT cleaning - validate qualifiers
                full_name = self._clean_component_name(full_name)
                
                if not full_name or len(full_name) < 3:
                    continue
                
                # Generate normalized code
                code = self._generate_component_code(full_name, comp_type)
                
                # Skip duplicates (same code = same entity)
                if code in seen_codes:
                    continue
                
                seen_codes.add(code)
                seen_positions.add((start, end))
                
                found.append({
                    "name": full_name,
                    "type": comp_type,
                    "code": code,
                    "position": start,
                    "source": "dictionary",
                })
        
        # 2. Extract equipment codes (P-101, V-205, etc.) - these are always valid
        equipment_codes = self._extract_equipment_codes(text)
        for eq_code in equipment_codes:
            if eq_code["code"] not in seen_codes:
                if not any(eq_code["position"] >= s and eq_code["position"] < e 
                           for s, e in seen_positions):
                    found.append(eq_code)
                    seen_codes.add(eq_code["code"])
                    seen_positions.add((eq_code["position"], eq_code["position"] + len(eq_code["name"])))
        
        # 3. Fallback components - DISABLED (too much noise for ingestion)
        # fallback_components = self._extract_fallback_components(text, text_lower, seen_positions, seen_codes)
        # found.extend(fallback_components)
        
        return found
    
    def _extract_equipment_codes(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract equipment tag codes like P-101, V-205, HE-001, etc.
        Common in maritime/industrial documentation.
        """
        found = []
        
        # Pattern: Letter(s) + hyphen/dash + numbers (optional suffix)
        # Examples: P-101, V-205A, HE-001, TK-102, ME-01
        pattern = re.compile(
            r'\b([A-Z]{1,4})[-–](\d{2,4}[A-Z]?)\b'
        )
        
        # Equipment code prefixes and their types
        code_types = {
            'P': 'pump',
            'V': 'valve', 
            'TK': 'tank',
            'HE': 'heat_exchanger',
            'FLT': 'filter',
            'SEP': 'separator',
            'ME': 'main_engine',
            'AE': 'auxiliary_engine',
            'GE': 'generator',
            'C': 'compressor',
            'BLR': 'boiler',
        }
        
        for match in pattern.finditer(text):
            prefix = match.group(1)
            number = match.group(2)
            full_code = f"{prefix}-{number}"
            
            # Determine component type from prefix
            comp_type = code_types.get(prefix, "equipment")
            
            found.append({
                "name": full_code,
                "type": comp_type,
                "code": f"eq_{prefix.lower()}_{number.lower()}",
                "position": match.start(),
                "source": "equipment_code",
                "is_equipment_tag": True,
            })
        
        return found
    
    def _extract_fallback_components(
        self,
        text: str,
        text_lower: str,
        seen_positions: Set[Tuple[int, int]],
        seen_codes: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        Fallback extraction for components not in dictionary.
        Uses generic patterns for common maritime equipment.
        
        NOTE: DISABLED by default - call only for explicit component discovery.
        """
        found = []
        
        # Generic component keywords that might not be in dictionary
        generic_patterns = [
            (r'\b(\w+\s+)?heater\b', 'heater'),
            (r'\b(\w+\s+)?cooler\b', 'cooler'),
            (r'\b(\w+\s+)?preheater\b', 'preheater'),
            (r'\b(\w+\s+)?purifier\b', 'purifier'),
            (r'\b(\w+\s+)?clarifier\b', 'clarifier'),
            (r'\b(\w+\s+)?centrifuge\b', 'centrifuge'),
            (r'\b(\w+\s+)?evaporator\b', 'evaporator'),
            (r'\b(\w+\s+)?condenser\b', 'condenser'),
            (r'\b(\w+\s+)?economizer\b', 'economizer'),
            (r'\b(\w+\s+)?turbocharger\b', 'turbocharger'),
            (r'\b(\w+\s+)?governor\b', 'governor'),
            (r'\b(\w+\s+)?injector\b', 'injector'),
            (r'\b(\w+\s+)?nozzle\b', 'nozzle'),
            (r'\b(\w+\s+)?bearing\b', 'bearing'),
            (r'\b(\w+\s+)?shaft\b', 'shaft'),
            (r'\b(\w+\s+)?piston\b', 'piston'),
            (r'\b(\w+\s+)?cylinder\b', 'cylinder'),
            (r'\b(\w+\s+)?crankcase\b', 'crankcase'),
        ]
        
        for pattern_str, comp_type in generic_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            
            for match in pattern.finditer(text_lower):
                start, end = match.span()
                
                # Skip if overlapping
                if any(start < seen_end and end > seen_start 
                       for seen_start, seen_end in seen_positions):
                    continue
                
                full_name = text[start:end].strip()
                full_name = self._clean_component_name(full_name)
                
                if not full_name or len(full_name) < 4:
                    continue
                
                code = self._generate_component_code(full_name, comp_type)
                
                # Skip duplicates
                if code in seen_codes:
                    continue
                
                found.append({
                    "name": full_name,
                    "type": comp_type,
                    "code": code,
                    "position": start,
                    "source": "fallback",
                })
                
                seen_codes.add(code)
                seen_positions.add((start, end))
        
        return found
    
    def _clean_component_name(self, name: str) -> str:
        """
        Clean component name - STRICT validation.
        
        Only allows known qualifier patterns:
        - Valid adjectives: main, auxiliary, primary, etc.
        - Valid substance qualifiers: fuel, oil, water, etc.  
        - Valid position qualifiers: inlet, outlet, suction, etc.
        
        Rejects:
        - Verbs, prepositions, articles as qualifiers
        - Phrases that look like sentences
        """
        if not name:
            return ""
            
        name = name.strip()
        words = name.split()
        
        if not words:
            return ""
        
        # Last word should be the component type (pump, valve, heater, etc.)
        component_word = words[-1].lower()
        
        # If only one word, return it as-is (just the component)
        if len(words) == 1:
            return name
        
        # For multi-word names, validate qualifiers
        qualifiers = words[:-1]  # All words except the component type
        valid_qualifiers = []
        
        for q in qualifiers:
            q_lower = q.lower()
            
            # Skip stop words completely
            if q_lower in self._stop_words:
                continue
            
            # Only accept known valid qualifiers
            if q_lower in self._valid_qualifiers:
                valid_qualifiers.append(q)
            else:
                # Unknown word - could be valid or garbage
                # Accept if it looks like a proper noun (capitalized) or acronym
                if q.isupper() and len(q) <= 4:  # Acronym like "FO", "HFO"
                    valid_qualifiers.append(q)
                # Reject anything else - be strict
        
        # Rebuild name with valid qualifiers only
        if valid_qualifiers:
            # Max 3 qualifiers to prevent long garbage names
            valid_qualifiers = valid_qualifiers[-3:]  # Keep last 3 (closest to component)
            cleaned = ' '.join(valid_qualifiers + [words[-1]])
        else:
            # No valid qualifiers - just return component type
            cleaned = words[-1]
        
        # Final validation: minimum length
        if len(cleaned) < 3:
            return ""
            
        return cleaned.strip()
    
    def _generate_component_code(self, name: str, comp_type: str) -> str:
        """Generate normalized component code."""
        # Normalize name for code
        normalized = name.lower()
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        normalized = '_'.join(normalized.split())
        
        # Truncate if too long
        if len(normalized) > 40:
            normalized = normalized[:40]
        
        return f"comp_{comp_type}_{normalized}"
    
    def _infer_system_hierarchy(
        self,
        found_systems: Set[str],
        found_components: List[Dict[str, Any]],
        text_lower: str,
    ) -> Set[str]:
        """
        Infer parent system from component context.
        E.g., "fuel oil pump" → add sys_fuel_oil
        """
        systems = self.dictionary.get("systems", {})
        
        for comp in found_components:
            comp_name_lower = comp["name"].lower()
            
            # Check if component name contains system keywords
            for sys_key, sys_data in systems.items():
                code = sys_data.get("code", f"sys_{sys_key}")
                
                # Skip if already found
                if code in found_systems:
                    continue
                
                # Check keywords in component name
                for keyword in sys_data.get("keywords", []):
                    if keyword.lower() in comp_name_lower:
                        found_systems.add(code)
                        logger.debug(f"Inferred system {code} from component '{comp['name']}'")
                        break
        
        return found_systems
    
    def get_system_hierarchy(self, system_code: str) -> List[str]:
        """
        Get hierarchy path for a system (child → parent).
        
        :param system_code: System code (e.g., 'sys_sw_cooling')
        :return: List of codes from this system to root
        """
        hierarchy = [system_code]
        systems = self.dictionary.get("systems", {})
        
        # Find system in dictionary
        current_code = system_code
        max_depth = 5  # Prevent infinite loops
        
        for _ in range(max_depth):
            parent_found = False
            
            for sys_key, sys_data in systems.items():
                if sys_data.get("code") == current_code:
                    parent_key = sys_data.get("parent")
                    if parent_key and parent_key in systems:
                        parent_code = systems[parent_key].get("code", f"sys_{parent_key}")
                        hierarchy.append(parent_code)
                        current_code = parent_code
                        parent_found = True
                        break
            
            if not parent_found:
                break
        
        return hierarchy
    
    def expand_entity_ids(self, entity_ids: List[str]) -> List[str]:
        """
        Expand entity IDs with hierarchy (add parent systems).
        
        :param entity_ids: List of entity codes
        :return: Expanded list including parent systems
        """
        expanded = set(entity_ids)
        
        for entity_id in entity_ids:
            if entity_id.startswith("sys_"):
                hierarchy = self.get_system_hierarchy(entity_id)
                expanded.update(hierarchy)
        
        return list(expanded)


# Singleton instance for reuse
_extractor_instance: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get singleton EntityExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractor()
    return _extractor_instance

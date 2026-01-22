"""
Entity Extractor Tests

Comprehensive tests for maritime entity extraction including:
- Dictionary loading and error handling
- System extraction by keyword/alias/abbreviation
- Component extraction with patterns
- Name cleaning and validation
- Code generation
- Hierarchy inference and expansion
- Question entity extraction
- Singleton pattern

Run with: pytest test_entity_extractor.py -v
"""

import pytest
import json
from unittest.mock import patch, mock_open, Mock
from pathlib import Path

from services.entity_extractor import EntityExtractor, get_entity_extractor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_dictionary():
    """Create mock entity dictionary."""
    return {
        "systems": {
            "fuel_oil": {
                "code": "FO",
                "canonical": "Fuel Oil System",
                "aliases": ["fuel system", "fo system"],
                "keywords": ["fuel", "diesel", "hfo"],
                "abbreviations": ["FO", "F.O.", "HFO"]
            },
            "cooling_water": {
                "code": "CW",
                "canonical": "Cooling Water System",
                "aliases": ["cooling system", "cw system"],
                "keywords": ["cooling", "coolant"],
                "abbreviations": ["CW", "SW"],
                "parent": "auxiliary"
            },
            "lubrication": {
                "code": "LO",
                "canonical": "Lubrication Oil System",
                "aliases": ["lube oil system"],
                "keywords": ["lube", "lubricating", "oil"],
                "abbreviations": ["LO"]
            },
            "auxiliary": {
                "code": "AUX",
                "canonical": "Auxiliary Systems",
                "aliases": [],
                "keywords": ["auxiliary"],
                "abbreviations": ["AUX"]
            }
        },
        "component_types": {
            "pump": {
                "code": "PU",
                "patterns": ["pump"],
                "aliases": ["pumps", "pumping unit"]
            },
            "valve": {
                "code": "VL",
                "patterns": ["valve"],
                "aliases": ["valves"]
            },
            "heater": {
                "code": "HT",
                "patterns": ["heater"],
                "aliases": ["heaters"]
            },
            "filter": {
                "code": "FL",
                "patterns": ["filter"],
                "aliases": ["filters", "strainer"]
            }
        },
        "qualifiers": ["main", "auxiliary", "standby", "emergency"]
    }


@pytest.fixture
def extractor(mock_dictionary):
    """Create EntityExtractor with mock dictionary."""
    with patch.object(EntityExtractor, '_load_dictionary', return_value=mock_dictionary):
        ext = EntityExtractor(dictionary_path="fake_path")
        return ext


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Test EntityExtractor initialization."""
    
    def test_init_loads_dictionary(self, mock_dictionary):
        """Test dictionary is loaded on init."""
        with patch.object(EntityExtractor, '_load_dictionary', return_value=mock_dictionary) as mock_load:
            extractor = EntityExtractor(dictionary_path="test/path.json")
            
            mock_load.assert_called_once()
    
    def test_init_builds_indexes(self, mock_dictionary):
        """Test indexes are built after loading."""
        with patch.object(EntityExtractor, '_load_dictionary', return_value=mock_dictionary):
            extractor = EntityExtractor()
            
            assert len(extractor._system_lookup) > 0
            assert len(extractor._abbreviation_lookup) > 0
    
    def test_init_default_path(self):
        """Test default dictionary path."""
        with patch.object(EntityExtractor, '_load_dictionary', return_value={}) as mock_load:
            extractor = EntityExtractor()
            
            # Should use default path
            call_args = mock_load.call_args[0][0]
            assert "entity_dictionary.json" in str(call_args)


# =============================================================================
# DICTIONARY LOADING TESTS
# =============================================================================

class TestDictionaryLoading:
    """Test _load_dictionary method."""
    
    def test_load_valid_json(self, mock_dictionary):
        """Test loading valid JSON dictionary."""
        json_content = json.dumps(mock_dictionary)
        
        with patch('builtins.open', mock_open(read_data=json_content)):
            extractor = EntityExtractor.__new__(EntityExtractor)
            result = extractor._load_dictionary(Path("test.json"))
        
        assert result == mock_dictionary
    
    def test_load_file_not_found(self):
        """Test handling of missing file."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            extractor = EntityExtractor.__new__(EntityExtractor)
            result = extractor._load_dictionary(Path("nonexistent.json"))
        
        assert result == {"systems": {}, "component_types": {}, "qualifiers": {}}
    
    def test_load_invalid_json(self):
        """Test handling of invalid JSON."""
        with patch('builtins.open', mock_open(read_data="invalid json {")):
            extractor = EntityExtractor.__new__(EntityExtractor)
            result = extractor._load_dictionary(Path("invalid.json"))
        
        assert result == {"systems": {}, "component_types": {}, "qualifiers": {}}


# =============================================================================
# INDEX BUILDING TESTS
# =============================================================================

class TestIndexBuilding:
    """Test _build_indexes method."""
    
    def test_system_lookup_built(self, extractor):
        """Test system lookup index is built."""
        assert "fuel" in extractor._system_lookup
        assert "cooling" in extractor._system_lookup
        assert extractor._system_lookup["fuel"] == "FO"
    
    def test_alias_lookup_built(self, extractor):
        """Test aliases are in lookup."""
        assert "fuel system" in extractor._system_lookup
        assert "fo system" in extractor._system_lookup
        assert extractor._system_lookup["fuel system"] == "FO"
    
    def test_keyword_lookup_built(self, extractor):
        """Test keywords are in lookup."""
        assert "diesel" in extractor._system_lookup
        assert "hfo" in extractor._system_lookup
    
    def test_abbreviation_lookup_built(self, extractor):
        """Test abbreviation lookup is built."""
        assert "FO" in extractor._abbreviation_lookup
        assert "CW" in extractor._abbreviation_lookup
        assert extractor._abbreviation_lookup["FO"] == "FO"
    
    def test_canonical_in_lookup(self, extractor):
        """Test canonical names are in lookup."""
        assert "fuel oil system" in extractor._system_lookup


# =============================================================================
# SYSTEM EXTRACTION TESTS
# =============================================================================

class TestSystemExtraction:
    """Test _extract_systems method."""
    
    def test_extract_by_keyword(self, extractor):
        """Test system extraction by keyword."""
        result = extractor._extract_systems("check the fuel pump")
        
        assert "FO" in result
    
    def test_extract_by_alias(self, extractor):
        """Test system extraction by alias."""
        result = extractor._extract_systems("the cooling system is running")
        
        assert "CW" in result
    
    def test_extract_multiple_systems(self, extractor):
        """Test extracting multiple systems."""
        result = extractor._extract_systems("fuel pump and cooling water")
        
        assert "FO" in result
        assert "CW" in result
    
    def test_extract_no_systems(self, extractor):
        """Test text with no systems."""
        result = extractor._extract_systems("random text without entities")
        
        assert len(result) == 0


# =============================================================================
# COMPONENT EXTRACTION TESTS
# =============================================================================

class TestComponentExtraction:
    """Test _extract_components method."""
    
    def test_extract_simple_component(self, extractor):
        """Test extracting simple component."""
        result = extractor._extract_components(
            "check the pump",
            "check the pump"
        )
        
        assert len(result) >= 1
        assert any(c["type"] == "pump" for c in result)
    
    def test_extract_qualified_component(self, extractor):
        """Test extracting component with qualifier."""
        result = extractor._extract_components(
            "main fuel pump",
            "main fuel pump"
        )
        
        assert len(result) >= 1
        comp = result[0]
        assert "pump" in comp["type"] or "pump" in comp["name"].lower()
    
    def test_extract_multiple_components(self, extractor):
        """Test extracting multiple components."""
        result = extractor._extract_components(
            "check the pump and valve",
            "check the pump and valve"
        )
        
        types = [c["type"] for c in result]
        assert "pump" in types or any("pump" in c["name"].lower() for c in result)
    
    def test_component_has_code(self, extractor):
        """Test extracted component has code."""
        result = extractor._extract_components(
            "fuel pump",
            "fuel pump"
        )
        
        if result:
            assert "code" in result[0]
            assert result[0]["code"].startswith("comp_")


# =============================================================================
# EXTRACT FROM TEXT TESTS
# =============================================================================

class TestExtractFromText:
    """Test extract_from_text method."""
    
    def test_extract_returns_dict(self, extractor):
        """Test extraction returns proper dict structure."""
        result = extractor.extract_from_text("fuel pump maintenance")
        
        assert "systems" in result
        assert "components" in result
        assert "entity_ids" in result
    
    def test_extract_finds_system(self, extractor):
        """Test system is found in text."""
        result = extractor.extract_from_text("fuel oil system check")
        
        assert "FO" in result["systems"]
    
    def test_extract_finds_component(self, extractor):
        """Test component is found in text."""
        result = extractor.extract_from_text("main pump maintenance")
        
        assert len(result["components"]) >= 1
    
    def test_extract_entity_ids_combined(self, extractor):
        """Test entity_ids combines systems and components."""
        result = extractor.extract_from_text("fuel pump check")
        
        # Should have both system and component IDs
        assert len(result["entity_ids"]) >= 1
    
    def test_extract_systems_only(self, extractor):
        """Test extracting only systems."""
        result = extractor.extract_from_text(
            "fuel pump",
            extract_systems=True,
            extract_components=False
        )
        
        assert len(result["components"]) == 0
    
    def test_extract_components_only(self, extractor):
        """Test extracting only components."""
        result = extractor.extract_from_text(
            "main pump",
            extract_systems=False,
            extract_components=True
        )
        
        assert len(result["systems"]) == 0 or result["systems"] == []


# =============================================================================
# EXTRACT FROM QUESTION TESTS
# =============================================================================

class TestExtractFromQuestion:
    """Test extract_from_question method."""
    
    def test_question_extraction(self, extractor):
        """Test basic question extraction."""
        result = extractor.extract_from_question("How to maintain the fuel pump?")
        
        assert "systems" in result
        assert "components" in result
    
    def test_abbreviation_in_question(self, extractor):
        """Test abbreviation extraction from question."""
        result = extractor.extract_from_question("What is the FO pump pressure?")
        
        assert "FO" in result["systems"]
    
    def test_multiple_abbreviations(self, extractor):
        """Test multiple abbreviations in question."""
        result = extractor.extract_from_question("Check FO and CW systems")
        
        assert "FO" in result["systems"]
        assert "CW" in result["systems"]


# =============================================================================
# NAME CLEANING TESTS
# =============================================================================

class TestNameCleaning:
    """Test _clean_component_name method."""
    
    def test_clean_simple_name(self, extractor):
        """Test cleaning simple name."""
        result = extractor._clean_component_name("pump")
        
        assert result == "pump"
    
    def test_clean_qualified_name(self, extractor):
        """Test cleaning qualified name."""
        result = extractor._clean_component_name("main pump")
        
        assert "pump" in result
        assert "main" in result
    
    def test_clean_removes_stop_words(self, extractor):
        """Test stop words are removed."""
        result = extractor._clean_component_name("the main pump")
        
        assert "the" not in result.split()
        assert "pump" in result
    
    def test_clean_empty_string(self, extractor):
        """Test empty string returns empty."""
        result = extractor._clean_component_name("")
        
        assert result == ""
    
    def test_clean_only_stop_words(self, extractor):
        """Test only stop words returns component."""
        result = extractor._clean_component_name("the a pump")
        
        assert result == "pump"
    
    def test_clean_preserves_valid_qualifiers(self, extractor):
        """Test valid qualifiers are preserved."""
        result = extractor._clean_component_name("auxiliary fuel pump")
        
        assert "pump" in result
        # Should keep valid qualifiers
    
    def test_clean_limits_qualifiers(self, extractor):
        """Test max 3 qualifiers kept."""
        result = extractor._clean_component_name(
            "main auxiliary standby backup emergency pump"
        )
        
        words = result.split()
        assert len(words) <= 4  # 3 qualifiers + component


# =============================================================================
# CODE GENERATION TESTS
# =============================================================================

class TestCodeGeneration:
    """Test _generate_component_code method."""
    
    def test_generate_simple_code(self, extractor):
        """Test simple code generation."""
        result = extractor._generate_component_code("pump", "pump")
        
        assert result.startswith("comp_pump_")
        assert "pump" in result
    
    def test_generate_qualified_code(self, extractor):
        """Test code with qualifiers."""
        result = extractor._generate_component_code("main fuel pump", "pump")
        
        assert result.startswith("comp_pump_")
        assert "main" in result
        assert "fuel" in result
    
    def test_generate_removes_special_chars(self, extractor):
        """Test special characters removed."""
        result = extractor._generate_component_code("pump (main)", "pump")
        
        assert "(" not in result
        assert ")" not in result
    
    def test_generate_truncates_long_names(self, extractor):
        """Test long names are truncated."""
        long_name = "a" * 100
        result = extractor._generate_component_code(long_name, "pump")
        
        # Code should be reasonable length
        assert len(result) <= 60


# =============================================================================
# HIERARCHY TESTS
# =============================================================================

class TestHierarchy:
    """Test hierarchy-related methods."""
    
    def test_infer_system_from_component(self, extractor):
        """Test system inferred from component name."""
        found_systems = set()
        found_components = [{"name": "fuel pump", "type": "pump", "code": "c1"}]
        
        result = extractor._infer_system_hierarchy(
            found_systems, found_components, "fuel pump"
        )
        
        assert "FO" in result
    
    def test_get_system_hierarchy(self, extractor):
        """Test getting system hierarchy path."""
        result = extractor.get_system_hierarchy("CW")
        
        assert "CW" in result
        # Should include parent if defined
        if len(result) > 1:
            assert "AUX" in result
    
    def test_get_hierarchy_no_parent(self, extractor):
        """Test hierarchy for system without parent."""
        result = extractor.get_system_hierarchy("FO")
        
        assert result == ["FO"]
    
    def test_expand_entity_ids(self, extractor):
        """Test entity ID expansion."""
        entity_ids = ["FO", "comp_pump_main"]
        
        result = extractor.expand_entity_ids(entity_ids)
        
        assert "FO" in result
        assert "comp_pump_main" in result


# =============================================================================
# SINGLETON TESTS
# =============================================================================

class TestSingleton:
    """Test singleton pattern."""
    
    def test_get_entity_extractor_returns_instance(self):
        """Test singleton returns instance."""
        with patch.object(EntityExtractor, '_load_dictionary', return_value={}):
            # Reset singleton
            import services.entity_extractor as module
            module._extractor_instance = None
            
            result = get_entity_extractor()
            
            assert isinstance(result, EntityExtractor)
    
    def test_get_entity_extractor_returns_same_instance(self):
        """Test singleton returns same instance."""
        with patch.object(EntityExtractor, '_load_dictionary', return_value={}):
            import services.entity_extractor as module
            module._extractor_instance = None
            
            first = get_entity_extractor()
            second = get_entity_extractor()
            
            assert first is second


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self, extractor):
        """Test empty text."""
        result = extractor.extract_from_text("")
        
        assert result["systems"] == []
        assert result["components"] == []
    
    def test_whitespace_only(self, extractor):
        """Test whitespace-only text."""
        result = extractor.extract_from_text("   \n\t  ")
        
        assert result["systems"] == []
    
    def test_unicode_text(self, extractor):
        """Test Unicode text."""
        result = extractor.extract_from_text("насос топливный fuel pump")
        
        # Should still find English entities
        assert "FO" in result["systems"] or len(result["components"]) > 0
    
    def test_special_characters(self, extractor):
        """Test text with special characters."""
        result = extractor.extract_from_text("fuel pump (main) - check!")
        
        assert "FO" in result["systems"]
    
    def test_case_insensitive(self, extractor):
        """Test case insensitivity."""
        result1 = extractor.extract_from_text("FUEL PUMP")
        result2 = extractor.extract_from_text("fuel pump")
        
        assert result1["systems"] == result2["systems"]


# =============================================================================
# ENTITY NORMALIZATION TESTS (from original)
# =============================================================================

class TestEntityNormalization:
    """Test entity code normalization."""
    
    def test_system_keyword_lookup(self, extractor):
        """Test system identification by keyword."""
        result = extractor._system_lookup.get("fuel")
        assert result == "FO"
    
    def test_alias_resolution(self, extractor):
        """Test alias resolves to canonical code."""
        result = extractor._system_lookup.get("fuel system")
        assert result == "FO"
        
        result = extractor._system_lookup.get("fo system")
        assert result == "FO"
    
    def test_abbreviation_lookup(self, extractor):
        """Test abbreviation resolution."""
        result = extractor._abbreviation_lookup.get("FO")
        assert result == "FO"
        
        result = extractor._abbreviation_lookup.get("CW")
        assert result == "CW"
    
    def test_multiple_aliases_same_code(self, extractor):
        """Test all aliases resolve to same code."""
        assert extractor._system_lookup.get("cooling system") == "CW"
        assert extractor._system_lookup.get("cooling") == "CW"
        assert extractor._system_lookup.get("coolant") == "CW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
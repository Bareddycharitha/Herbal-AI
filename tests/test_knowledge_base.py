"""
Tests for Knowledge Base Classes
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from ai.recommendation.knowledge_base import KnowledgeBase
from ai.recommendation.herbal_knowledge_base import HerbalKnowledgeBase
from ai.herb.knowledge_base import HerbKnowledgeBase


class TestKnowledgeBase:
    """Tests for disease KnowledgeBase."""

    @pytest.fixture
    def disease_kb_data(self):
        """Sample disease knowledge base data."""
        return {
            "diseases": [
                {
                    "label": "Acne",
                    "description": "Acne is a common skin condition...",
                    "symptoms": ["pimples", "blackheads", "inflammation"],
                    "causes": ["hormones", "bacteria", "excess oil"],
                    "prevention": ["clean face regularly", "avoid touching face"],
                    "self_care": ["use gentle cleanser", "moisturize"],
                    "when_to_consult_doctor": "If severe or persistent",
                    "medical_disclaimer": "Not medical advice",
                    "recommended_herbs": [
                        {"name": "Tea Tree", "efficacy": 8, "weight": 0.8},
                        {"name": "Aloe Vera", "efficacy": 7, "weight": 0.7},
                    ],
                },
                {
                    "label": "Eczema",
                    "description": "Eczema causes dry, itchy skin...",
                    "symptoms": ["dry skin", "itching", "redness"],
                    "causes": ["genetics", "immune system", "environment"],
                    "prevention": ["moisturize daily", "avoid triggers"],
                    "self_care": ["use fragrance-free products"],
                    "when_to_consult_doctor": "If widespread",
                    "medical_disclaimer": "Not medical advice",
                    "recommended_herbs": [
                        {"name": "Chamomile", "efficacy": 9, "weight": 0.9},
                    ],
                },
            ]
        }

    @pytest.fixture
    def temp_kb_file(self, disease_kb_data):
        """Create temporary KB file."""
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(disease_kb_data, f)
        f.close()  # Close so it can be reopened on Windows
        yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_load_kb(self, temp_kb_file):
        """Test loading knowledge base."""
        kb = KnowledgeBase(temp_kb_file)
        assert kb.disease_count == 2

    def test_get_disease_exact_match(self, temp_kb_file):
        """Test getting disease by exact name."""
        kb = KnowledgeBase(temp_kb_file)
        disease = kb.get_disease("Acne")
        assert disease is not None
        assert disease["label"] == "Acne"

    def test_get_disease_case_insensitive(self, temp_kb_file):
        """Test case-insensitive disease lookup."""
        kb = KnowledgeBase(temp_kb_file)
        disease = kb.get_disease("acne")
        assert disease is not None
        assert disease["label"] == "Acne"
        disease = kb.get_disease("ACNE")
        assert disease is not None

    def test_get_disease_not_found(self, temp_kb_file):
        """Test getting non-existent disease."""
        kb = KnowledgeBase(temp_kb_file)
        disease = kb.get_disease("NonExistent")
        assert disease is None

    def test_get_recommendations(self, temp_kb_file):
        """Test getting herbal recommendations."""
        kb = KnowledgeBase(temp_kb_file)
        herbs = kb.get_recommendations("Acne")
        assert len(herbs) == 2
        assert herbs[0]["name"] == "Tea Tree"
        assert herbs[1]["name"] == "Aloe Vera"

    def test_get_recommendations_not_found(self, temp_kb_file):
        """Test recommendations for non-existent disease."""
        kb = KnowledgeBase(temp_kb_file)
        herbs = kb.get_recommendations("NonExistent")
        assert herbs == []

    def test_get_disease_information(self, temp_kb_file):
        """Test getting formatted disease information."""
        kb = KnowledgeBase(temp_kb_file)
        info = kb.get_disease_information("Acne")
        assert info["description"] == "Acne is a common skin condition..."
        assert info["symptoms"] == ["pimples", "blackheads", "inflammation"]
        assert info["causes"] == ["hormones", "bacteria", "excess oil"]
        assert info["prevention"] == ["clean face regularly", "avoid touching face"]

    def test_get_disease_information_not_found(self, temp_kb_file):
        """Test disease info for non-existent disease."""
        kb = KnowledgeBase(temp_kb_file)
        info = kb.get_disease_information("NonExistent")
        assert info == {}

    def test_list_diseases(self, temp_kb_file):
        """Test listing all diseases."""
        kb = KnowledgeBase(temp_kb_file)
        diseases = kb.list_diseases()
        assert set(diseases) == {"Acne", "Eczema"}

    def test_hot_reload(self, temp_kb_file, disease_kb_data):
        """Test hot reload functionality."""
        # Create KB with hot reload
        kb = KnowledgeBase(temp_kb_file, hot_reload=True)
        assert kb.disease_count == 2

        # Modify file
        disease_kb_data["diseases"].append({
            "label": "Psoriasis",
            "description": "Psoriasis is...",
            "symptoms": ["scaly patches"],
            "causes": ["immune system"],
            "prevention": [],
            "self_care": [],
            "when_to_consult_doctor": "",
            "medical_disclaimer": "",
            "recommended_herbs": [],
        })
        with open(temp_kb_file, "w") as f:
            json.dump(disease_kb_data, f)

        # Wait for reload
        time.sleep(3)

        # Check reload happened
        assert kb.disease_count == 3
        assert "Psoriasis" in kb.list_diseases()


class TestHerbalKnowledgeBase:
    """Tests for HerbalKnowledgeBase."""

    @pytest.fixture
    def herbal_kb_data(self):
        """Sample herbal knowledge base data."""
        return {
            "Tea Tree": {
                "name": "Tea Tree",
                "botanical_name": "Melaleuca alternifolia",
                "family": "Myrtaceae",
                "active_compounds": ["terpinen-4-ol"],
                "phytochemicals": ["terpenes"],
                "benefits": ["antimicrobial", "anti-inflammatory"],
                "preparation_method": "Topical application",
                "side_effects": ["skin irritation"],
                "contraindications": ["pregnancy"],
                "research_papers": ["PMID:12345"],
                "skin_types": ["oily", "acne-prone"],
                "evidence_level": "moderate",
            },
            "Aloe Vera": {
                "name": "Aloe Vera",
                "botanical_name": "Aloe barbadensis",
                "family": "Asphodelaceae",
                "active_compounds": ["aloin", "acemannan"],
                "phytochemicals": ["polysaccharides"],
                "benefits": ["soothing", "healing", "moisturizing"],
                "preparation_method": "Gel extraction",
                "side_effects": [],
                "contraindications": [],
                "research_papers": [],
                "skin_types": ["all"],
                "evidence_level": "high",
            },
        }

    @pytest.fixture
    def temp_kb_file(self, herbal_kb_data):
        """Create temporary KB file."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(herbal_kb_data, f)
        f.close()
        yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_load_kb(self, temp_kb_file):
        """Test loading herbal knowledge base."""
        kb = HerbalKnowledgeBase(temp_kb_file)
        assert kb.herb_count == 2

    def test_get_herb(self, temp_kb_file):
        """Test getting herb details."""
        kb = HerbalKnowledgeBase(temp_kb_file)
        herb = kb.get_herb("Tea Tree")
        assert herb is not None
        assert herb["name"] == "Tea Tree"
        assert herb["botanical_name"] == "Melaleuca alternifolia"
        assert herb["family"] == "Myrtaceae"
        assert herb["benefits"] == ["antimicrobial", "anti-inflammatory"]

    def test_get_herb_not_found(self, temp_kb_file):
        """Test getting non-existent herb."""
        kb = HerbalKnowledgeBase(temp_kb_file)
        herb = kb.get_herb("NonExistent")
        assert herb is None

    def test_list_herbs(self, temp_kb_file):
        """Test listing all herbs."""
        kb = HerbalKnowledgeBase(temp_kb_file)
        herbs = kb.list_herbs()
        assert set(herbs) == {"Tea Tree", "Aloe Vera"}

    def test_hot_reload(self, temp_kb_file, herbal_kb_data):
        """Test hot reload functionality."""
        kb = HerbalKnowledgeBase(temp_kb_file, hot_reload=True)
        assert kb.herb_count == 2

        # Add new herb
        herbal_kb_data["Chamomile"] = {
            "name": "Chamomile",
            "botanical_name": "Matricaria chamomilla",
            "family": "Asteraceae",
            "active_compounds": ["bisabolol"],
            "phytochemicals": ["flavonoids"],
            "benefits": ["calming", "anti-inflammatory"],
            "preparation_method": "Tea or topical",
            "side_effects": ["allergic reactions"],
            "contraindications": ["allergy to Asteraceae"],
            "research_papers": [],
            "skin_types": ["sensitive"],
            "evidence_level": "moderate",
        }
        with open(temp_kb_file, "w") as f:
            json.dump(herbal_kb_data, f)

        time.sleep(3)

        assert kb.herb_count == 3
        assert "Chamomile" in kb.list_herbs()


class TestHerbKnowledgeBase:
    """Tests for HerbKnowledgeBase (herb module)."""

    @pytest.fixture
    def herbal_kb_data(self):
        """Sample herbal knowledge base data."""
        return {
            "Neem": {
                "name": "Neem",
                "botanical_name": "Azadirachta indica",
                "family": "Meliaceae",
                "active_compounds": ["azadirachtin"],
                "phytochemicals": ["limonoids"],
                "benefits": ["antibacterial", "antifungal", "anti-inflammatory"],
                "preparation_method": "Paste or decoction",
                "side_effects": ["possible skin irritation"],
                "contraindications": ["pregnancy"],
                "research_papers": [],
                "skin_types": ["oily", "acne-prone"],
                "evidence_level": "high",
            }
        }

    @pytest.fixture
    def temp_kb_file(self, herbal_kb_data):
        """Create temporary KB file."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(herbal_kb_data, f)
        f.close()
        yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_load_kb(self, temp_kb_file):
        """Test loading herb knowledge base."""
        kb = HerbKnowledgeBase(temp_kb_file)
        assert kb.herb_count == 1

    def test_get_herb(self, temp_kb_file):
        """Test getting herb details."""
        kb = HerbKnowledgeBase(temp_kb_file)
        herb = kb.get_herb("Neem")
        assert herb is not None
        assert herb["name"] == "Neem"
        assert herb["botanical_name"] == "Azadirachta indica"

    def test_exists(self, temp_kb_file):
        """Test exists method."""
        kb = HerbKnowledgeBase(temp_kb_file)
        assert kb.exists("Neem") is True
        assert kb.exists("NonExistent") is False

    def test_list_herbs(self, temp_kb_file):
        """Test listing all herbs."""
        kb = HerbKnowledgeBase(temp_kb_file)
        herbs = kb.list_herbs()
        assert herbs == ["Neem"]


# ==========================================================
# Integration Tests Against Real Knowledge Base Files
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "ai" / "datasets" / "knowledge_base"

DISEASE_KB_PATH = KB_DIR / "disease_knowledge_base.json"
HERBAL_KB_PATH = KB_DIR / "herbal_knowledge_base.json"
CLASS_MAPPING_PATH = KB_DIR / "class_to_kb_mapping.json"

# Fields expected in disease entries (used by KnowledgeBase.get_disease_information)
REQUIRED_DISEASE_FIELDS = {
    "id", "label", "description", "symptoms",
    "self_care", "when_to_consult_doctor", "medical_disclaimer",
}


class TestDiseaseKnowledgeBaseIntegration:
    """
    Integration tests against the real disease knowledge base file.

    These tests verify the actual production knowledge base JSON is valid,
    well-structured, and contains the fields the application relies on.
    """

    def test_disease_kb_file_exists(self):
        """Verify the disease knowledge base file exists on disk."""
        assert DISEASE_KB_PATH.exists(), (
            f"Disease knowledge base not found at {DISEASE_KB_PATH}"
        )

    def test_disease_kb_json_parses(self):
        """Verify the disease knowledge base is valid JSON."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_disease_kb_has_diseases_key(self):
        """Verify the knowledge base has a 'diseases' key."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "diseases" in data
        assert isinstance(data["diseases"], list)

    def test_disease_kb_not_empty(self):
        """Verify the knowledge base is not empty."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["diseases"]) > 0, "Disease knowledge base has no disease entries"

    def test_disease_entries_have_required_fields(self):
        """Verify every disease entry has the fields the application uses."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for disease in data["diseases"]:
            label = disease.get("label", "<unknown>")
            for field_name in REQUIRED_DISEASE_FIELDS:
                assert field_name in disease, (
                    f"Disease '{label}' is missing required field: {field_name}"
                )

    def test_disease_entries_have_recommended_herbs(self):
        """Verify disease entries contain recommended_herbs lists."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for disease in data["diseases"]:
            label = disease.get("label", "<unknown>")
            assert "recommended_herbs" in disease, (
                f"Disease '{label}' is missing 'recommended_herbs'"
            )
            assert isinstance(disease["recommended_herbs"], list)

    def test_recommended_herbs_reference_valid_names(self):
        """Verify recommended herb entries have a 'name' field."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for disease in data["diseases"]:
            for herb in disease.get("recommended_herbs", []):
                assert "name" in herb, (
                    f"Recommended herb in '{disease.get('label')}' is missing 'name'"
                )
                assert isinstance(herb["name"], str)
                assert len(herb["name"]) > 0

    def test_recommended_herbs_exist_in_herbal_kb(self):
        """Verify disease-recommended herbs exist in the herbal KB (where applicable)."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            disease_data = json.load(f)
        with open(HERBAL_KB_PATH, "r", encoding="utf-8") as f:
            herbal_data = json.load(f)
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)

        herbal_keys = set(herbal_data.keys())
        # Build set of all valid herb names (keys + mapped names)
        all_valid_herbs = herbal_keys | set(class_mapping.values())

        for disease in disease_data["diseases"]:
            for herb in disease.get("recommended_herbs", []):
                herb_name = herb["name"]
                assert herb_name in all_valid_herbs, (
                    f"Recommended herb '{herb_name}' for disease "
                    f"'{disease.get('label')}' not found in herbal knowledge base"
                )

    def test_disease_labels_are_unique(self):
        """Verify disease labels are unique."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels = [d["label"] for d in data["diseases"]]
        assert len(labels) == len(set(labels)), (
            f"Duplicate disease labels found: {[l for l in labels if labels.count(l) > 1]}"
        )

    def test_disease_kb_loads_via_knowledge_base_class(self):
        """Verify the KnowledgeBase class can load the real disease KB."""
        kb = KnowledgeBase(DISEASE_KB_PATH)
        assert kb.disease_count > 0
        # Verify we can look up at least one disease
        diseases = kb.list_diseases()
        assert len(diseases) > 0
        # Pick one and look it up
        info = kb.get_disease_information(diseases[0])
        assert "description" in info

    def test_disease_kb_has_metadata(self):
        """Verify the knowledge base has metadata."""
        with open(DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)


class TestHerbalKnowledgeBaseIntegration:
    """
    Integration tests against the real herbal knowledge base file.
    """

    def test_herbal_kb_file_exists(self):
        """Verify the herbal knowledge base file exists on disk."""
        assert HERBAL_KB_PATH.exists(), (
            f"Herbal knowledge base not found at {HERBAL_KB_PATH}"
        )

    def test_herbal_kb_json_parses(self):
        """Verify the herbal knowledge base is valid JSON."""
        with open(HERBAL_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_herbal_kb_not_empty(self):
        """Verify the knowledge base is not empty."""
        with open(HERBAL_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, "Herbal knowledge base has no entries"

    def test_herbal_entries_have_required_fields(self):
        """Verify every herb entry has the fields the application uses."""
        # Fields returned by HerbalKnowledgeBase.get_herb
        required_fields = {
            "name", "botanical_name", "family", "active_compounds",
            "phytochemicals", "benefits", "preparation_method",
            "side_effects", "contraindications", "research_papers",
            "skin_types", "evidence_level",
        }
        with open(HERBAL_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for herb_name, herb_data in data.items():
            for field_name in required_fields:
                assert field_name in herb_data, (
                    f"Herb '{herb_name}' is missing required field: {field_name}"
                )

    def test_herbal_kb_loads_via_herbal_knowledge_base_class(self):
        """Verify the HerbalKnowledgeBase class can load the real herbal KB."""
        kb = HerbalKnowledgeBase(HERBAL_KB_PATH)
        assert kb.herb_count > 0
        herbs = kb.list_herbs()
        assert len(herbs) > 0

    def test_herbal_kb_get_herb_returns_data(self):
        """Verify get_herb returns properly formatted data for a known herb."""
        kb = HerbalKnowledgeBase(HERBAL_KB_PATH)
        herbs = kb.list_herbs()
        first_herb_name = herbs[0]
        herb = kb.get_herb(first_herb_name)
        assert herb is not None
        assert "name" in herb
        assert "benefits" in herb
        assert "botanical_name" in herb


class TestClassMappingIntegration:
    """
    Integration tests against the real class-to-KB mapping file.
    """

    def test_class_mapping_file_exists(self):
        """Verify the class-to-KB mapping file exists."""
        assert CLASS_MAPPING_PATH.exists(), (
            f"Class mapping not found at {CLASS_MAPPING_PATH}"
        )

    def test_class_mapping_json_parses(self):
        """Verify the class mapping is valid JSON."""
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_class_mapping_not_empty(self):
        """Verify the class mapping is not empty."""
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0

    def test_class_mapping_references_exist_in_herbal_kb(self):
        """Verify all mapped KB keys exist in the herbal knowledge base."""
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
        with open(HERBAL_KB_PATH, "r", encoding="utf-8") as f:
            herbal_data = json.load(f)

        herbal_keys = set(herbal_data.keys())
        for class_name, kb_key in class_mapping.items():
            assert kb_key in herbal_keys, (
                f"Class '{class_name}' maps to '{kb_key}' which is not in the herbal KB"
            )
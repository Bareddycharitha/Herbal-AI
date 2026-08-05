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
"""Tests for LLM parser."""

import pytest
from src.llm_interface.llm_parser import parse_symptom_text, _mock_parse


def test_parse_symptom_text_structure():
    """Test that parsed symptoms have correct structure."""
    result = parse_symptom_text("I have chest pain and shortness of breath")
    
    assert "symptom_categories" in result
    assert "severity" in result
    assert "duration_days" in result
    assert "pattern" in result
    assert "red_flags" in result
    
    assert isinstance(result["symptom_categories"], list)
    assert isinstance(result["severity"], (int, float))
    assert 0 <= result["severity"] <= 10
    assert isinstance(result["duration_days"], int)
    assert isinstance(result["red_flags"], list)


def test_mock_parse_chest_pain():
    """Test mock parser detects chest pain."""
    result = _mock_parse("severe chest pain")
    
    assert "chest_pain" in result["symptom_categories"]
    assert result["severity"] > 5.0
    assert len(result["red_flags"]) > 0


def test_mock_parse_red_flags():
    """Test mock parser detects red flags."""
    result = _mock_parse("I passed out and can't breathe")
    
    assert "loss_of_consciousness" in result["red_flags"] or "difficulty_breathing" in result["red_flags"]


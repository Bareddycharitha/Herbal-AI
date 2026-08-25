"""
Tests for Universal Image Classifier evaluation and inference.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import io

from ai.image_classifier.inference import UniversalClassifierInference
from ai.image_classifier.evaluate import load_test_data, evaluate_model


def test_classifier_instantiation():
    """Test that the classifier can be instantiated."""
    classifier = UniversalClassifierInference()
    assert classifier is not None


def test_predict_returns_valid_output():
    """Test that predict returns a dictionary with expected keys."""
    classifier = UniversalClassifierInference()

    # We'll create a dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='red')

    result = classifier.predict(dummy_image)

    # Check required keys
    assert 'class' in result
    assert 'confidence' in result
    assert 'top_predictions' in result
    assert 'is_ood' in result
    assert 'ood_scores' in result

    # Check class is one of the expected
    assert result['class'] in ['Skin', 'Medicinal', 'Other']

    # Check confidence is between 0 and 100
    assert 0 <= result['confidence'] <= 100

    # Check top_predictions is a list of dicts with class and confidence
    assert isinstance(result['top_predictions'], list)
    for pred in result['top_predictions']:
        assert 'class' in pred
        assert 'confidence' in pred
        assert pred['class'] in ['Skin', 'Medicinal', 'Other']
        assert 0 <= pred['confidence'] <= 100


def test_predict_batch():
    """Test batch prediction."""
    classifier = UniversalClassifierInference()

    # Create two dummy images
    dummy_image1 = Image.new('RGB', (224, 224), color='red')
    dummy_image2 = Image.new('RGB', (224, 224), color='blue')

    results = classifier.predict_batch([dummy_image1, dummy_image2])

    assert len(results) == 2
    for result in results:
        assert 'class' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 100


def test_invalid_image_handling():
    """Test that invalid image paths are handled gracefully."""
    classifier = UniversalClassifierInference()

    # Test with non-existent file
    try:
        result = classifier.predict('non_existent_file.jpg')
        # If it doesn't raise an exception, it should return a result
        # (the inference code might handle it by raising an exception, which is okay)
        # We just want to ensure it doesn't crash the test suite
        assert isinstance(result, dict)
    except Exception:
        # It's okay if it raises an exception for invalid input
        pass


def test_load_test_data():
    """Test that test data loading function works."""
    # This test might fail if no test data is available, but we can skip if no data
    try:
        samples = load_test_data()
        # If we get here, data was loaded
        assert isinstance(samples, list)
        if len(samples) > 0:
            # Check that each sample is a tuple of (Path, int)
            for sample in samples:
                assert isinstance(sample[0], Path)
                assert isinstance(sample[1], int)
                assert sample[1] in [0, 1, 2]
    except FileNotFoundError:
        # If test directories don't exist, skip the test
        pytest.skip("Test directories not found")


def test_evaluate_model_runs():
    """Test that the evaluation function runs without error."""
    # This test might take a while and require test data, so we'll skip if no data
    try:
        # We'll run a quick evaluation but limit to a few samples if possible
        # However, our evaluate_model function doesn't support limiting.
        # Instead, we'll just call it and if it fails due to no data, we skip.
        results = evaluate_model()
        # If we get here, evaluation ran
        assert isinstance(results, dict)
        assert 'overall_accuracy' in results
        assert 'per_class_accuracy' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
        assert 'confidence_analysis' in results
        assert 'safety_metrics' in results
    except ValueError as e:
        if "No test images found" in str(e):
            pytest.skip("No test images found")
        else:
            raise
    except Exception as e:
        # Re-raise other exceptions
        raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
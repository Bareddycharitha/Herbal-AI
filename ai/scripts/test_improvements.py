"""
Test Script for Model Improvements

Verifies that all three models work correctly with:
- New inference classes
- OOD detection
- Calibration
- Ensemble support
"""

import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.image_classifier.inference import UniversalClassifierInference
from ai.training.inference import SkinDiseaseInference
from ai.herb.inference import HerbInference


def test_universal_classifier():
    """Test universal classifier inference."""
    print("\n" + "="*60)
    print("TESTING UNIVERSAL CLASSIFIER")
    print("="*60)

    try:
        classifier = UniversalClassifierInference()
        print("[OK] Universal classifier loaded successfully")

        # Test with a dummy image (create one if needed)
        from PIL import Image
        import numpy as np

        # Create a dummy test image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        result = classifier.predict(dummy_img)
        print(f"[OK] Prediction: {result['class']} ({result['confidence']:.2f}%)")
        print(f"  OOD: {result['is_ood']}")
        print(f"  OOD Scores: {result['ood_scores']}")
        print(f"  Top Predictions: {result['top_predictions']}")

        return True
    except Exception as e:
        print(f"[FAIL] Universal classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skin_disease():
    """Test skin disease inference."""
    print("\n" + "="*60)
    print("TESTING SKIN DISEASE CLASSIFIER")
    print("="*60)

    try:
        inference = SkinDiseaseInference()
        print("[OK] Skin disease inference loaded successfully")

        from PIL import Image
        import numpy as np

        dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        result = inference.predict(dummy_img)
        print(f"[OK] Prediction: {result['prediction']['disease']} ({result['prediction']['confidence']:.2f}%)")
        print(f"  Confidence Level: {result['prediction']['confidence_level']}")
        print(f"  Is Healthy: {result['is_healthy']}")
        print(f"  Is OOD: {result['is_ood']}")
        print(f"  OOD Scores: {result['ood_scores']}")
        print(f"  Binary Stage: {result.get('binary_stage', 'N/A')}")
        print(f"  Top Predictions: {result['top_predictions'][:3]}")

        return True
    except Exception as e:
        print(f"[FAIL] Skin disease test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_herb_identification():
    """Test herb identification inference (handles missing checkpoints gracefully)."""
    print("\n" + "="*60)
    print("TESTING HERB IDENTIFICATION")
    print("="*60)

    try:
        # Check if checkpoint exists
        from ai.herb.config import BEST_MODEL_PATH
        if not Path(BEST_MODEL_PATH).exists():
            print("[WARN] Herb checkpoint not found, testing with random weights")
            # Create inference with dummy checkpoint path to avoid loading
            inference = HerbInference(model_path=None)
        else:
            inference = HerbInference()
        print("[OK] Herb inference loaded successfully")

        from PIL import Image
        import numpy as np

        dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        result = inference.predict(dummy_img)
        print(f"[OK] Prediction: {result['herb']} ({result['confidence']:.2f}%)")
        print(f"  Is Confident: {result['is_confident']}")
        print(f"  Is Leaf: {result['is_leaf']}")
        print(f"  Leaf Confidence: {result['leaf_confidence']:.2f}%")
        print(f"  Is OOD: {result['is_ood']}")
        print(f"  OOD Scores: {result['ood_scores']}")
        print(f"  Top Predictions: {result['top_predictions'][:3]}")

        return True
    except Exception as e:
        print(f"[FAIL] Herb identification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_integration():
    """Test backend API integration."""
    print("\n" + "="*60)
    print("TESTING BACKEND INTEGRATION")
    print("="*60)

    try:
        # Test universal classifier import
        from backend.app.services.universal_classifier import classifier as universal_clf
        print("[OK] Backend universal classifier imported")

        # Test recommendation engine
        from ai.recommendation.recommendation_engine import get_recommendation
        print("[OK] Skin recommendation engine imported")

        # Test herb recommendation engine
        from ai.recommendation.herb_recommendation_engine import get_herb_recommendation
        print("[OK] Herb recommendation engine imported")

        return True
    except Exception as e:
        print(f"[FAIL] Backend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration_module():
    """Test calibration module."""
    print("\n" + "="*60)
    print("TESTING CALIBRATION MODULE")
    print("="*60)

    try:
        from ai.training.calibration import TemperatureScaling, calibrate_model, compute_ece
        print("[OK] Calibration module imported successfully")

        # Test temperature scaling
        model = torch.nn.Linear(10, 5)
        temp_scaler = TemperatureScaling()
        logits = torch.randn(32, 10)
        scaled = temp_scaler(logits)
        assert scaled.shape == logits.shape
        print("[OK] TemperatureScaling works")

        # Test ECE computation
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 5, (100,))
        ece = compute_ece(logits, labels)
        print(f"[OK] ECE computation works: {ece:.4f}")

        return True
    except Exception as e:
        print(f"[FAIL] Calibration module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ood_detection_module():
    """Test OOD detection module."""
    print("\n" + "="*60)
    print("TESTING OOD DETECTION MODULE")
    print("="*60)

    try:
        from ai.training.ood_detection import (
            EnergyBasedOOD, MSPBasedOOD, EntropyBasedOOD, CombinedOODDetector
        )
        print("[OK] OOD detection module imported successfully")

        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5)
        ).to('cpu')

        energy_ood = EnergyBasedOOD(model, 'cpu')
        msp_ood = MSPBasedOOD(model, 'cpu')
        entropy_ood = EntropyBasedOOD(model, 'cpu')
        combined = CombinedOODDetector(model, 'cpu')

        # Test with dummy data
        dummy_input = torch.randn(4, 10)
        energy_scores = energy_ood.compute_scores(dummy_input)
        msp_scores = msp_ood.compute_scores(dummy_input)
        entropy_scores = entropy_ood.compute_scores(dummy_input)
        combined_scores = combined.compute_combined_score(dummy_input)

        print(f"[OK] Energy scores: {energy_scores}")
        print(f"[OK] MSP scores: {msp_scores}")
        print(f"[OK] Entropy scores: {entropy_scores}")
        print(f"[OK] Combined scores: {combined_scores}")

        return True
    except Exception as e:
        print(f"[FAIL] OOD detection module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transforms():
    """Test updated transforms."""
    print("\n" + "="*60)
    print("TESTING TRANSFORMS")
    print("="*60)

    try:
        from ai.image_classifier.transforms import train_transform, test_transform, get_tta_transforms
        from PIL import Image
        import numpy as np

        # Test train transform
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        transformed = train_transform(img)
        assert transformed.shape == (3, 224, 224)
        print("[OK] Train transform works")

        # Test test transform
        transformed = test_transform(img)
        assert transformed.shape == (3, 224, 224)
        print("[OK] Test transform works")

        # Test TTA transforms
        tta_transforms = get_tta_transforms()
        assert len(tta_transforms) >= 5
        for t in tta_transforms:
            out = t(img)
            assert out.shape == (3, 224, 224)
        print(f"[OK] TTA transforms work ({len(tta_transforms)} transforms)")

        return True
    except Exception as e:
        print(f"[FAIL] Transforms test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("HERBAL-AI MODEL IMPROVEMENT TESTS")
    print("="*60)

    results = {
        "universal_classifier": test_universal_classifier(),
        "skin_disease": test_skin_disease(),
        "herb_identification": test_herb_identification(),
        "backend_integration": test_backend_integration(),
        "calibration_module": test_calibration_module(),
        "ood_detection_module": test_ood_detection_module(),
        "transforms": test_transforms(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
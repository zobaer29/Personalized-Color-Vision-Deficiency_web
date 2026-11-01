"""
Comprehensive Automated Master Test Suite for Personalized CVD Browser Assistant.
Directly implements Section 34 of MASTER TASK specifications:
- Health endpoint (healthy & fallback)
- Model loading & parameter count verification
- Checkpoint compatibility & metadata serialization
- Valid recolor (WebP, PNG, JPEG)
- Invalid Base64 (400)
- Invalid condition (422)
- Invalid severity (422)
- Unsupported format (422)
- Oversized image handling
- Preprocessing & padding to multiple of 8
- Output image dimensions matching input
- Severity 0 (original fast-path)
- Severity 1 (full neural correction)
- Cache hit & latency
- Cache condition isolation
- Cache severity isolation
- CPU inference
- Dataset pairing integrity
- Dataset leakage test (zero identity overlap)
- Missing checkpoint behavior (503)
- Screening / diagnostic estimation & disclaimer
"""

import base64
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app
from app.api import routes
from app.core.predictor import CVDRecolorPredictor
from app.core.model import create_model, count_trainable_parameters, ALL_MODEL_NAMES
from app.core.preprocessing import decode_base64_image, prepare_input_tensor
from app.core.postprocessing import unpad_and_clamp_tensor, encode_base64_image
from app.core.cache import recolor_cache
from app.core.screening import public_catalog, score
from train_or_eval import build_dataset_pairs, find_default_data_root


def create_test_image_base64(width: int = 64, height: int = 64, color: tuple = (180, 80, 50), format_type: str = "PNG") -> str:
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format_type)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format_type.lower()};base64,{encoded}"


class MasterTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create a synthetic verified Restormer_RDB checkpoint for testing plumbing
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_checkpoint_path = Path(cls.temp_dir.name) / "test_restormer.pt"

        model = create_model("Restormer_RDB")
        torch.save({
            "architecture": "Restormer_RDB",
            "model_version": "1.0.0",
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "best_validation_metric": 0.08,
            "val_psnr": 22.36,
            "val_ms_ssim": 0.8929,
            "training_config": {"epochs": 60, "batch_size": 16, "lr": 3e-4},
            "condition_mapping": {"Protanopia": 0, "Deuteranopia": 1, "Tritanopia": 2},
        }, cls.test_checkpoint_path)

        cls.predictor = CVDRecolorPredictor(weights_path=str(cls.test_checkpoint_path), device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        recolor_cache.clear()

    # 1. Health endpoint with untrained / missing weights
    def test_01_health_endpoint_degraded(self):
        with patch.object(routes.predictor, "weights_loaded", False):
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "checkpoint_required")
            self.assertFalse(data["trained_weights_loaded"])

    # 2. Health endpoint healthy
    def test_02_health_endpoint_healthy(self):
        with patch.object(routes.predictor, "weights_loaded", True):
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["trained_weights_loaded"])
            self.assertEqual(data["architecture"], "Restormer_RDB")
            self.assertTrue(data["cache_enabled"])

    # 3. Model loading and parameter counts match notebook
    def test_03_model_loading_and_parameter_counts(self):
        expected_params = {
            "Restormer_RDB": 461694,
            "Modified_Conditional_Restormer": 462888,
            "Base_Restormer_Style": 223870,
            "Restormer_CBAM": 225064,
            "Plain_Conditional_Residual_CNN": 337251,
            "Conditional_Residual_UNet": 1927939,
        }
        for name, expected in expected_params.items():
            net = create_model(name)
            p = count_trainable_parameters(net)
            self.assertEqual(p, expected, f"Param mismatch for {name}: {p} != {expected}")
            x = torch.rand(1, 3, 32, 32)
            c = torch.tensor([1])
            out = net(x, c)
            self.assertEqual(out.shape, (1, 3, 32, 32))

    # 4. Checkpoint compatibility & metadata serialization
    def test_04_checkpoint_compatibility(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            # Wrong architecture checkpoint
            mismatched_model = create_model("Plain_Conditional_Residual_CNN")
            torch.save({"architecture": "Plain_Conditional_Residual_CNN", "model_state_dict": mismatched_model.state_dict()}, tmp.name)
            pred = CVDRecolorPredictor(weights_path=None, model_name="Restormer_RDB", device="cpu")
            loaded = pred.load_weights(tmp.name)
            self.assertFalse(loaded)
            self.assertFalse(pred.weights_loaded)

    # 5. Valid recolor WebP
    def test_05_valid_recolor_webp(self):
        img_b64 = create_test_image_base64(48, 48)
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": img_b64,
                "condition_id": 1,
                "severity": 0.85,
                "return_format": "webp",
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertTrue(data["recolored_base64"].startswith("data:image/webp;base64,"))
            self.assertFalse(data["cached"])
            self.assertGreater(data["total_time_ms"], 0)

    # 6. Valid recolor JPEG
    def test_06_valid_recolor_jpeg(self):
        img_b64 = create_test_image_base64(32, 32)
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": img_b64,
                "condition_id": 0,
                "severity": 1.0,
                "return_format": "jpeg",
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["recolored_base64"].startswith("data:image/jpeg;base64,"))

    # 7. Valid recolor PNG
    def test_07_valid_recolor_png(self):
        img_b64 = create_test_image_base64(32, 32)
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": img_b64,
                "condition_id": 2,
                "severity": 0.5,
                "return_format": "png",
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["recolored_base64"].startswith("data:image/png;base64,"))

    # 8. Invalid Base64 error handling (400)
    def test_08_invalid_base64(self):
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": "not_valid_base64!!!",
                "condition_id": 1,
            })
            self.assertEqual(resp.status_code, 400)

    # 9. Invalid condition error handling (422)
    def test_09_invalid_condition(self):
        img_b64 = create_test_image_base64(16, 16)
        with patch.object(routes, "predictor", self.predictor):
            # Invalid ID
            resp = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 5})
            self.assertEqual(resp.status_code, 422)
            # Mismatched name and ID
            resp2 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 0, "condition": "Tritanopia"})
            self.assertEqual(resp2.status_code, 422)

    # 10. Invalid severity error handling (422)
    def test_10_invalid_severity(self):
        img_b64 = create_test_image_base64(16, 16)
        with patch.object(routes, "predictor", self.predictor):
            resp1 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "severity": -0.5})
            self.assertEqual(resp1.status_code, 422)
            resp2 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "severity": 1.2})
            self.assertEqual(resp2.status_code, 422)

    # 11. Unsupported format error handling (422)
    def test_11_unsupported_format(self):
        img_b64 = create_test_image_base64(16, 16)
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "return_format": "gif"})
            self.assertEqual(resp.status_code, 422)

    # 12. Oversized image handling / rejection
    def test_12_oversized_image_handling(self):
        # Extremely large dimension image is safely downscaled by decode_base64_image
        large_b64 = create_test_image_base64(2400, 1800)
        img, _ = decode_base64_image(large_b64)
        self.assertLessEqual(max(img.size), 2048)

    # 13. Preprocessing: spatial padding divisible by 8 and unpadding
    def test_13_preprocessing_padding_divisible_by_8(self):
        # Odd dimensions 75 x 91
        raw_img = Image.new("RGB", (75, 91), color=(100, 150, 200))
        tensor, original_size, pad_amounts = prepare_input_tensor(raw_img, device=torch.device("cpu"))
        self.assertEqual(original_size, (91, 75))
        h_pad, w_pad = tensor.shape[-2:]
        self.assertEqual(h_pad % 8, 0)
        self.assertEqual(w_pad % 8, 0)
        # Unpadding restores exact (91, 75)
        out_np = unpad_and_clamp_tensor(tensor, original_size)
        self.assertEqual(out_np.shape, (91, 75, 3))

    # 14. Output dimensions exactly match input dimensions
    def test_14_output_dimensions_exact(self):
        w, h = 57, 83
        img_b64 = create_test_image_base64(w, h)
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 1, "return_format": "png"})
            self.assertEqual(resp.status_code, 200)
            data_url = resp.json()["recolored_base64"]
            img, _ = decode_base64_image(data_url)
            self.assertEqual(img.size, (w, h))

    # 15. Severity 0 fast-path returns original image
    def test_15_severity_0_returns_original(self):
        img_b64 = create_test_image_base64(32, 32, color=(120, 140, 160))
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": img_b64,
                "condition_id": 1,
                "severity": 0.0,
                "return_format": "png",
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["inference_time_ms"], 0.0)
            # Decoded pixel values must match original
            orig_pil, _ = decode_base64_image(img_b64)
            out_pil, _ = decode_base64_image(data["recolored_base64"])
            np.testing.assert_array_equal(np.array(orig_pil), np.array(out_pil))

    # 16. Severity 1 applies full neural correction
    def test_16_severity_1_full_neural(self):
        img_b64 = create_test_image_base64(32, 32, color=(120, 140, 160))
        with patch.object(routes, "predictor", self.predictor):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": img_b64,
                "condition_id": 1,
                "severity": 1.0,
                "return_format": "png",
            })
            self.assertEqual(resp.status_code, 200)
            self.assertGreater(resp.json()["inference_time_ms"], 0.0)

    # 17. Cache hit verification and fast latency
    def test_17_cache_hit(self):
        img_b64 = create_test_image_base64(32, 32)
        with patch.object(routes, "predictor", self.predictor):
            # Request 1 (miss)
            r1 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 1, "severity": 0.8}).json()
            self.assertFalse(r1["cached"])
            # Request 2 (hit)
            r2 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 1, "severity": 0.8}).json()
            self.assertTrue(r2["cached"])
            self.assertEqual(r2["inference_time_ms"], 0.0)
            self.assertEqual(r1["recolored_base64"], r2["recolored_base64"])

    # 18. Cache condition isolation
    def test_18_cache_condition_isolation(self):
        img_b64 = create_test_image_base64(32, 32)
        with patch.object(routes, "predictor", self.predictor):
            r_protan = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 0, "severity": 1.0}).json()
            r_deutan = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 1, "severity": 1.0}).json()
            self.assertFalse(r_protan["cached"])
            self.assertFalse(r_deutan["cached"])
            self.assertEqual(r_protan["condition_id"], 0)
            self.assertEqual(r_deutan["condition_id"], 1)

    # 19. Cache severity isolation
    def test_19_cache_severity_isolation(self):
        img_b64 = create_test_image_base64(32, 32)
        with patch.object(routes, "predictor", self.predictor):
            r_sev1 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 1, "severity": 0.3}).json()
            r_sev2 = self.client.post("/api/v1/recolor", json={"image_base64": img_b64, "condition_id": 1, "severity": 0.9}).json()
            self.assertFalse(r_sev1["cached"])
            self.assertFalse(r_sev2["cached"])

    # 20. CPU inference verification
    def test_20_cpu_inference(self):
        self.assertEqual(str(self.predictor.device), "cpu")
        res = self.predictor.recolor(create_test_image_base64(16, 16), condition_id_or_name=1, severity=0.5)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["device"], "cpu")

    # 21. Dataset integrity & pairing (858 identities, 2,574 pairs)
    def test_21_dataset_pairing(self):
        data_root = find_default_data_root()
        orig_dir = data_root / "original"
        corrected_dir = data_root / "corrected"
        self.assertTrue(orig_dir.exists())
        self.assertTrue(corrected_dir.exists())

        orig_files = list(orig_dir.glob("*.*"))
        self.assertEqual(len(orig_files), 858, f"Expected 858 original images, found {len(orig_files)}")

        for cond in ["protanopia", "deuteranopia", "tritanopia"]:
            cond_files = list((corrected_dir / cond).glob("*.*"))
            self.assertEqual(len(cond_files), 858, f"Expected 858 {cond} images, found {len(cond_files)}")

    # 22. Dataset zero leakage between train, val, and test splits
    def test_22_dataset_leakage(self):
        data_root = find_default_data_root()
        train_pairs, val_pairs, test_pairs = build_dataset_pairs(data_root, seed=2026, train_frac=0.70, val_frac=0.15)

        train_ids = set(p["image_id"] for p in train_pairs)
        val_ids = set(p["image_id"] for p in val_pairs)
        test_ids = set(p["image_id"] for p in test_pairs)

        self.assertEqual(len(train_ids), 601)
        self.assertEqual(len(val_ids), 129)
        self.assertEqual(len(test_ids), 128)

        self.assertEqual(len(train_ids & val_ids), 0, "Data leakage between train and val!")
        self.assertEqual(len(train_ids & test_ids), 0, "Data leakage between train and test!")
        self.assertEqual(len(val_ids & test_ids), 0, "Data leakage between val and test!")

    # 23. Missing checkpoint behavior (503 when weights unloaded)
    def test_23_missing_checkpoint_behavior(self):
        with patch.object(routes.predictor, "weights_loaded", False):
            resp = self.client.post("/api/v1/recolor", json={
                "image_base64": create_test_image_base64(16, 16),
                "condition_id": 1,
            })
            self.assertEqual(resp.status_code, 503)

    # 24. Screening / profile estimation workflow
    def test_24_screening_workflow(self):
        # GET /api/v1/plates
        resp = self.client.get("/api/v1/plates")
        self.assertEqual(resp.status_code, 200)
        catalog_data = resp.json()
        self.assertTrue(catalog_data["available"])
        self.assertFalse(catalog_data["clinical_diagnosis"])
        self.assertGreater(len(catalog_data["plates"]), 0)

        # POST /api/v1/diagnose with normal answers
        plates = catalog_data["plates"]
        answers = [{"plate_id": p["id"], "user_answer": p["options"][0]} for p in plates]
        diag_resp = self.client.post("/api/v1/diagnose", json={"answers": answers})
        self.assertEqual(diag_resp.status_code, 200)
        result = diag_resp.json()
        self.assertFalse(result["diagnosed"])
        self.assertIn("message", result)


if __name__ == "__main__":
    unittest.main()

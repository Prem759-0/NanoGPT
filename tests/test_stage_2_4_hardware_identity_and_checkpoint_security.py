"""
tests/test_stage_2_4_hardware_identity_and_checkpoint_security.py
Moon AI — Phase 2.4.3 Hardware Identity & Checkpoint Security Verification Suite
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Verification for:
1. Physical NVIDIA Tesla T4 identity enforcement (name, compute capability 7.5, minimum VRAM).
2. Non-T4 rejection (A100, L4, V100, CPU, wrong compute capability, insufficient VRAM).
3. Secure checkpoint deserialization (weights_only=True, blocked pickle code execution).
4. Structural checkpoint schema validation & failure modes (missing keys, non-tensors, invalid geometry).
5. Canonical benchmark report preservation and unique report isolation.
6. Anti-promotion enforcement (CI fixtures and unpromoted pilot checkpoints strictly rejected).
"""

import os
import sys
import re
import json
import uuid
import pickle
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.validate_t4_empirical import (
    check_t4_hardware_identity,
    save_benchmark_report,
    verify_checkpoint_roundtrip_isolated,
    run_empirical_t4_validation
)
from scripts.benchmark_teacher_t4 import run_benchmark as run_teacher_t4_benchmark
from webapp.model_service import (
    validate_checkpoint_structure,
    is_verified_genuine_checkpoint,
    is_checkpoint_fixture,
    SafeUnpickler
)
from scripts.promote_best_model import evaluate_and_promote
from model import GPTConfig, GPT


class TestStage24HardwareIdentityAndCheckpointSecurity(unittest.TestCase):
    """Test suite certifying Phase 2.4.3 hardware identity and deserialization security."""

    def test_01_tesla_t4_accepted(self):
        """Authentic NVIDIA Tesla T4 configurations are positively accepted."""
        valid_configs = [
            ("Tesla T4", "7.5", 15109.0),
            ("NVIDIA Tesla T4", "7.5", 15360.0),
            ("NVIDIA T4", "7.5", 14500.0),
            ("Tesla T4 16GB", "7.5", 15000.0),
        ]
        for name, cap, vram in valid_configs:
            is_match, reason, details = check_t4_hardware_identity(
                device_name=name,
                compute_capability=cap,
                total_vram_mb=vram,
                min_vram_mb=14000.0
            )
            self.assertTrue(is_match, f"Authentic T4 config failed validation: {name}, {cap}, {vram} (reason: {reason})")
            self.assertEqual(reason, "NVIDIA_T4_AUTHENTICATED")

    def test_02_a100_rejected_as_t4(self):
        """NVIDIA A100 is strictly rejected as non-T4 hardware."""
        a100_variants = [
            ("NVIDIA A100-SXM4-40GB", "8.0", 40960.0),
            ("NVIDIA A100-PCIE-80GB", "8.0", 81920.0),
            ("A100", "8.0", 40960.0),
            ("NVIDIA A100 (T4 emulation attempted)", "8.0", 40960.0),
        ]
        for name, cap, vram in a100_variants:
            is_match, reason, _ = check_t4_hardware_identity(name, cap, vram)
            self.assertFalse(is_match, f"A100 was falsely accepted as T4: {name}")
            self.assertIn("Hardware mismatch", reason)

    def test_03_l4_rejected_as_t4(self):
        """NVIDIA L4 (Ada Lovelace) is strictly rejected as non-T4 hardware."""
        l4_variants = [
            ("NVIDIA L4", "8.9", 24576.0),
            ("NVIDIA L4 24GB", "8.9", 24000.0),
        ]
        for name, cap, vram in l4_variants:
            is_match, reason, _ = check_t4_hardware_identity(name, cap, vram)
            self.assertFalse(is_match, f"L4 was falsely accepted as T4: {name}")
            self.assertIn("Hardware mismatch", reason)

    def test_04_cpu_and_cuda_unavailable_rejected(self):
        """CPU devices and CUDA-unavailable environments are safely rejected."""
        is_match, reason, _ = check_t4_hardware_identity("CPU", "0.0", 0.0)
        self.assertFalse(is_match)
        self.assertIn("Hardware mismatch", reason)

        # On this CPU-only test host, run_empirical_t4_validation must fail closed with CUDA_UNAVAILABLE
        if not torch.cuda.is_available():
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_empirical_t4_validation(report_dir=tmp_dir)
                self.assertEqual(res["status"], "CUDA_UNAVAILABLE")
                self.assertFalse(res["empirical_benchmark_verified"])

    def test_05_wrong_compute_capability_rejected(self):
        """Wrong compute capability (anything other than exactly 7.5) is strictly rejected."""
        wrong_caps = ["7.0", "8.0", "8.6", "8.9", "9.0", "6.1"]
        for cap in wrong_caps:
            is_match, reason, _ = check_t4_hardware_identity("Tesla T4", cap, 15360.0)
            self.assertFalse(is_match, f"Wrong compute capability {cap} was falsely accepted")
            self.assertIn("compute capability", reason)

    def test_06_insufficient_vram_rejected(self):
        """Insufficient physical VRAM (< 14,000 MB) is strictly rejected."""
        under_vrams = [8192.0, 10240.0, 12000.0, 13999.0]
        for vram in under_vrams:
            is_match, reason, _ = check_t4_hardware_identity("Tesla T4", "7.5", vram, min_vram_mb=14000.0)
            self.assertFalse(is_match, f"Insufficient VRAM {vram} MB was falsely accepted")
            self.assertIn("insufficient physical VRAM", reason)

    def test_07_invalid_checkpoint_structure_rejected(self):
        """ModelService strictly rejects invalid, corrupted, non-tensor, or malformed checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Non-existent path
            self.assertFalse(validate_checkpoint_structure(os.path.join(tmp_dir, "ghost.pt")))

            # 2. Corrupted raw bytes
            corrupt = os.path.join(tmp_dir, "corrupt.pt")
            with open(corrupt, "wb") as f:
                f.write(b"CORRUPTED_RAW_NON_PICKLE_STREAM")
            self.assertFalse(validate_checkpoint_structure(corrupt))

            # 3. Non-dict root
            non_dict = os.path.join(tmp_dir, "non_dict.pt")
            torch.save(["not", "a", "dict"], non_dict)
            self.assertFalse(validate_checkpoint_structure(non_dict))

            # 4. Missing model
            no_model = os.path.join(tmp_dir, "no_model.pt")
            torch.save({"model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512}}, no_model)
            self.assertFalse(validate_checkpoint_structure(no_model))

            # 5. Non-tensor in model state_dict
            bad_tensor = os.path.join(tmp_dir, "bad_tensor.pt")
            torch.save({
                "model": {"transformer.wte.weight": "NOT_A_TENSOR"},
                "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512}
            }, bad_tensor)
            self.assertFalse(validate_checkpoint_structure(bad_tensor))

            # 6. Missing architectural parameter
            missing_arg = os.path.join(tmp_dir, "missing_arg.pt")
            torch.save({
                "model": {"weight": torch.randn(4, 4)},
                "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512}  # missing block_size
            }, missing_arg)
            self.assertFalse(validate_checkpoint_structure(missing_arg))

            # 7. Invalid architecture: n_embd not divisible by n_head
            indivisible = os.path.join(tmp_dir, "indivisible.pt")
            torch.save({
                "model": {"weight": torch.randn(4, 4)},
                "model_args": {"n_layer": 8, "n_head": 7, "n_embd": 512, "vocab_size": 512, "block_size": 512}  # 512 % 7 != 0
            }, indivisible)
            self.assertFalse(validate_checkpoint_structure(indivisible))

            # 8. Unsupported schema version
            bad_schema = os.path.join(tmp_dir, "bad_schema.pt")
            torch.save({
                "schema_version": "999.0_UNSUPPORTED",
                "model": {"weight": torch.randn(4, 4)},
                "model_args": {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 512, "block_size": 512}
            }, bad_schema)
            self.assertFalse(validate_checkpoint_structure(bad_schema))

    def test_08_untrusted_pickle_exploit_blocked_without_execution(self):
        """Malicious pickle exploit payloads are blocked by weights_only=True without executing code."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sentinel_file = os.path.join(tmp_dir, "EXPLOIT_EXECUTED.txt")
            exploit_ckpt = os.path.join(tmp_dir, "malicious_checkpoint.pt")

            # Craft a pickle payload that attempts to execute arbitrary code
            class MaliciousPayload:
                def __reduce__(self):
                    # Attempts to write to sentinel_file if deserialized with weights_only=False
                    return (os.system, (f'echo PWNED > "{sentinel_file}"',))

            with open(exploit_ckpt, "wb") as f:
                pickle.dump(MaliciousPayload(), f)

            # Deserialization MUST fail closed and must NOT execute the os.system command
            res = validate_checkpoint_structure(exploit_ckpt)
            self.assertFalse(res.is_valid, "Malicious pickle exploit was not rejected!")
            self.assertFalse(os.path.exists(sentinel_file), "CRITICAL SECURITY VULNERABILITY: Arbitrary code execution occurred during checkpoint load!")

            # Verify SafeUnpickler also rejects arbitrary objects in meta.pkl
            exploit_meta = os.path.join(tmp_dir, "malicious_meta.pkl")
            with open(exploit_meta, "wb") as f:
                pickle.dump({"stoi": {}, "itos": {}, "payload": MaliciousPayload()}, f)

            with open(exploit_meta, "rb") as f:
                with self.assertRaises((pickle.UnpicklingError, Exception)):
                    SafeUnpickler(f).load()

            self.assertFalse(os.path.exists(sentinel_file), "CRITICAL SECURITY VULNERABILITY: Code execution occurred via SafeUnpickler!")

    def test_09_existing_checkpoint_and_benchmark_report_preserved(self):
        """Canonical benchmark reports and genuine checkpoints cannot be overwritten or deleted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = os.path.join(tmp_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            canonical_path = os.path.join(report_dir, "t4_empirical_benchmark.json")

            # 1. Establish an authentic, verified canonical report
            canonical_data = {
                "status": "T4_EMPIRICAL_BENCHMARK_VERIFIED",
                "run_id": "verified_t4_golden_sentinel",
                "device_name": "Tesla T4",
                "compute_capability": "7.5",
                "total_vram_mb": 15109.0,
                "empirical_benchmark_verified": True
            }
            with open(canonical_path, "w", encoding="utf-8") as f:
                json.dump(canonical_data, f, indent=2)

            # 2. Attempt saving an unverified hardware-mismatch report: canonical report MUST be preserved
            mismatch_summary = {
                "status": "HARDWARE_MISMATCH",
                "run_id": "failed_run_a100",
                "device_name": "NVIDIA A100",
                "empirical_benchmark_verified": False
            }
            exported_path = save_benchmark_report(mismatch_summary, report_dir=report_dir, allow_overwrite_verified=False)
            self.assertNotEqual(exported_path, canonical_path, "Failed run must not overwrite canonical verified report!")

            # Canonical report remains 100% intact
            with open(canonical_path, "r", encoding="utf-8") as f:
                current_canonical = json.load(f)
            self.assertEqual(current_canonical["run_id"], "verified_t4_golden_sentinel")
            self.assertEqual(current_canonical["status"], "T4_EMPIRICAL_BENCHMARK_VERIFIED")

            # 3. Attempt saving another verified run without explicit overwrite permission: canonical report preserved
            new_verified_summary = {
                "status": "T4_EMPIRICAL_BENCHMARK_VERIFIED",
                "run_id": "second_t4_run",
                "device_name": "Tesla T4",
                "empirical_benchmark_verified": True
            }
            new_exported = save_benchmark_report(new_verified_summary, report_dir=report_dir, allow_overwrite_verified=False)
            self.assertNotEqual(new_exported, canonical_path)
            self.assertTrue(os.path.isfile(new_exported))

            # Canonical report STILL intact
            with open(canonical_path, "r", encoding="utf-8") as f:
                current_canonical = json.load(f)
            self.assertEqual(current_canonical["run_id"], "verified_t4_golden_sentinel")

    def test_10_ci_fixture_and_pilot_remain_unpromotable(self):
        """Promotion gate rejects all CI fixtures, raw pilot checkpoints, and unverified training modes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")

            disallowed_statuses = [
                "CI_TEST_FIXTURE",
                "PILOT_TRAINED",
                "PILOT_TRAINED_BEST",
                "ENGINEERING_SMOKE_TEST_ONLY",
                "DATA_INSUFFICIENT"
            ]

            for dis_status in disallowed_statuses:
                cand_path = os.path.join(tmp_dir, f"candidate_{dis_status}.json")
                cand_data = {
                    "model_name": f"candidate_{dis_status}",
                    "status": dis_status,
                    "training_mode": dis_status,
                    "is_production_trained": False,
                    "is_production_eligible": False,
                    "promotable": False,
                    "composite_quality_score": 99.0,
                    "non_empty_rate_pct": 100.0,
                    "code_syntax_pct": 100.0
                }
                with open(cand_path, "w", encoding="utf-8") as f:
                    json.dump(cand_data, f)

                promoted = evaluate_and_promote(cand_path, registry_path=reg_path, base_dir=tmp_dir)
                self.assertFalse(promoted, f"Model with status '{dis_status}' must be rejected by promotion gate!")

    def test_11_strict_t4_device_name_matching_and_spoofing_rejected(self):
        """Strict regex matching rejects spoofed/non-T4 device names and accepts authentic T4s."""
        spoofed_or_non_t4_names = [
            "not4",
            "fast4",
            "RTX 4090 T4 compatible",
            "Virtual T4 emulator",
            "Generic T4 emulator",
            "Tesla T4 fake",
            "T4 mock device",
            "GeForce RTX 3080 T4 edition",
            "Tesla T40",
            "T400",
            "NVIDIA T4 proxy",
            "T4-compat-shim",
            "Intel T4",
            "CPU"
        ]
        for name in spoofed_or_non_t4_names:
            is_match, reason, _ = check_t4_hardware_identity(
                device_name=name,
                compute_capability="7.5",
                total_vram_mb=15109.0,
                min_vram_mb=14000.0
            )
            self.assertFalse(is_match, f"Spoofed/invalid device name '{name}' was falsely accepted as T4!")
            self.assertIn("Hardware mismatch", reason)

        authentic_t4_names = [
            "Tesla T4",
            "NVIDIA Tesla T4",
            "NVIDIA T4",
            "Tesla T4 16GB",
            "NVIDIA T4 16GB",
            "T4",
            "Tesla T4 PCIe",
            "NVIDIA Tesla T4 PCIe"
        ]
        for name in authentic_t4_names:
            is_match, reason, _ = check_t4_hardware_identity(
                device_name=name,
                compute_capability="7.5",
                total_vram_mb=15109.0,
                min_vram_mb=14000.0
            )
            self.assertTrue(is_match, f"Authentic T4 name '{name}' was unexpectedly rejected: {reason}")
            self.assertEqual(reason, "NVIDIA_T4_AUTHENTICATED")

    def test_12_compatibility_estimate_metadata_integrity(self):
        """Analytical compatibility estimates contain explicit metadata and no fabricated VRAM on CPU."""
        report_path = os.path.join(PROJECT_ROOT, "data", "reports", "t4_compatibility_benchmark.json")
        self.assertTrue(os.path.isfile(report_path), f"Expected report at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            original_content = f.read()
            data = json.loads(original_content)

        self.assertEqual(data.get("benchmark_type"), "COMPATIBILITY_ESTIMATE")
        self.assertFalse(data.get("is_empirical"))
        self.assertFalse(data.get("is_cuda"))
        self.assertEqual(data.get("device_name"), "CPU")
        self.assertIsNone(data.get("total_vram_mb"), "CPU compatibility estimate must not fabricate VRAM measurements")
        self.assertEqual(data.get("assumed_planning_vram_mb"), 15360.0)
        self.assertTrue(data.get("analytical_t4_compatible"))
        self.assertFalse(data.get("empirical_t4_verified"), "CPU report must not claim empirical T4 verification")
        self.assertTrue(data.get("t4_compatible"))

        try:
            # Verify executing run_teacher_t4_benchmark produces the same non-empirical schema
            fresh_report = run_teacher_t4_benchmark(context_length=1024, repetitions=1)
            self.assertEqual(fresh_report.get("benchmark_type"), "COMPATIBILITY_ESTIMATE")
            self.assertFalse(fresh_report.get("is_empirical"))
            if not fresh_report.get("is_cuda"):
                self.assertIsNone(fresh_report.get("total_vram_mb"))
        finally:
            # Restore original canonical report so tests leave working tree clean
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(original_content)


if __name__ == "__main__":
    unittest.main()


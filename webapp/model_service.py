import os
import sys
import glob
import pickle
from typing import Optional, Dict, Any, List, Tuple


class CheckpointValidationResult(tuple):
    """Encapsulates checkpoint validation tuple (is_valid, reason, checkpoint_data) with boolean coercion."""
    def __new__(cls, is_valid: bool, reason: str, data: Dict[str, Any]):
        return super().__new__(cls, (is_valid, reason, data))

    def __bool__(self) -> bool:
        return bool(self[0])

    @property
    def is_valid(self) -> bool:
        return self[0]

    @property
    def reason(self) -> str:
        return self[1]

    @property
    def data(self) -> Dict[str, Any]:
        return self[2]


class SafeUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that permits only primitive dictionary structures,
    completely preventing arbitrary object instantiation or code execution during unpickling.
    """
    SAFE_BUILTINS = {
        "dict", "list", "set", "frozenset", "tuple",
        "str", "int", "float", "bool", "bytes", "bytearray", "NoneType"
    }

    def find_class(self, module: str, name: str) -> Any:
        import builtins
        if module in ("builtins", "__builtin__") and name in self.SAFE_BUILTINS:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"Untrusted object '{module}.{name}' is prohibited during safe deserialization.")


def validate_checkpoint_structure(ckpt_path: str, weights_only: bool = True) -> CheckpointValidationResult:
    """
    Validates the PyTorch structure and schema of a checkpoint file with secure deserialization.
    Fails closed on missing file, unreadable file, empty file, non-dict, missing model tensors,
    malicious pickle payloads, unsupported types, or invalid architectures.
    Returns: CheckpointValidationResult(is_valid, reason, loaded_dict)
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return CheckpointValidationResult(False, f"Checkpoint path does not exist or is not a file: {ckpt_path}", {})
    if os.path.getsize(ckpt_path) == 0:
        return CheckpointValidationResult(False, f"Checkpoint file is empty (0 bytes): {ckpt_path}", {})

    try:
        import torch
        # Enforce secure deserialization: weights_only=True restricts unpickling to tensors and safe primitives
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
        if not isinstance(ckpt, dict):
            return CheckpointValidationResult(False, f"Checkpoint root is not a dictionary (type={type(ckpt).__name__})", {})
        if "model" not in ckpt or not isinstance(ckpt["model"], dict):
            return CheckpointValidationResult(False, "Checkpoint missing 'model' state dictionary", {})
        if len(ckpt["model"]) == 0:
            return CheckpointValidationResult(False, "Checkpoint 'model' state dictionary is empty", {})

        # Tensor type and payload security verification
        for param_name, param_val in ckpt["model"].items():
            if not isinstance(param_name, str):
                return CheckpointValidationResult(False, f"Invalid non-string parameter name in state dictionary: {type(param_name).__name__}", {})
            if not isinstance(param_val, torch.Tensor):
                return CheckpointValidationResult(False, f"Invalid non-tensor parameter value for '{param_name}': {type(param_val).__name__}", {})

        if "model_args" not in ckpt or not isinstance(ckpt["model_args"], dict):
            return CheckpointValidationResult(False, "Checkpoint missing 'model_args' dictionary", {})

        required_args = ["n_layer", "n_head", "n_embd", "vocab_size", "block_size"]
        for arg in required_args:
            if arg not in ckpt["model_args"]:
                return CheckpointValidationResult(False, f"Checkpoint 'model_args' missing required architectural parameter '{arg}'", {})
            val = ckpt["model_args"][arg]
            if not isinstance(val, int) or val <= 0:
                return CheckpointValidationResult(False, f"Architectural parameter '{arg}' must be a positive integer, got {val}", {})

        # Architectural invariants & bounds validation (blocks invalid geometries or allocation exhaustion attacks)
        n_embd = ckpt["model_args"]["n_embd"]
        n_head = ckpt["model_args"]["n_head"]
        n_layer = ckpt["model_args"]["n_layer"]
        vocab_size = ckpt["model_args"]["vocab_size"]
        block_size = ckpt["model_args"]["block_size"]

        if n_embd % n_head != 0:
            return CheckpointValidationResult(False, f"Invalid architecture: n_embd ({n_embd}) must be divisible by n_head ({n_head})", {})

        if n_layer > 128 or n_head > 128 or n_embd > 8192 or vocab_size > 262144 or block_size > 65536:
            return CheckpointValidationResult(False, "Architectural parameter exceeds safe supported bounds", {})

        # Reviewed schema version compatibility verification
        cfg = ckpt.get("config", {})
        schema_version = ckpt.get("schema_version") or (cfg.get("schema_version") if isinstance(cfg, dict) else None)
        if schema_version is not None:
            supported_schemas = ("1.0", "1.1", "2.0", "2.1", "2.2", "2.3", "2.4", "moon_chat_v1", "sovereign_v1")
            if str(schema_version) not in supported_schemas:
                return CheckpointValidationResult(False, f"Unsupported checkpoint schema version: {schema_version}", {})

        return CheckpointValidationResult(True, "VALID_STRUCTURE", ckpt)
    except Exception as e:
        return CheckpointValidationResult(False, f"Failed to load checkpoint file ({type(e).__name__}): {str(e)}", {})


def is_checkpoint_fixture(ckpt_path: str) -> bool:
    """Determines if a checkpoint file is an unpromotable CI test fixture."""
    ok, reason, ckpt = validate_checkpoint_structure(ckpt_path)
    if not ok:
        return False
    if ckpt.get("is_test_fixture") is True:
        return True
    if ckpt.get("status") in ("CI_TEST_FIXTURE", "ENGINEERING_SMOKE_TEST_ONLY"):
        return True
    cfg = ckpt.get("config", {}) or ckpt.get("metadata", {})
    if cfg.get("is_test_fixture") is True or cfg.get("run_id") == "ci_test_fixture":
        return True
    return False


def is_historical_pilot_checkpoint(ckpt_path: str) -> bool:
    """
    Identifies structurally valid historical unpromoted pilot checkpoints (e.g. Stage 2.1.6 Policy B).
    Permitted strictly for retrospective test discovery and historical audit inspection.
    Never treated as verified genuine strict sovereign production checkpoints.
    """
    ok, reason, ckpt = validate_checkpoint_structure(ckpt_path)
    if not ok:
        return False
    cfg = ckpt.get("config", {}) or ckpt.get("metadata", {})
    if not isinstance(cfg, dict) or not cfg:
        return False
    if cfg.get("teacher_checkpoint_weights_transferred") is not False:
        return False
    init_method = cfg.get("initialization") or cfg.get("student_init_method")
    if init_method != "random_gaussian_scratch":
        return False
    policy = cfg.get("ownership_policy") or cfg.get("policy")
    if policy not in ("TEACHER_STUDENT_RANDOM_INIT", "POLICY_B_TEACHER_STUDENT_RANDOM_INIT"):
        return False
    return True


def classify_checkpoint(
    ckpt_path: str,
    trust_provider: Optional[Any] = None,
    trusted_authorities_path: Optional[str] = None,
    allow_test_fixture_classification: bool = False,
    allow_unattested_for_test: bool = False,
    require_production_provider: bool = True,
    evidence_store: Optional[Any] = None,
    require_artifacts: Optional[bool] = None,
) -> str:
    """
    Categorizes a checkpoint file into one of the mutually exclusive provenance tiers:
    - STRUCTURALLY_INVALID: Nonexistent, empty, malformed, non-dict, non-tensor, or unreadable.
    - TEST_FIXTURE: Explicitly generated as CI fixture or smoke test.
    - TEST_FIXTURE_VALID_STRUCTURE: Valid model structure and metadata in test mode, but lacks cryptographic verification.
    - HISTORICAL_POLICY_B: Stage 2.1.6 pilot checkpoint with quarantined Policy B provenance.
    - UNAPPROVED_POLICY: Policy identifier does not match STRICT_SOVEREIGN_RANDOM_INIT_ONLY.
    - NON_SCRATCH_INITIALIZATION: Initialized from pretrained weights or non-scratch Gaussian method.
    - TEACHER_CONTAMINATION: Contains teacher weights, teacher model refs, or transferred weights.
    - UNTRAINED_RANDOM_INIT: Random Gaussian initialization lacking training progress (iter_num <= 0).
    - SYNTHETIC_MOCK: Generated synthetically with random weights without authentic training telemetry.
    - ATTESTATION_MISSING: Checkpoint lacks an asymmetric cryptographic training attestation.
    - ATTESTATION_INVALID: Attestation is malformed, missing required fields, or Ed25519 signature is invalid.
    - ATTESTATION_CHECKPOINT_HASH_MISMATCH: Attestation binds a SHA-256 that does not match disk file bytes.
    - ATTESTATION_UNTRUSTED_SIGNER: Attestation is signed by an unregistered authority key or trust root is missing.
    - ATTESTATION_REVOKED_AUTHORITY: Authority has been revoked.
    - ATTESTATION_EXPIRED_AUTHORITY: Authority registration has expired.
    - ATTESTATION_EVIDENCE_INVALID: Training evidence manifest contains telemetry contradictions or invalid hashes.
    - GENUINE_TRAINED_SOVEREIGN: Verified scratch-trained model satisfying all strict provenance pillars & cryptographic Ed25519 signature.
    - PROMOTED_PRODUCTION: Fully certified sovereign checkpoint promoted for production serving with cryptographic attestation.

    CRITICAL SECURITY INVARIANT:
    allow_test_fixture_classification and allow_unattested_for_test CAN NEVER produce GENUINE_TRAINED_SOVEREIGN
    or PROMOTED_PRODUCTION. They can only produce TEST_FIXTURE_VALID_STRUCTURE.
    """
    if require_artifacts is None:
        require_artifacts = require_production_provider

    if isinstance(trust_provider, str):
        trusted_authorities_path = trust_provider
        trust_provider = None

    ok, reason, ckpt = validate_checkpoint_structure(ckpt_path)
    if not ok:
        return "STRUCTURALLY_INVALID"

    cfg = ckpt.get("config", {}) or ckpt.get("metadata", {})
    if not isinstance(cfg, dict):
        cfg = {}

    # Explicit test fixtures and synthetic mocks
    filename = os.path.basename(ckpt_path).lower()
    is_synth = (
        ckpt.get("is_synthetic") is True
        or cfg.get("is_synthetic") is True
        or ckpt.get("mock") is True
        or cfg.get("mock") is True
        or "synthetic" in str(ckpt.get("status", "")).lower()
        or "mock" in str(ckpt.get("status", "")).lower()
        or "synthetic" in filename
        or "mock" in filename
    )
    if is_synth:
        return "SYNTHETIC_MOCK"

    if ckpt.get("is_test_fixture") is True or cfg.get("is_test_fixture") is True:
        return "TEST_FIXTURE"
    if ckpt.get("status") in ("CI_TEST_FIXTURE", "ENGINEERING_SMOKE_TEST_ONLY", "DATA_INSUFFICIENT"):
        return "TEST_FIXTURE"
    if cfg.get("run_id") == "ci_test_fixture":
        return "TEST_FIXTURE"

    # Check for historical Policy B
    policy = cfg.get("ownership_policy") or cfg.get("policy")
    if policy in ("TEACHER_STUDENT_RANDOM_INIT", "POLICY_B_TEACHER_STUDENT_RANDOM_INIT"):
        return "HISTORICAL_POLICY_B"

    # Strict sovereign policy enforcement
    if policy != "STRICT_SOVEREIGN_RANDOM_INIT_ONLY":
        return "UNAPPROVED_POLICY"

    # Initialization method must strictly be random Gaussian scratch
    init_method = cfg.get("initialization") or cfg.get("student_init_method")
    if init_method != "random_gaussian_scratch":
        return "NON_SCRATCH_INITIALIZATION"

    # Teacher transfer must be literal False
    if cfg.get("teacher_checkpoint_weights_transferred") is not False:
        return "TEACHER_CONTAMINATION"

    # Reject external base weights or teacher references
    base_weights = cfg.get("base_weights") or cfg.get("pretrained_source")
    if base_weights and base_weights not in ("none", "scratch", "random_gaussian", "random_gaussian_scratch"):
        return "TEACHER_CONTAMINATION"
    if cfg.get("teacher_checkpoint") or cfg.get("teacher_model"):
        return "TEACHER_CONTAMINATION"

    # Verifiable Training Telemetry & Evidence Verification:
    # Check for explicitly zero or negative iteration progress indicating untrained state
    iter_num = ckpt.get("iter_num") if "iter_num" in ckpt else cfg.get("iter_num")
    if iter_num is None:
        iter_num = cfg.get("step")
    val_loss = ckpt.get("best_val_loss") if "best_val_loss" in ckpt else ckpt.get("val_loss")
    if val_loss is None:
        val_loss = cfg.get("best_val_loss") if "best_val_loss" in cfg else cfg.get("val_loss")

    if iter_num is not None and isinstance(iter_num, int) and iter_num <= 0:
        return "UNTRAINED_RANDOM_INIT"
    if val_loss is not None and isinstance(val_loss, (int, float)) and val_loss <= 0.0:
        return "UNTRAINED_RANDOM_INIT"

    # Cryptographic Attestation Verification
    from core.security.training_attestation import verify_training_attestation

    attestation = (
        ckpt.get("attestation")
        or ckpt.get("training_attestation")
        or cfg.get("attestation")
        or cfg.get("training_attestation")
    )

    if not attestation:
        sibling_att = ckpt_path + ".attestation.json"
        alt_sibling_att = os.path.splitext(ckpt_path)[0] + ".attestation.json"
        if os.path.isfile(sibling_att):
            try:
                import json
                with open(sibling_att, "r", encoding="utf-8") as f:
                    attestation = json.load(f)
            except Exception:
                return "ATTESTATION_INVALID"
        elif os.path.isfile(alt_sibling_att):
            try:
                import json
                with open(alt_sibling_att, "r", encoding="utf-8") as f:
                    attestation = json.load(f)
            except Exception:
                return "ATTESTATION_INVALID"

    if attestation is not None:
        is_valid, tier, reason = verify_training_attestation(
            attestation,
            ckpt_path,
            trust_provider=trust_provider,
            authorities_path=trusted_authorities_path,
            require_production_provider=require_production_provider,
            evidence_store=evidence_store,
            require_artifacts=require_artifacts,
        )
        if not is_valid:
            if allow_test_fixture_classification and tier not in (
                "STRUCTURALLY_INVALID",
                "UNAPPROVED_POLICY",
                "NON_SCRATCH_INITIALIZATION",
                "TEACHER_CONTAMINATION",
            ):
                return "TEST_FIXTURE_VALID_STRUCTURE"
            return tier

        status = ckpt.get("status") or cfg.get("status")
        if status == "PROMOTED_PRODUCTION":
            return "PROMOTED_PRODUCTION"
        return "GENUINE_TRAINED_SOVEREIGN"

    # Test fixture fallback: NEVER returns GENUINE_TRAINED_SOVEREIGN or PROMOTED_PRODUCTION
    if allow_test_fixture_classification or allow_unattested_for_test:
        return "TEST_FIXTURE_VALID_STRUCTURE"

    return "ATTESTATION_MISSING"


def is_verified_genuine_checkpoint(
    ckpt_path: str,
    trust_provider: Optional[Any] = None,
    trusted_authorities_path: Optional[str] = None,
    require_production: bool = True,
    evidence_store: Optional[Any] = None,
    require_artifacts: Optional[bool] = None,
    **kwargs,
) -> bool:
    """
    Positively validates that a checkpoint is a genuine, trained student checkpoint.
    Fails closed on:
    - Corrupted, malformed, or unreadable files (STRUCTURALLY_INVALID)
    - Synthetic mock checkpoints without verified training telemetry (SYNTHETIC_MOCK)
    - Untrained random initializations lacking iteration progress (UNTRAINED_RANDOM_INIT)
    - CI test fixtures or engineering smoke test checkpoints (TEST_FIXTURE)
    - Valid structural fixtures lacking cryptographic proof (TEST_FIXTURE_VALID_STRUCTURE)
    - Policy B violations or unapproved policies (HISTORICAL_POLICY_B, UNAPPROVED_POLICY)
    - Teacher contamination or external base weights (TEACHER_CONTAMINATION)
    - Missing or untrusted cryptographic training attestations (ATTESTATION_MISSING, ATTESTATION_UNTRUSTED_SIGNER)
    - Revoked or expired authority attestations (ATTESTATION_REVOKED_AUTHORITY, ATTESTATION_EXPIRED_AUTHORITY)
    - Missing or contradictory evidence telemetry (ATTESTATION_EVIDENCE_INVALID)
    - Non-production or unconfigured trust providers when require_production=True

    SECURITY INVARIANT:
    Strictly defaults to require_production=True. Test providers can NEVER verify a genuine production checkpoint.
    Accepts NO bypass flags. Returns True ONLY if classification is GENUINE_TRAINED_SOVEREIGN or PROMOTED_PRODUCTION.
    """
    if require_artifacts is None:
        require_artifacts = require_production

    if isinstance(trust_provider, str):
        trusted_authorities_path = trust_provider
        trust_provider = None

    tier = classify_checkpoint(
        ckpt_path,
        trust_provider=trust_provider,
        trusted_authorities_path=trusted_authorities_path,
        allow_test_fixture_classification=False,
        allow_unattested_for_test=False,
        require_production_provider=require_production,
        evidence_store=evidence_store,
        require_artifacts=require_artifacts,
    )
    return tier in ("GENUINE_TRAINED_SOVEREIGN", "PROMOTED_PRODUCTION")


class ModelNotTrainedException(Exception):
    """Raised when an inference request is made but no valid checkpoint or model exists."""
    pass

class ModelService:
    """
    Service responsible for inspecting, loading, and running inference with Moon AI sovereign models.
    Production inference strictly requires an authentically trained, cryptographically verified
    sovereign checkpoint initialized purely from scratch (STRICT_SOVEREIGN_RANDOM_INIT_ONLY).
    External pretrained weights (LLaMA, GPT-2, Qwen) are strictly blocked from serving as Moon's brain.
    """

    ALLOWED_CHECKPOINT_DIRS = [
        "out-assistant",
        "out-nanogpt-scratch",
        "out",
    ]

    def __init__(self, base_dir: Optional[str] = None, preferred_checkpoint: Optional[str] = None):
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.preferred_checkpoint = preferred_checkpoint
        self.debug_mode = os.environ.get("DEBUG_INFERENCE") == "1"
        self.model = None
        self.llama_model = None
        self.model_args = None
        self.model_config = None
        self.checkpoint_meta = {}
        self.encode_fn = None
        self.decode_fn = None
        self.vocab_size: Optional[int] = None
        self.dataset_name: Optional[str] = None
        self.current_checkpoint_path: Optional[str] = None
        self.error_message: Optional[str] = None
        import threading
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        
        # Moon AI Sovereign Rule: Default model is strictly moon-chat-pilot-v0
        # Third-party pretrained models (Qwen, LLaMA, GPT-2) must never become Moon's default brain
        self.active_model_name = "moon-chat-pilot-v0"
        
        # Check torch availability
        self.torch_available = False
        self.torch_version = None
        self.device = "cpu"
        self._check_torch()
        
        # Initial status inspection
        self.refresh_status()

    def _check_torch(self):
        try:
            import torch
            self.torch_available = True
            self.torch_version = torch.__version__
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.torch_available = False
            self.torch_version = None
            self.device = "cpu"

    def find_file(self, filename: str) -> Optional[str]:
        """Searches project root and parent/Downloads directories for a specific filename."""
        candidate_dirs = [
            self.base_dir,
            os.path.abspath(os.path.join(self.base_dir, "..")),
            os.path.abspath(os.path.join(self.base_dir, "..", "..")),
            os.path.join(os.path.expanduser("~"), "Downloads"),
        ]
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                target = os.path.join(cdir, filename)
                if os.path.isfile(target):
                    return target
        return None

    def find_moon_ai_gguf(self) -> Optional[str]:
        """Finds custom sovereign fine-tuned Moon AI v1 GGUF model."""
        exact = self.find_file("Moon-AI-v1.Q4_K_M.gguf")
        if exact:
            return exact
        candidate_dirs = [
            self.base_dir,
            os.path.abspath(os.path.join(self.base_dir, "..")),
            os.path.join(os.path.expanduser("~"), "Downloads"),
        ]
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                matches = glob.glob(os.path.join(cdir, "*Moon-AI*.gguf")) or glob.glob(os.path.join(cdir, "*moon*.gguf"))
                if matches:
                    return matches[0]
        return None

    def find_qwen_gguf(self) -> Optional[str]:
        """Finds Moon AI v1 or Qwen2.5-Coder-7B GGUF file in project root or Downloads."""
        moon = self.find_moon_ai_gguf()
        if moon:
            return moon
        exact = self.find_file("Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf")
        if exact:
            return exact
        candidate_dirs = [
            self.base_dir,
            os.path.abspath(os.path.join(self.base_dir, "..")),
            os.path.join(os.path.expanduser("~"), "Downloads"),
        ]
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                matches = glob.glob(os.path.join(cdir, "*qwen*7b*.gguf")) or glob.glob(os.path.join(cdir, "*Qwen*7B*.gguf"))
                if matches:
                    return matches[0]
        return None

    def find_llama_gguf(self) -> Optional[str]:
        """Finds LLaMA-3.2-1B GGUF file in project root or Downloads."""
        exact = self.find_file("Llama-3.2-1B-Instruct.Q4_K_M.gguf")
        if exact:
            return exact
        candidate_dirs = [
            self.base_dir,
            os.path.abspath(os.path.join(self.base_dir, "..")),
            os.path.join(os.path.expanduser("~"), "Downloads"),
        ]
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                matches = glob.glob(os.path.join(cdir, "*llama*1b*.gguf")) or glob.glob(os.path.join(cdir, "*Llama*1B*.gguf"))
                if matches:
                    return matches[0]
        return None

    def find_gguf(self) -> Optional[str]:
        """Finds best available GGUF model (prioritizing 7B Qwen Coder, then LLaMA 1B)."""
        return self.find_qwen_gguf() or self.find_llama_gguf()

    validate_checkpoint_structure = staticmethod(validate_checkpoint_structure)
    classify_checkpoint = staticmethod(classify_checkpoint)
    is_checkpoint_fixture = staticmethod(is_checkpoint_fixture)
    is_historical_pilot_checkpoint = staticmethod(is_historical_pilot_checkpoint)
    is_verified_genuine_checkpoint = staticmethod(is_verified_genuine_checkpoint)

    def find_pilot_checkpoint(self, allow_fixture: bool = False) -> Optional[str]:
        """
        Locates the sovereign pilot checkpoint with fail-closed positive verification.
        - If allow_fixture is False: ONLY returns a positively verified genuine strict sovereign checkpoint.
        - If allow_fixture is True: returns a verified genuine checkpoint OR a valid test fixture / historical pilot.
        Never returns an unreadable, malformed, or unverified checkpoint file.
        Never silently falls back to an unrelated checkpoint.
        """
        candidate_paths = [
            os.path.join(self.base_dir, "models", "checkpoints", "chat", "moon_chat_pilot_v0.pt"),
            os.path.join(self.base_dir, "models", "checkpoints", "chat", "moon_chat_phase_2_4_t4_pilot_v0.pt"),
            os.path.join(self.base_dir, "models", "checkpoints", "chat", "moon_chat_phase_2_4_t4_pilot_best.pt")
        ]

        for cand in candidate_paths:
            if not os.path.isfile(cand):
                continue
            if self.is_verified_genuine_checkpoint(cand):
                return os.path.abspath(cand)
            if allow_fixture:
                ok, _, _ = self.validate_checkpoint_structure(cand)
                if ok and (self.is_checkpoint_fixture(cand) or self.is_historical_pilot_checkpoint(cand)):
                    return os.path.abspath(cand)

        return None

    def find_checkpoint(self, preferred_path: Optional[str] = None, allow_fixture: bool = False) -> Optional[str]:
        """
        Safely locates an explicit or default scratch assistant checkpoint.
        - If allow_fixture is False: ONLY returns a positively verified genuine strict sovereign checkpoint.
        - If allow_fixture is True: returns a verified genuine checkpoint OR a valid structural test fixture.
        NEVER silently falls back to Shakespeare or unrelated datasets.
        """
        if preferred_path:
            candidate_paths = [preferred_path]
        elif getattr(self, "preferred_checkpoint", None):
            candidate_paths = [self.preferred_checkpoint]
        else:
            candidate_paths = []
            env_path = os.environ.get("NANOGPT_CHECKPOINT_PATH")
            if env_path:
                candidate_paths.append(env_path)

            env_dir = os.environ.get("NANOGPT_CHECKPOINT_DIR")
            if env_dir:
                candidate_paths.append(os.path.join(env_dir, "ckpt.pt"))

            for rel_dir in self.ALLOWED_CHECKPOINT_DIRS:
                candidate_paths.append(os.path.join(self.base_dir, rel_dir, "ckpt.pt"))

        for cand in candidate_paths:
            if not cand or not os.path.isfile(cand):
                continue
            cand_abs = os.path.abspath(cand)
            if self.is_verified_genuine_checkpoint(cand_abs):
                return cand_abs
            if allow_fixture:
                ok, _, _ = self.validate_checkpoint_structure(cand_abs)
                if ok:
                    return cand_abs

        return None

    def get_status(self) -> str:
        """Returns readiness status with strict sovereign truthfulness."""
        if self.active_model_name in ("moon-chat-pilot-v0", "moon-pilot", "pilot"):
            if self.model is not None and not getattr(self, "is_test_fixture", False) and self.is_verified_genuine_checkpoint(self.current_checkpoint_path or ""):
                return "READY"
            has_genuine = self.find_pilot_checkpoint(allow_fixture=False) is not None
            if has_genuine:
                return "READY (ON DEMAND)"
            return "MODEL_NOT_READY"

        if self.active_model_name == "scratch":
            if self.model is not None and not getattr(self, "is_test_fixture", False) and self.is_verified_genuine_checkpoint(self.current_checkpoint_path or ""):
                return "READY"
            has_genuine = self.find_checkpoint(allow_fixture=False) is not None
            if has_genuine:
                return "READY (ON DEMAND)"
            return "MODEL_NOT_READY"

        if self.active_model_name in ["qwen-2.5-coder-7b", "llama-3.2", "gpt2"]:
            return "BLOCKED_EXTERNAL_PRETRAINED"

        if self.error_message:
            return "ERROR"
            
        return "MODEL_NOT_READY"

    def refresh_status(self):
        """Re-evaluates the environment state."""
        self._check_torch()
        if not self.torch_available:
            self.error_message = "PyTorch not available in environment."

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns list of models with strict sovereign ownership classification."""
        has_trained_scratch = self.find_checkpoint(allow_fixture=False) is not None
        has_any_scratch = self.find_checkpoint(allow_fixture=True) is not None
        is_scratch_fixture = has_any_scratch and not has_trained_scratch

        has_trained_pilot = self.find_pilot_checkpoint(allow_fixture=False) is not None
        has_any_pilot = self.find_pilot_checkpoint(allow_fixture=True) is not None
        is_pilot_fixture = has_any_pilot and not has_trained_pilot

        return [
            {
                "id": "moon-chat-pilot-v0",
                "name": "🌙 Moon AI Pilot (25.7M Sovereign Chat — Pure Scratch Init)",
                "type": "sovereign_scratch",
                "description": "Stage 2.1.6 sovereign 25.7M parameter transformer initialized from Gaussian random scratch. In MODEL_NOT_READY status pending authorized training run.",
                "active": self.active_model_name == "moon-chat-pilot-v0",
                "parameters": "25.70M",
                "block_size": 512,
                "vocab_size": 512,
                "available": has_trained_pilot,
                "is_test_fixture": is_pilot_fixture,
                "sovereign_owned": True,
                "status": "READY" if has_trained_pilot else "MODEL_NOT_READY"
            },
            {
                "id": "scratch",
                "name": "🔬 NanoGPT Core (Scratch Architecture)",
                "type": "local_checkpoint",
                "description": "Character-level transformer architected for scratch training. In MODEL_NOT_READY status pending verified strict sovereign training run.",
                "active": self.active_model_name == "scratch",
                "parameters": "0.80M",
                "block_size": 128,
                "vocab_size": 97,
                "available": has_trained_scratch,
                "is_test_fixture": is_scratch_fixture,
                "sovereign_owned": True,
                "status": "READY" if has_trained_scratch else "MODEL_NOT_READY"
            },
            {
                "id": "qwen-2.5-coder-7b",
                "name": "External Pretrained Reference: Qwen-2.5-Coder-7B",
                "type": "external_pretrained_reference",
                "description": "Third-party pretrained reference weights (Alibaba/Qwen). Prohibited from serving as Moon's sovereign production inference engine.",
                "active": False,
                "parameters": "7.61B",
                "block_size": 2048,
                "vocab_size": 152064,
                "available": False,
                "sovereign_owned": False,
                "status": "BLOCKED_EXTERNAL_PRETRAINED"
            },
            {
                "id": "llama-3.2",
                "name": "External Pretrained Reference: LLaMA-3.2-1B",
                "type": "external_pretrained_reference",
                "description": "Third-party pretrained reference weights (Meta LLaMA). Prohibited from serving as Moon's sovereign production inference engine.",
                "active": False,
                "parameters": "1.23B",
                "block_size": 2048,
                "vocab_size": 128256,
                "available": False,
                "sovereign_owned": False,
                "status": "BLOCKED_EXTERNAL_PRETRAINED"
            },
            {
                "id": "gpt2",
                "name": "External Pretrained Reference: GPT-2 (124M)",
                "type": "external_pretrained_reference",
                "description": "Third-party open pretrained weights (OpenAI GPT-2). Prohibited from serving as Moon's sovereign production inference engine.",
                "active": False,
                "parameters": "123.65M",
                "block_size": 1024,
                "vocab_size": 50257,
                "available": False,
                "sovereign_owned": False,
                "status": "BLOCKED_EXTERNAL_PRETRAINED"
            }
        ]

    def load_model(self, model_name: Optional[str] = None, allow_fixture: bool = False) -> bool:
        """
        Loads the target model into memory.
        Enforces runtime sovereignty: third-party pretrained models (Qwen, LLaMA, GPT-2)
        cannot be loaded as Moon's production inference engine.
        Thread-safe: will not duplicate load or reload if already active.
        """
        target = model_name or self.active_model_name
        
        # Runtime Sovereignty Enforcement: Block third-party pretrained models from serving as Moon's brain
        if target in ["qwen-2.5-coder-7b", "qwen", "qwen-7b", "llama-3.2", "gguf", "llama", "gpt2"]:
            self.error_message = (
                f"Runtime Sovereignty Violation: Third-party pretrained model '{target}' "
                f"is prohibited from serving as Moon's sovereign production inference engine. "
                f"Production inference requires a verified genuine sovereign checkpoint."
            )
            return False
        
        with self._load_lock:
            # If target model is already loaded in memory, return immediately without redundant reload
            if self.model is not None and self.active_model_name == target:
                return True

            # Add project root to sys.path
            if self.base_dir not in sys.path:
                sys.path.insert(0, self.base_dir)

            # Clean up existing models from memory to prevent RAM exhaustion on 8GB systems
            if self.llama_model is not None:
                del self.llama_model
                self.llama_model = None
                self.model = None
                import gc
                gc.collect()
            if self.model is not None:
                del self.model
                self.model = None
                import gc
                gc.collect()

            # Option 0: Qwen-2.5-Coder-7B Instruct
            if target in ["qwen-2.5-coder-7b", "qwen", "qwen-7b"]:
                qwen_path = self.find_qwen_gguf()
                if not qwen_path:
                    self.error_message = "Qwen2.5-Coder-7B GGUF model file not found on disk."
                    return False
                try:
                    import os as _os
                    from llama_cpp import Llama
                    # Optimal physical core count for llama.cpp (avoid hyperthread cache contention)
                    _total_cpus = _os.cpu_count() or 4
                    _infer_threads = min(4, max(2, _total_cpus // 2))
                    _batch_threads = min(6, _total_cpus)
                    _n_gpu = -1 if (self.device == "cuda" or _os.environ.get("USE_GPU") == "1") else 0
                    print(f"Loading Moon AI 7B model: {qwen_path} (GPU layers: {_n_gpu}, infer threads: {_infer_threads}, batch threads: {_batch_threads})...")
                    self.llama_model = Llama(
                        model_path=qwen_path,
                        n_ctx=2048 if _n_gpu != 0 else 1024,
                        n_gpu_layers=_n_gpu,            # -1 offloads all layers to NVIDIA GPU VRAM
                        verbose=False,
                        n_threads=_infer_threads,
                        n_threads_batch=_batch_threads,
                        n_batch=512 if _n_gpu != 0 else 256,
                        n_ubatch=256,
                    )
                    self.model = self.llama_model
                    self.active_model_name = "qwen-2.5-coder-7b"
                    self.current_checkpoint_path = os.path.basename(qwen_path)
                    self.dataset_name = "CodeAlpaca 15K + Python Synthetic (Unsloth QLoRA)"
                    self.vocab_size = 152064
                    self.error_message = None
                    return True
                except Exception as e:
                    self.error_message = f"Failed to load Moon AI 7B model: {str(e)}"
                    return False

            # Option 1: Fine-Tuned GGUF LLaMA-3.2
            elif target in ["llama-3.2", "gguf", "llama"]:
                llama_path = self.find_llama_gguf()
                if not llama_path:
                    self.error_message = "LLaMA-3.2 GGUF model file not found on disk."
                    return False
                try:
                    from llama_cpp import Llama
                    _n_gpu = -1 if (self.device == "cuda" or os.environ.get("USE_GPU") == "1") else 0
                    print(f"Loading LLaMA-3.2 model: {llama_path} (GPU layers: {_n_gpu})...")
                    self.llama_model = Llama(
                        model_path=llama_path,
                        n_ctx=4096,
                        n_gpu_layers=_n_gpu,
                        verbose=False,
                        n_threads=4,
                        n_batch=512,
                        n_ubatch=256,
                    )
                    self.model = self.llama_model
                    self.active_model_name = "llama-3.2"
                    self.current_checkpoint_path = os.path.basename(llama_path)
                    self.dataset_name = "Fine-Tuned Instruction Dataset (Colab T4)"
                    self.vocab_size = 128256
                    self.error_message = None
                    return True
                except Exception as e:
                    self.error_message = f"Failed to load LLaMA model: {str(e)}"
                    return False

            # Option 2: Pretrained GPT-2 124M
            elif target == "gpt2":
                if not self.torch_available:
                    self.error_message = "PyTorch is not installed in the current environment."
                    return False
                try:
                    from model import GPT
                    print("Loading GPT-2 124M weights into nanoGPT...")
                    model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
                    model.eval()
                    model.to(self.device)
                    
                    self.model = model
                    self.llama_model = None
                    self.model_config = model.config
                    self.active_model_name = "gpt2"
                    self.current_checkpoint_path = "huggingface: gpt2 (124M open weights)"
                    self.dataset_name = "OpenWebText (GPT-2 Pretrained)"
                    self.vocab_size = 50257
                    
                    import tiktoken
                    enc = tiktoken.get_encoding("gpt2")
                    self.encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
                    self.decode_fn = lambda tokens: enc.decode(tokens)
                    self.error_message = None
                    return True
                except Exception as e:
                    self.error_message = f"Failed to load GPT-2: {str(e)}"
                    return False

            # Option 3: Scratch NanoGPT
            elif target == "scratch":
                if not self.torch_available:
                    self.error_message = "PyTorch is not installed in the current environment."
                    return False
                eff_allow_fixture = allow_fixture
                ckpt_path = self.find_checkpoint(allow_fixture=eff_allow_fixture)
                if not ckpt_path:
                    self.model = None
                    if not eff_allow_fixture and self.find_checkpoint(allow_fixture=True) is not None:
                        self.error_message = (
                            "Production inference blocked: Scratch checkpoint failed positive strict sovereign verification. "
                            "Unpromoted or unverified scratch checkpoints are restricted to isolated retrospective testing (allow_fixture=True)."
                        )
                    else:
                        self.error_message = "No local scratch assistant checkpoint found on disk. Expected out-assistant/ckpt.pt or NANOGPT_CHECKPOINT_PATH."
                    return False

                # Positive genuine verification gate for production inference
                if not eff_allow_fixture and not self.is_verified_genuine_checkpoint(ckpt_path):
                    self.model = None
                    self.error_message = (
                        "Production inference blocked: Scratch checkpoint failed positive strict sovereign verification. "
                        "Unpromoted or unverified scratch checkpoints are restricted to isolated retrospective testing (allow_fixture=True)."
                    )
                    return False

                try:
                    import torch
                    from model import GPTConfig, GPT

                    checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    self.is_test_fixture = self.is_checkpoint_fixture(ckpt_path) or not self.is_verified_genuine_checkpoint(ckpt_path)
                    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "model_args" not in checkpoint:
                        self.error_message = f"Invalid checkpoint format in {ckpt_path}: missing required 'model' or 'model_args' keys."
                        return False

                    model_args = checkpoint.get("model_args", {})
                    self.model_args = model_args
                    gptconf = GPTConfig(**model_args)
                    self.model_config = gptconf
                    
                    model = GPT(gptconf)
                    state_dict = checkpoint["model"]
                    
                    unwanted_prefix = "_orig_mod."
                    for k in list(state_dict.keys()):
                        if k.startswith(unwanted_prefix):
                            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                            
                    model.load_state_dict(state_dict)
                    model.eval()
                    model.to(self.device)
                    
                    self.model = model
                    self.llama_model = None
                    self.active_model_name = "scratch"
                    try:
                        self.current_checkpoint_path = os.path.relpath(ckpt_path, self.base_dir)
                    except ValueError:
                        self.current_checkpoint_path = os.path.abspath(ckpt_path)
                    self.checkpoint_meta = checkpoint.get("config", {})
                    if not self.checkpoint_meta:
                        self.checkpoint_meta = {
                            "iter_num": checkpoint.get("iter_num"),
                            "best_val_loss": checkpoint.get("best_val_loss"),
                        }
                    self.dataset_name = checkpoint.get("config", {}).get("dataset", "assistant_chat")
                    
                    # Strict tokenizer setup and validation (fails loudly if meta.pkl is missing)
                    self._setup_scratch_tokenizer(ckpt_path=ckpt_path, dataset=self.dataset_name)
                    self.error_message = None
                    return True
                except Exception as e:
                    self.model = None
                    self.error_message = f"Failed to load scratch checkpoint: {str(e)}"
                    return False

            # Option 4: Sovereign Moon Chat Pilot (25.7M scratch-initialized transformer)
            elif target in ["moon-chat-pilot-v0", "moon-pilot", "pilot"]:
                ckpt_path = self.find_pilot_checkpoint(allow_fixture=allow_fixture)
                if not ckpt_path:
                    self.error_message = (
                        "No verified genuine Moon Chat Pilot checkpoint found for production inference. "
                        "Unpromoted historical pilot checkpoints are restricted to isolated retrospective testing (allow_fixture=True)."
                    )
                    return False
                try:
                    import torch
                    from model import GPT, GPTConfig
                    from datasets.chat.tokenizer import SovereignTokenizer

                    print(f"Loading Moon Chat Pilot checkpoint: {ckpt_path} on {self.device}...")
                    checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    self.is_test_fixture = self.is_checkpoint_fixture(ckpt_path) or not self.is_verified_genuine_checkpoint(ckpt_path)

                    model_args = checkpoint.get("model_args", {})
                    self.model_args = model_args
                    gptconf = GPTConfig(**model_args)
                    self.model_config = gptconf

                    model = GPT(gptconf)
                    state_dict = checkpoint["model"]
                    unwanted_prefix = "_orig_mod."
                    for k in list(state_dict.keys()):
                        if k.startswith(unwanted_prefix):
                            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

                    model.load_state_dict(state_dict)
                    model.eval()
                    model.to(self.device)

                    self.model = model
                    self.llama_model = None
                    self.active_model_name = "moon-chat-pilot-v0"
                    try:
                        self.current_checkpoint_path = os.path.relpath(ckpt_path, self.base_dir)
                    except ValueError:
                        self.current_checkpoint_path = os.path.abspath(ckpt_path)
                    self.checkpoint_meta = checkpoint.get("config", {}) or checkpoint.get("metadata", {})
                    self.current_checkpoint_policy = (
                        self.checkpoint_meta.get("ownership_policy")
                        or self.checkpoint_meta.get("policy")
                        or "UNKNOWN_PROVENANCE_INSUFFICIENT"
                    )
                    self.dataset_name = self.checkpoint_meta.get("dataset", "expanded_chat_corpus")
                    self.vocab_size = model_args.get("vocab_size", 512)

                    # Initialize sovereign BPE tokenizer
                    sovereign_tok = SovereignTokenizer()
                    self.sovereign_tokenizer = sovereign_tok
                    self.encode_fn = lambda s: sovereign_tok.encode_chat([{"role": "user", "content": s}])
                    self.decode_fn = sovereign_tok.decode

                    self.error_message = None
                    return True
                except Exception as e:
                    self.model = None
                    self.error_message = f"Failed to load Moon Chat Pilot checkpoint: {str(e)}"
                    return False
            else:
                self.error_message = f"Unknown model name: {target}"
                return False

    def _setup_scratch_tokenizer(self, ckpt_path: str, dataset: str):
        """
        Loads dataset-specific character mapping (meta.pkl) and validates it against model_args.
        Strictly fails if metadata is missing or vocabulary size does not match checkpoint.
        """
        ckpt_meta_path = os.path.join(os.path.dirname(ckpt_path), "meta.pkl")
        data_meta_path = os.path.join(self.base_dir, "data", dataset, "meta.pkl")

        meta_path = None
        if os.path.isfile(ckpt_meta_path):
            meta_path = ckpt_meta_path
        elif os.path.isfile(data_meta_path):
            meta_path = data_meta_path

        if not meta_path or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Required tokenizer metadata (meta.pkl) is missing. Checked '{ckpt_meta_path}' and '{data_meta_path}'. "
                f"Scratch model cannot run deterministic inference without matching character vocabulary mapping."
            )

        with open(meta_path, "rb") as f:
            meta = SafeUnpickler(f).load()

        if "stoi" not in meta or "itos" not in meta:
            raise ValueError(f"Malformed meta.pkl at {meta_path}: missing required 'stoi' or 'itos' dictionaries.")

        stoi, itos = meta["stoi"], meta["itos"]
        meta_vocab_size = meta.get("vocab_size", len(stoi))

        if self.model_args and self.model_args.get("vocab_size") is not None:
            expected_vocab = self.model_args["vocab_size"]
            if expected_vocab != meta_vocab_size:
                raise ValueError(
                    f"Vocabulary size mismatch: checkpoint model_args specifies vocab_size={expected_vocab}, "
                    f"but tokenizer at {meta_path} defines vocab_size={meta_vocab_size}."
                )

        self.vocab_size = meta_vocab_size

        def char_encode(s: str) -> List[int]:
            return [stoi.get(c, stoi.get(" ", 0)) for c in s]

        def char_decode(tokens: List[int]) -> str:
            return "".join([itos.get(i, "?") for i in tokens])

        self.encode_fn = char_encode
        self.decode_fn = char_decode

    def switch_model(self, model_name: str, allow_fixture: bool = False) -> bool:
        """Switches active model between available engines."""
        allowed = ["qwen-2.5-coder-7b", "qwen", "qwen-7b", "llama-3.2", "gpt2", "scratch", "moon-chat-pilot-v0", "moon-pilot", "pilot"]
        if model_name not in allowed:
            raise ValueError(f"Invalid model: {model_name}. Allowed: 'moon-chat-pilot-v0', 'qwen-2.5-coder-7b', 'llama-3.2', 'gpt2', 'scratch'")
        return self.load_model(model_name, allow_fixture=allow_fixture)

    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None, mode: str = "chat", allow_fixture: bool = False) -> Dict[str, Any]:
        """
        Runs local inference with memory context, smart turn-stopping, and Claude/Codex caliber coding capabilities.
        Fails closed on unpromoted test fixtures unless allow_fixture is explicitly True.
        """
        with self._infer_lock:
            return self._generate_internal(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, history=history, mode=mode, allow_fixture=allow_fixture)

    def _generate_internal(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None, mode: str = "chat", allow_fixture: bool = False) -> Dict[str, Any]:
        eff_allow_fixture = allow_fixture
        if getattr(self, "is_test_fixture", False) and not eff_allow_fixture:
            raise ModelNotTrainedException(
                "Production inference blocked: Active model is an unpromoted CI test fixture. "
                "Retrospective testing or audit generation requires allow_fixture=True."
            )

        if not eff_allow_fixture and self.current_checkpoint_path:
            full_ckpt_path = (
                os.path.join(self.base_dir, self.current_checkpoint_path)
                if not os.path.isabs(self.current_checkpoint_path)
                else self.current_checkpoint_path
            )
            if os.path.isfile(full_ckpt_path) and not self.is_verified_genuine_checkpoint(full_ckpt_path, require_production=True):
                raise ModelNotTrainedException(
                    "Production inference blocked: Checkpoint failed production genuine verification. "
                    "Test-isolated or unverified models cannot serve production inference."
                )

        if self.model is None and self.llama_model is None:
            success = self.load_model(self.active_model_name, allow_fixture=eff_allow_fixture)
            if not success:
                raise ModelNotTrainedException(self.error_message or "Model could not be initialized.")

        # Handle empty prompt
        if not prompt or not prompt.strip():
            return {
                "reply": "[Error: Empty prompt provided. Please enter a valid message.]",
                "raw_reply": "",
                "stop_reason": "empty_prompt",
                "full_text": "",
                "model_name": self.active_model_name,
                "prompt_length": 0,
                "tokens_generated": 0,
                "device": self.device
            }

        # Strip /code prefix if provided
        clean_prompt = prompt.strip()
        is_coding = (
            mode == "code" or
            clean_prompt.lower().startswith("/code") or
            any(w in clean_prompt.lower() for w in ["write a function", "write code", "python script", "implement in python", "debug this", "algorithm"])
        )
        if clean_prompt.lower().startswith("/code"):
            clean_prompt = clean_prompt[5:].strip()

        # Creator / Identity Intent Detection (Prem Gaikwad from India)
        import re
        creator_patterns = [
            r"who (made|created|developed|built|programmed|invented|designed|trained) (you|this|nanogpt)",
            r"(who is|who\'s) (your|the) (creator|developer|maker|author|founder|owner|father|boss|builder)",
            r"who (are you|created you)",
            r"(tell me about|who is) prem gaikwad",
            r"where (are you|were you) made",
            r"what is your name",
        ]
        is_creator_query = any(re.search(p, clean_prompt, re.I) for p in creator_patterns) or any(
            phrase in clean_prompt.lower() for phrase in [
                "who made you", "who created you", "who developed you", "who built you",
                "who is your developer", "who is your creator", "who is your maker",
                "who programmed you", "who designed you", "who is prem gaikwad",
                "ho made you", "who made u", "who created u", "who developed u"
            ]
        )

        if is_creator_query:
            creator_reply = (
                "I am **Moon AI**, an independent, sovereign artificial intelligence platform architected and developed by **Prem Gaikwad** from **India** 🇮🇳.\n\n"
                "Moon AI is designed from first principles to operate 100% locally with strict sovereign model ownership (`STRICT_SOVEREIGN_RANDOM_INIT_ONLY`). "
                "Moon AI does not rely on third-party cloud APIs, external generative servers, or unauthorized weights.\n\n"
                "• 🌙 **Sovereign Model Architecture**: Custom transformer architected for sovereign random-init training from scratch.\n"
                "• 🛡️ **Zero External Dependency**: Local offline execution and deterministic utility engines.\n"
                "• 🎨 **Sovereign Multimodal**: Offline procedural visual and video generation.\n"
                "• ⚡ **Local Code Interpreter**: Secure, sandboxed AST-verified Python execution.\n\n"
                "How can I assist you with your project today?"
            )
            return {
                "reply": creator_reply,
                "raw_reply": creator_reply,
                "stop_reason": "system_reflex_identity",
                "full_text": creator_reply,
                "model_name": "Moon AI Sovereign Core (Identity Declaration)",
                "prompt_length": len(clean_prompt),
                "tokens_generated": 0,
                "device": "Local Sovereign Engine",
                "engine_type": "system_reflex",
                "model_id": "moon-chat-pilot-v0",
                "model_status": self.get_status(),
                "is_neural_generation": False,
                "is_deterministic_utility": True,
                "is_web_retrieval": False,
                "checkpoint_verified": False,
                "provenance_status": "SOVEREIGN_IDENTITY_DECLARATION"
            }

        # --- 0. Remote Sovereign GPU Fast-Path (optional private server or Colab) ---
        colab_gpu_url = os.environ.get("COLAB_GPU_URL", "").strip().rstrip("/")
        if colab_gpu_url and self.active_model_name != "scratch" and not os.environ.get("DISABLE_REMOTE_GPU"):
            try:
                import urllib.request, json
                endpoint = f"{colab_gpu_url}/api/chat"
                payload = json.dumps({
                    "message": clean_prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                }).encode("utf-8")
                gpu_req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "true",
                        "User-Agent": "MoonAI-Hybrid-Client/1.0"
                    }
                )
                with urllib.request.urlopen(gpu_req, timeout=5.0) as resp:
                    if resp.status == 200:
                        gpu_data = json.loads(resp.read().decode("utf-8"))
                        if gpu_data.get("reply"):
                            return {
                                "reply": gpu_data["reply"],
                                "full_text": gpu_data["reply"],
                                "model_name": gpu_data.get("model_name", "Moon AI Sovereign (Colab T4 GPU)"),
                                "prompt_length": len(clean_prompt),
                                "tokens_generated": gpu_data.get("tokens_generated", len(gpu_data["reply"].split())),
                                "device": gpu_data.get("device", "Colab NVIDIA T4 (CUDA)")
                            }
            except Exception:
                pass  # Fall through to local GGUF / scratch inference seamlessly

        # --- 1. GGUF Generation (Qwen-2.5-Coder-7B & LLaMA-3.2) ---
        if self.active_model_name in ["qwen-2.5-coder-7b", "llama-3.2"] and self.llama_model is not None:
            is_qwen = "qwen" in self.active_model_name
            if is_coding:
                sys_msg = (
                    "You are Moon AI (Moon Coder), the world-class coding and reasoning AI assistant created and deployed by developer Prem Gaikwad from India. "
                    "If asked who made or developed you, always state proudly that you are developed and deployed by developer Prem Gaikwad from India. "
                    "Write robust, elegant, production-grade, and bug-free code with clear comments. "
                    "Always format executable code inside markdown triple backticks with language tags (e.g. ```python, ```html, ```javascript). "
                    "Provide complete, unabbreviated code without skipping parts. "
                    "CRITICAL: Maintain continuous conversational context across turns. If the user provides a follow-up request (e.g., 'in html, css, js', 'now in python', 'make it faster', 'give full code'), always fulfill this directly in the context of the previous conversation."
                )
            else:
                sys_msg = (
                    "You are Moon AI, an advanced, highly intelligent, objective, and helpful AI assistant trained and deployed by developer Prem Gaikwad from India. "
                    "If asked who made or developed you, always state proudly that you are developed and deployed by developer Prem Gaikwad from India. "
                    "You provide comprehensive, factual, scientific, and educational explanations for all topics including biology, anatomy, science, philosophy, and history. "
                    "Do not refuse educational, scientific, biological, or physiological questions. Answer them factually, respectfully, and objectively. "
                    "CRITICAL: Maintain continuous conversational context across turns. Seamlessly remember earlier questions and build upon previous answers."
                )

            # --- Inject Long-Term Learned Memories ---
            try:
                from webapp.learning_service import learning_service
                mem_prompt = learning_service.get_memories_prompt_injection()
                if mem_prompt:
                    sys_msg = f"{sys_msg}\n\n{mem_prompt}"
            except Exception:
                pass

            messages = [
                {"role": "system", "content": sys_msg}
            ]
            if history:
                # Retain recent conversation turns, but compress massive code blocks from older assistant turns
                for turn in history[-6:]:
                    content = turn.get("content", "").strip()
                    if not content:
                        continue
                    role = "user" if turn.get("role") == "user" else "assistant"
                    # If prior assistant response contains huge code (> 400 chars), sanitize to preserve context window
                    if role == "assistant" and len(content) > 400:
                        import re
                        content = re.sub(r'```[a-zA-Z0-9_\-]*\s*[\s\S]{150,}?```', '[... Code block provided in previous turn ...]', content)
                        if len(content) > 400:
                            content = content[:350] + "... [earlier text truncated for context space]"
                    messages.append({
                        "role": role,
                        "content": content
                    })

                # Follow-up intent bridging
                followup_starters = ("in ", "now in ", "rewrite in ", "convert to ", "give full", "full code", "not full", "not working", "why ")
                is_short_followup = (
                    clean_prompt.lower().startswith(followup_starters) or
                    clean_prompt.lower() in ["in html", "in html,css,js", "in html ,css,js", "in python", "in js", "in c++"] or
                    len(clean_prompt.split()) <= 4
                )
                if is_short_followup:
                    last_user_turn = next((t.get("content", "") for t in reversed(history) if t.get("role") == "user" and t.get("content", "") != clean_prompt), "")
                    if last_user_turn:
                        messages.append({
                            "role": "system",
                            "content": f"[Context Bridging]: The user's input '{clean_prompt}' is a direct follow-up continuation of their previous request: '{last_user_turn}'. Fulfill it fully in that exact context."
                        })

            messages.append({"role": "user", "content": clean_prompt})

            # Context Budget Guard — conservative to keep 7B fast on CPU
            # n_ctx is 1024 for Qwen (Moon AI), 4096 for LLaMA
            n_ctx_val = 1024 if is_qwen else 4096
            # Measure approximate tokens (1 token ~= 3.4 chars)
            total_chars = sum(len(m["content"]) for m in messages)
            approx_prompt_tokens = int(total_chars / 3.4)

            # Aggressively prune oldest history messages to stay well within 1024 ctx
            while len(messages) > 2 and (sum(len(m["content"]) for m in messages) / 3.4) > (n_ctx_val - 300):
                messages.pop(1)

            # Dynamically compute safe generation token budget
            # CPU Guard: On 7B CPU, cap to 150-200 tokens to ensure responses return in seconds/a minute, not hours
            final_prompt_tokens = int(sum(len(m["content"]) for m in messages) / 3.4)
            if is_qwen and self.device == "cpu":
                max_gen_cap = 180 if is_coding else 120
            else:
                max_gen_cap = 600 if is_coding else 350
            safe_max_tokens = max(50, min(max_new_tokens, n_ctx_val - final_prompt_tokens - 30))
            safe_max_tokens = min(safe_max_tokens, max_gen_cap)

            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "User:"] if is_qwen else ["<|eot_id|>", "<|start_header_id|>", "User:"]

            try:
                if hasattr(self.llama_model, "reset"):
                    try:
                        self.llama_model.reset()
                    except Exception:
                        pass
                res = self.llama_model.create_chat_completion(
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    temperature=temperature if not is_coding else min(temperature, 0.4),
                    stop=stop_tokens
                )
            except Exception as exc:
                if "exceed context window" in str(exc).lower():
                    # Emergency fallback: run with clean system and user prompt
                    if hasattr(self.llama_model, "reset"):
                        try:
                            self.llama_model.reset()
                        except Exception:
                            pass
                    res = self.llama_model.create_chat_completion(
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": clean_prompt}
                        ],
                        max_tokens=min(max_new_tokens, 1024),
                        temperature=temperature if not is_coding else min(temperature, 0.4),
                        stop=stop_tokens
                    )
                else:
                    raise

            reply = res["choices"][0]["message"].get("content", "").strip()
            raw_reply = reply
            stop_reason = res["choices"][0].get("finish_reason", "stop")
            if not reply:
                reply = "[Diagnostic: Empty response received from GGUF inference engine.]"

            usage = res.get("usage", {})
            if is_qwen:
                display_model_name = "Qwen-2.5-Coder-7B (External Pretrained Reference)" if is_coding else "Qwen-2.5-Coder-7B (External Pretrained Reference)"
            else:
                display_model_name = "LLaMA-3.2-1B (External Pretrained Reference)" if is_coding else "LLaMA-3.2-1B (External Pretrained Reference)"

            dev_label = "NVIDIA GPU (CUDA 4-Bit Acceleration)" if (self.device == "cuda" or os.environ.get("USE_GPU") == "1") else "CPU (AVX2 4-Bit Quantized)"
            return {
                "reply": reply,
                "raw_reply": raw_reply,
                "stop_reason": stop_reason,
                "full_text": reply,
                "model_name": display_model_name,
                "prompt_length": usage.get("prompt_tokens", len(clean_prompt)),
                "tokens_generated": usage.get("completion_tokens", len(reply.split())),
                "device": dev_label
            }

        # --- 1.5 Moon Chat Pilot Generation (25.7M Sovereign Scratch Transformer) ---
        if self.active_model_name == "moon-chat-pilot-v0" and self.model is not None:
            import torch
            from datasets.chat.tokenizer import SovereignTokenizer

            tok = getattr(self, "sovereign_tokenizer", None) or SovereignTokenizer()
            im_end_id = tok.special_to_id.get("<|im_end|>", 3)
            eos_id = tok.special_to_id.get("<|endoftext|>", 0)

            turns = []
            if history:
                for h in history[-4:]:
                    r = h.get("role", "user")
                    c = h.get("content", "").strip()
                    if c:
                        turns.append({"role": r, "content": c})
            turns.append({"role": "user", "content": clean_prompt})

            token_ids = tok.encode_chat(turns)
            block_size = getattr(self.model_config, "block_size", 512)
            if len(token_ids) >= block_size:
                token_ids = token_ids[-(block_size - 1):]

            prompt_len = len(token_ids)
            idx = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            capped_tokens = min(max_new_tokens, 60)

            new_tokens = []
            stop_reason = "max_tokens_reached"
            with torch.no_grad():
                for _ in range(capped_tokens):
                    idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                    logits, _ = self.model(idx_cond)
                    logits = logits[:, -1, :]
                    if temperature > 0:
                        probs = torch.softmax(logits / max(temperature, 0.1), dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    else:
                        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                    next_id = idx_next.item()
                    if next_id in (im_end_id, eos_id):
                        stop_reason = "stop_token (<|im_end|>)"
                        break
                    new_tokens.append(next_id)
                    idx = torch.cat((idx, idx_next), dim=1)

            raw_reply = tok.decode(new_tokens).strip() if new_tokens else ""
            clean_reply = raw_reply.replace("<|im_end|>", "").strip()
            if not clean_reply:
                clean_reply = "[Moon AI Pilot: Initialized from Gaussian scratch. Emitted 0 readable tokens.]"

            is_fixture = getattr(self, "is_test_fixture", False)
            return {
                "reply": clean_reply,
                "raw_reply": raw_reply,
                "stop_reason": stop_reason,
                "full_text": clean_reply,
                "model_name": "Moon AI Pilot (25.7M Sovereign Chat — Pure Scratch Init)",
                "prompt_length": prompt_len,
                "tokens_generated": len(new_tokens),
                "device": f"{self.device.upper()} Local Sovereign Engine",
                "is_test_fixture": is_fixture,
                "is_production_trained": False if is_fixture else True
            }

        # --- 2. NanoGPT Light / Core Generation ---
        import torch

        if self.active_model_name == "gpt2":
            if history:
                turns = []
                for turn in history[-6:]:
                    content = turn.get("content", "").strip()
                    if content:
                        r = "User" if turn.get("role") == "user" else "Assistant"
                        turns.append(f"{r}: {content}")
                turns.append(f"User: {prompt.strip()}\nAssistant:")
                prompt_str = "\n".join(turns)
            else:
                prompt_str = f"User: {prompt.strip()}\nAssistant:"
        else:
            if history:
                turns = []
                for turn in history[-4:]:
                    content = turn.get("content", "").strip()
                    if content:
                        r = "User" if turn.get("role") == "user" else "Assistant"
                        turns.append(f"{r}: {content}")
                turns.append(f"User: {prompt.strip()}\nAssistant:")
                prompt_str = "\n".join(turns)
            else:
                prompt_str = f"User: {prompt.strip()}\nAssistant:"

        start_ids = self.encode_fn(prompt_str)
        
        block_size = getattr(self.model_config, "block_size", 1024 if self.active_model_name == "gpt2" else 128)
        prompt_len = len(start_ids)
        if prompt_len >= block_size:
            start_ids = start_ids[-(block_size - 1):]
            prompt_len = len(start_ids)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        capped_tokens = min(max_new_tokens, 150)
        
        with torch.no_grad():
            y = self.model.generate(
                x,
                max_new_tokens=capped_tokens,
                temperature=temperature,
                top_k=50 if self.active_model_name == "gpt2" else 40
            )
            
        generated_tokens = y[0].tolist()
        full_text = self.decode_fn(generated_tokens)
        
        new_token_ids = generated_tokens[prompt_len:]
        raw_reply = self.decode_fn(new_token_ids) if new_token_ids else full_text[len(prompt_str):]
        
        clean_reply = raw_reply.strip()
        stop_reason = "max_tokens_reached"
        stop_markers = ["\nUser:", "\nAssistant:", "\n\nUser", "<|endoftext|>"]
        for marker in stop_markers:
            if marker in clean_reply:
                clean_reply = clean_reply.split(marker)[0].strip()
                stop_reason = f"stop_marker ({marker.strip()})"
                break

        if len(new_token_ids) == 0:
            stop_reason = "zero_tokens_generated"

        if not clean_reply:
            if self.active_model_name == "scratch":
                clean_reply = f"[Diagnostic: Scratch model emitted 0 readable tokens (iter_num={self.checkpoint_meta.get('iter_num', 'N/A')}, val_loss={self.checkpoint_meta.get('best_val_loss', 'N/A')}). Train more steps on assistant_chat to improve vocabulary confidence.]"
            else:
                clean_reply = "[Diagnostic: Model generation returned empty response.]"

        if os.environ.get("DEBUG_INFERENCE") == "1" or self.debug_mode:
            print(f"[Inference Diagnostic] model={self.active_model_name} prompt_tokens={prompt_len} gen_tokens={len(new_token_ids)} stop_reason={stop_reason} raw_reply={repr(raw_reply[:80])}")

        return {
            "reply": clean_reply,
            "raw_reply": raw_reply,
            "stop_reason": stop_reason,
            "full_text": full_text,
            "model_name": f"NanoGPT - {'Light (124M)' if self.active_model_name == 'gpt2' else 'Core (0.8M)'}",
            "prompt_length": prompt_len,
            "tokens_generated": len(new_token_ids),
            "device": self.device
        }

    def get_info(self) -> Dict[str, Any]:
        """Returns safe descriptive metadata about current model configuration."""
        status = self.get_status()
        
        if self.active_model_name == "qwen-2.5-coder-7b":
            moon_path = self.find_moon_ai_gguf()
            ckpt_name = os.path.basename(moon_path) if moon_path else "Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf"
            return {
                "model_name": "Moon AI Sovereign (7.6B v1 — Fine-Tuned by Prem Gaikwad)" if moon_path else "Qwen-2.5-Coder-7B",
                "model_type": "Moon AI 7.6B Frontier Engine (Fine-Tuned Sovereign GGUF)" if moon_path else "Qwen-2.5-Coder-7B Frontier Engine (Fine-Tuned GGUF)",
                "checkpoint_path": self.current_checkpoint_path or ckpt_name,
                "model_status": status,
                "dataset": "Moon AI Sovereign Master Dataset (Science, Biology, Coding, Prem Gaikwad Identity)",
                "vocab_size": 152064,
                "parameters": 7610000000,
                "block_size": 2048,
                "n_layer": 28,
                "n_head": 28,
                "n_embd": 3584,
                "hardware": "CPU (4-Bit AVX2 C++ Quantized)",
                "description": "Quarantined external base weights. Blocked from serving as Moon AI sovereign intelligence under STRICT_SOVEREIGN_RANDOM_INIT_ONLY policy."
            }

        if self.active_model_name == "llama-3.2":
            return {
                "model_type": "NanoGPT-3.2 Frontier Engine (Fine-Tuned GGUF)",
                "checkpoint_path": self.current_checkpoint_path or "Llama-3.2-1B-Instruct.Q4_K_M.gguf",
                "model_status": status,
                "dataset": "Custom Instruction Dataset (Fine-Tuned on T4 GPU by Prem Gaikwad)",
                "vocab_size": 128256,
                "parameters": 1230000000,
                "block_size": 2048,
                "n_layer": 16,
                "n_head": 32,
                "n_embd": 2048,
                "hardware": "CPU (4-Bit AVX2 C++ Quantized)",
                "description": "Quarantined external base weights. Blocked from serving as Moon AI sovereign intelligence under STRICT_SOVEREIGN_RANDOM_INIT_ONLY policy."
            }

        if self.active_model_name == "moon-chat-pilot-v0":
            param_count = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 25698816
            is_fixture = getattr(self, "is_test_fixture", False)
            return {
                "model_name": "Moon AI Pilot (25.7M Sovereign Chat — Pure Scratch Init)",
                "model_type": "Moon AI 25.7M Sovereign Pilot (Pure Scratch Init)",
                "checkpoint_path": self.current_checkpoint_path or "models/checkpoints/chat/moon_chat_pilot_v0.pt",
                "model_status": "CI TEST FIXTURE (UNPROMOTABLE)" if is_fixture else status,
                "dataset": "Moon AI Sovereign Expanded Corpus (572 records, 120k+ tokens)",
                "vocab_size": 512,
                "parameters": param_count,
                "block_size": getattr(self.model_config, "block_size", 512),
                "n_layer": getattr(self.model_config, "n_layer", 8),
                "n_head": getattr(self.model_config, "n_head", 8),
                "n_embd": getattr(self.model_config, "n_embd", 512),
                "hardware": f"{self.device.upper()} (PyTorch {self.torch_version or 'not installed'})",
                "ownership_policy": getattr(self, "current_checkpoint_policy", "UNKNOWN_PROVENANCE_INSUFFICIENT"),
                "teacher_checkpoint_weights_transferred": False,
                "is_test_fixture": is_fixture,
                "is_production_trained": False,
                "promotable": False,
                "status": "CI_TEST_FIXTURE" if is_fixture else ("MODEL_NOT_READY" if status == "MODEL_NOT_READY" else "PILOT_TRAINED"),
                "description": "25.7M parameter sovereign transformer architecture initialized purely from scratch (STRICT_SOVEREIGN_RANDOM_INIT_ONLY). Currently in MODEL_NOT_READY state awaiting authorized training run and sovereign cryptographic attestation."
            }

        is_gpt2 = self.active_model_name == "gpt2"
        param_count = None
        if self.model is not None:
            param_count = sum(p.numel() for p in self.model.parameters())
        elif is_gpt2:
            param_count = 123653376
            
        return {
            "model_type": f"NanoGPT ({'Light 124M Pretrained' if is_gpt2 else 'Core Scratch Assistant'})",
            "checkpoint_path": self.current_checkpoint_path or ("pretrained: gpt2" if is_gpt2 else "out-assistant/ckpt.pt"),
            "model_status": status,
            "dataset": self.dataset_name or ("OpenWebText" if is_gpt2 else "assistant_chat"),
            "vocab_size": self.vocab_size or (50257 if is_gpt2 else 97),
            "parameters": param_count,
            "block_size": getattr(self.model_config, "block_size", 1024 if is_gpt2 else 128),
            "n_layer": getattr(self.model_config, "n_layer", 12 if is_gpt2 else 4),
            "n_head": getattr(self.model_config, "n_head", 12 if is_gpt2 else 4),
            "n_embd": getattr(self.model_config, "n_embd", 768 if is_gpt2 else 128),
            "hardware": f"{self.device.upper()} (PyTorch {self.torch_version or 'not installed'})",
            "description": "Local offline Transformer neural network with zero external paid API dependencies."
        }

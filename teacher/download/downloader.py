"""
Moon AI — Controlled Public Model Downloader
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Implements safe, unauthenticated, resumable public model downloads for qualified local teachers.
Invariants:
1. Zero API keys, zero Hugging Face tokens, zero login credentials required.
2. Direct public HTTPS download via standard GET.
3. Strict target directory isolation: weights stored ONLY in models/teachers/checkpoints/, NEVER in models/chat/.
4. Pre-download disk space check against available filesystem capacity.
5. Content verification: rejects HTML, error pages, and non-GGUF files using magic-byte checks.
6. Full cryptographic integrity verification using SHA-256 checksums.
7. Support for resumable downloads via HTTP Range headers.
8. Zero automatic multi-GB downloads during unit tests.
"""

import os
import sys
import shutil
import hashlib
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple, Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TEACHER_DIR = os.path.join(PROJECT_ROOT, "models", "teachers", "checkpoints")
STUDENT_MODEL_DIRS = [
    os.path.join(PROJECT_ROOT, "models", "chat"),
    os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat"),
]

GGUF_MAGIC = b"GGUF"

# Qualified Teacher Checkpoint Registry
QUALIFIED_TEACHER_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-7b-instruct-q4_k_m": {
        "display_name": "Qwen 2.5 7B Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "license": "Apache-2.0",
        "commit": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        "quantization": "Q4_K_M",
        "total_size_bytes": 4683073632,
        "shards": [
            {
                "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "size_bytes": 3993201344,
                "sha256": "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
                "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            },
            {
                "filename": "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
                "size_bytes": 689872288,
                "sha256": "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a",
                "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
            }
        ]
    }
}

PRIMARY_TEACHER_KEY = "qwen2.5-7b-instruct-q4_k_m"


class DownloadVerificationError(Exception):
    """Raised when downloaded payload fails size, magic byte, or SHA-256 verification."""
    pass


class InsufficientStorageError(Exception):
    """Raised when available filesystem storage is inadequate for model weights."""
    pass


class TeacherStudentIsolationViolation(Exception):
    """Raised when destination path violates teacher/student directory separation."""
    pass


class PublicModelDownloader:
    """
    Controlled public model downloader implementing unauthenticated,
    resumable, verified downloads with directory isolation.
    """

    def __init__(
        self,
        dest_dir: Optional[str] = None,
        model_key: str = PRIMARY_TEACHER_KEY,
        chunk_size: int = 1048576,  # 1 MB chunk
        allow_network_download: bool = True
    ):
        self.dest_dir = os.path.abspath(dest_dir or DEFAULT_TEACHER_DIR)
        self.model_key = model_key
        self.chunk_size = chunk_size
        self.allow_network_download = allow_network_download
        self._validate_destination_directory()

        if model_key not in QUALIFIED_TEACHER_MODELS:
            raise ValueError(
                f"Unknown teacher model key '{model_key}'. "
                f"Supported keys: {list(QUALIFIED_TEACHER_MODELS.keys())}"
            )
        self.model_info = QUALIFIED_TEACHER_MODELS[model_key]

    def _validate_destination_directory(self):
        """Enforces physical separation between teacher and student model files."""
        norm_dest = os.path.normpath(self.dest_dir)
        for s_dir in STUDENT_MODEL_DIRS:
            norm_s_dir = os.path.normpath(os.path.abspath(s_dir))
            if norm_dest.startswith(norm_s_dir):
                raise TeacherStudentIsolationViolation(
                    f"Destination '{self.dest_dir}' violates isolation: inside student directory '{s_dir}'. "
                    "Teacher model weights must reside in models/teachers/checkpoints/."
                )

    def check_disk_space(self, buffer_gb: float = 2.0) -> Tuple[bool, int, int]:
        """
        Checks whether target drive has sufficient free space for the model.
        Returns: (has_space: bool, available_bytes: int, required_bytes: int)
        """
        os.makedirs(self.dest_dir, exist_ok=True)
        usage = shutil.disk_usage(self.dest_dir)
        available_bytes = usage.free
        required_bytes = self.model_info["total_size_bytes"] + int(buffer_gb * 1024 * 1024 * 1024)
        has_space = available_bytes >= required_bytes
        return has_space, available_bytes, required_bytes

    def verify_existing_shards(self) -> Dict[str, Any]:
        """
        Checks whether model shard files already exist and pass size/hash verification.
        """
        results = {"complete": True, "shards": []}
        for shard in self.model_info["shards"]:
            filepath = os.path.join(self.dest_dir, shard["filename"])
            shard_status = {
                "filename": shard["filename"],
                "path": filepath,
                "exists": os.path.isfile(filepath),
                "size_match": False,
                "sha256_match": False
            }
            if shard_status["exists"]:
                actual_size = os.path.getsize(filepath)
                shard_status["size_match"] = (actual_size == shard["size_bytes"])
                if shard_status["size_match"]:
                    actual_sha = self._compute_file_sha256(filepath)
                    shard_status["sha256_match"] = (actual_sha.lower() == shard["sha256"].lower())
            if not (shard_status["exists"] and shard_status["size_match"] and shard_status["sha256_match"]):
                results["complete"] = False
            results["shards"].append(shard_status)
        return results

    def verify_file_content(self, filepath: str, expected_size: int, expected_sha: str) -> Tuple[bool, str]:
        """Validates that a downloaded file is a genuine GGUF and matches expected hash."""
        if not os.path.isfile(filepath):
            return False, f"File does not exist: {filepath}"

        actual_size = os.path.getsize(filepath)
        if actual_size != expected_size:
            return False, f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes."

        # Check GGUF magic bytes
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                # Inspect for HTML error response
                f.seek(0)
                prefix = f.read(256)
                if b"<html" in prefix.lower() or b"<!doctype" in prefix.lower() or b"error" in prefix.lower():
                    return False, "File is an HTML/text error page, not a binary GGUF model."
                return False, f"Invalid GGUF magic bytes: expected {GGUF_MAGIC}, got {magic}."

        actual_sha = self._compute_file_sha256(filepath)
        if actual_sha.lower() != expected_sha.lower():
            return False, f"Cryptographic integrity failed: expected SHA-256 {expected_sha}, got {actual_sha}."

        return True, "File verified successfully."

    def download_shard(
        self,
        shard_info: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Downloads a single model shard with resume capability and zero credentials.
        """
        if not self.allow_network_download:
            raise RuntimeError("Network download is disabled in current runtime mode.")

        has_space, avail, req = self.check_disk_space()
        if not has_space:
            raise InsufficientStorageError(
                f"Insufficient disk space in '{self.dest_dir}': "
                f"available {avail / (1024**3):.2f} GB, required {req / (1024**3):.2f} GB."
            )

        target_file = os.path.join(self.dest_dir, shard_info["filename"])
        part_file = target_file + ".part"

        # Check if target already exists and is valid
        if os.path.isfile(target_file):
            valid, msg = self.verify_file_content(target_file, shard_info["size_bytes"], shard_info["sha256"])
            if valid:
                return target_file

        downloaded_bytes = 0
        if os.path.isfile(part_file):
            downloaded_bytes = os.path.getsize(part_file)
            if downloaded_bytes > shard_info["size_bytes"]:
                # Corrupted part file, restart
                os.remove(part_file)
                downloaded_bytes = 0

        req_headers = {"User-Agent": "MoonAI-OfflineDownloader/2.1.5"}
        if downloaded_bytes > 0:
            req_headers["Range"] = f"bytes={downloaded_bytes}-"

        request = urllib.request.Request(shard_info["url"], headers=req_headers)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status_code = response.getcode()
                mode = "ab" if status_code == 206 else "wb"
                if status_code != 206:
                    downloaded_bytes = 0

                with open(part_file, mode) as out_f:
                    while True:
                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded_bytes, shard_info["size_bytes"])

        except urllib.error.HTTPError as e:
            raise DownloadVerificationError(f"HTTP error during download: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            raise DownloadVerificationError(f"Network error during download: {e.reason}")

        # Rename part file to final target
        if os.path.isfile(target_file):
            os.remove(target_file)
        os.rename(part_file, target_file)

        # Validate newly completed file
        valid, msg = self.verify_file_content(target_file, shard_info["size_bytes"], shard_info["sha256"])
        if not valid:
            if os.path.isfile(target_file):
                os.remove(target_file)
            raise DownloadVerificationError(f"Verification failed after download: {msg}")

        return target_file

    def download_model(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[str]:
        """
        Downloads all shards for the qualified model.
        Returns list of verified local shard file paths.
        """
        downloaded = []
        for shard in self.model_info["shards"]:
            cb = None
            if progress_callback:
                cb = lambda cur, tot, s_name=shard["filename"]: progress_callback(s_name, cur, tot)
            path = self.download_shard(shard, progress_callback=cb)
            downloaded.append(path)
        return downloaded

    @staticmethod
    def _compute_file_sha256(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1048576):
                h.update(chunk)
        return h.hexdigest()


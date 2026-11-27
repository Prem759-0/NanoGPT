"""
datasets/chat/acquisition.py
Phase 2.7.8 Permitted Open-Source Data Acquisition & Source Provenance Registry.

Fetches or imports explicitly permitted public-domain, CC0, CC-BY, CC-BY-SA, MIT,
Apache-2.0, BSD, and PSF-2.0 educational materials cleanly without unauthorized scraping
and without AI/teacher answer generation.

All imported sources are cryptographically checksummed, recorded with full metadata in
`datasets/sources/source_registry.json`, and staged as `PENDING_REVIEW` conversion work items
until verified and transformed into genuine educational examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.chat.quality_gate import check_license_permissibility
from datasets.chat.schema import APPROVAL_STATUS_PENDING, canonicalize_category
from datasets.chat.truth_audit import _safe_relpath, file_sha256

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SOURCES_DIR = os.path.join(PROJECT_ROOT, "datasets", "sources")
DEFAULT_REGISTRY_PATH = os.path.join(DEFAULT_SOURCES_DIR, "source_registry.json")
DEFAULT_RAW_CACHE_DIR = os.path.join(DEFAULT_SOURCES_DIR, "raw_cache")
DEFAULT_STAGING_QUEUE_PATH = os.path.join(PROJECT_ROOT, "datasets", "staging", "pending_review_queue.jsonl")

REGISTRY_SCHEMA_VERSION = "moon_chat_source_registry_v1"
USER_AGENT = "MoonAI-SovereignDatasetAcquisition/2.7.8 (https://github.com/Prem759-0/NanoGPT)"


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_source_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    if not os.path.isfile(registry_path):
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "updated_at_utc": None, "sources": {}}
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "sources" not in data:
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "updated_at_utc": None, "sources": {}}
    return data


def save_source_registry(registry: Dict[str, Any], registry_path: str = DEFAULT_REGISTRY_PATH) -> None:
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    registry["schema_version"] = REGISTRY_SCHEMA_VERSION
    registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(registry_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")


def register_source_entry(
    *,
    source_id: str,
    source_url: str,
    author_or_publisher: str,
    license_name: str,
    license_url: str,
    attribution: str,
    original_file_sha256: str,
    retrieval_date: Optional[str] = None,
    derived_record_ids: Optional[Sequence[str]] = None,
    transformation_performed: str,
    reviewer: Optional[str] = None,
    approval_status: str = APPROVAL_STATUS_PENDING,
    title: Optional[str] = None,
    category: Optional[str] = None,
    source_revision: Optional[str] = None,
    raw_cache_relpath: Optional[str] = None,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> Dict[str, Any]:
    """Validates license & attribution rules and records a source entry into source_registry.json."""
    lic_ok, lic_err = check_license_permissibility(license_name, source_url)
    if not lic_ok:
        raise ValueError(f"Refusing source '{source_id}': {lic_err}")
    if not source_url or not str(source_url).strip():
        raise ValueError(f"Source '{source_id}' missing source_url.")
    if not author_or_publisher or len(str(author_or_publisher).strip()) < 2:
        raise ValueError(f"Source '{source_id}' missing author_or_publisher.")
    if not attribution or len(str(attribution).strip()) < 3:
        raise ValueError(f"Source '{source_id}' missing explicit attribution text.")
    if not original_file_sha256 or len(original_file_sha256) != 64:
        raise ValueError(f"Source '{source_id}' requires valid 64-hex original_file_sha256.")

    registry = load_source_registry(registry_path)
    existing = registry["sources"].get(source_id) or {}
    merged_ids = sorted(set(existing.get("derived_record_ids") or []) | set(derived_record_ids or []))

    entry = {
        "source_id": source_id,
        "title": title or existing.get("title") or source_id,
        "category": canonicalize_category(category) if category else existing.get("category"),
        "source_url": source_url,
        "source_revision": source_revision or existing.get("source_revision"),
        "author_or_publisher": author_or_publisher,
        "license": license_name,
        "license_url": license_url,
        "share_alike_required": "SA" in license_name.upper(),
        "attribution": attribution,
        "original_file_sha256": original_file_sha256,
        "retrieval_date": retrieval_date or datetime.now(timezone.utc).isoformat(),
        "derived_record_ids": merged_ids,
        "transformation_performed": transformation_performed,
        "reviewer": reviewer,
        "approval_status": approval_status,
        "raw_cache_relpath": raw_cache_relpath or existing.get("raw_cache_relpath"),
    }
    registry["sources"][source_id] = entry
    save_source_registry(registry, registry_path)
    return entry


def _stage_pending_conversion_draft(
    entry: Dict[str, Any],
    summary_excerpt: str,
    suggested_question: str,
    queue_path: str = DEFAULT_STAGING_QUEUE_PATH,
) -> Optional[str]:
    """
    Appends a clearly marked PENDING_REVIEW conversion work-item draft to the staging queue
    if not already queued. Never auto-populates fabricated answers or copies articles as final Q&A.
    """
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)
    record_id = f"src_draft_{entry['source_id']}"
    existing_ids = set()
    if os.path.isfile(queue_path):
        with open(queue_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        continue
    if record_id in existing_ids:
        return None

    draft_item = {
        "id": record_id,
        "record_kind": "SOURCE_CONVERSION_DRAFT",
        "category": entry.get("category") or "general_knowledge",
        "source": entry["title"],
        "source_id": entry["source_id"],
        "source_url": entry["source_url"],
        "source_revision": entry.get("source_revision"),
        "author_or_publisher": entry["author_or_publisher"],
        "license": entry["license"],
        "license_url": entry["license_url"],
        "attribution": entry["attribution"],
        "provenance_type": "SOURCE_DERIVED_EDUCATIONAL_SUMMARY",
        "is_modified": True,
        "original_file_sha256": entry["original_file_sha256"],
        "retrieval_date": entry["retrieval_date"],
        "transformation_performed": entry["transformation_performed"],
        "modification_summary": "Awaiting reviewer educational Q&A synthesis from verified public source metadata.",
        "reviewer": None,
        "approval_status": APPROVAL_STATUS_PENDING,
        "needs_original_human_summary": True,
        "source_reference_extract": summary_excerpt[:600],
        "messages": [
            {"role": "user", "content": suggested_question},
            {"role": "assistant", "content": ""},
        ],
    }

    with open(queue_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(draft_item, ensure_ascii=False) + "\n")
    return record_id


def fetch_wikipedia_source(
    page_title: str,
    category: str,
    suggested_question: str,
    *,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    raw_cache_dir: str = DEFAULT_RAW_CACHE_DIR,
    queue_path: str = DEFAULT_STAGING_QUEUE_PATH,
    refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetches official Wikimedia REST API metadata/summary under CC-BY-SA-4.0 with exact revid permalink,
    caches raw bytes with SHA-256 verification, registers full provenance, and stages a
    PENDING_REVIEW source-conversion draft (never copied directly as an approved chat answer).
    """
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", page_title.strip()).strip("_").lower()
    source_id = f"wikipedia_{slug}"
    registry = load_source_registry(registry_path)

    if not refresh and source_id in registry["sources"]:
        existing = registry["sources"][source_id]
        return {"status": "CACHED", "source_entry": existing, "staged_id": None}

    encoded_title = urllib.parse.quote(page_title.replace(" ", "_"), safe="()_")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})

    with urllib.request.urlopen(req, timeout=15) as resp:
        raw_bytes = resp.read()

    raw_sha = _bytes_sha256(raw_bytes)
    data = json.loads(raw_bytes.decode("utf-8"))
    if data.get("type") == "disambiguation":
        raise ValueError(f"Wikipedia title '{page_title}' resolved to a disambiguation page; specify exact article.")

    revision_id = str(data.get("revision") or "")
    timestamp = str(data.get("timestamp") or datetime.now(timezone.utc).isoformat())
    canonical_title = str(data.get("title") or page_title)
    page_url = (data.get("content_urls") or {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{encoded_title}"
    if revision_id:
        permalink = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&oldid={revision_id}"
        rev_label = f"Wikipedia revision {revision_id} ({timestamp}, CC-BY-SA 4.0)"
    else:
        permalink = page_url
        rev_label = f"Wikipedia REST summary ({timestamp}, CC-BY-SA 4.0)"

    os.makedirs(raw_cache_dir, exist_ok=True)
    raw_file_path = os.path.join(raw_cache_dir, f"{source_id}_{raw_sha[:12]}.json")
    with open(raw_file_path, "wb") as rf:
        rf.write(raw_bytes)

    extract = " ".join(str(data.get("extract") or "").split())
    draft_id = f"src_draft_{source_id}"

    entry = register_source_entry(
        source_id=source_id,
        title=f"Wikipedia: {canonical_title}",
        category=category,
        source_url=permalink,
        source_revision=rev_label,
        author_or_publisher="Wikipedia contributors / Wikimedia Foundation",
        license_name="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution=f"Wikipedia contributors, '{canonical_title}' ({permalink}), licensed under CC-BY-SA-4.0",
        original_file_sha256=raw_sha,
        derived_record_ids=[draft_id],
        transformation_performed=(
            "Fetched official Wikimedia REST v1 page summary JSON; staged as PENDING_REVIEW "
            "conversion draft awaiting human-authored original educational Q&A synthesis."
        ),
        reviewer=None,
        approval_status=APPROVAL_STATUS_PENDING,
        raw_cache_relpath=_safe_relpath(raw_file_path, PROJECT_ROOT),
        registry_path=registry_path,
    )

    staged = _stage_pending_conversion_draft(
        entry=entry,
        summary_excerpt=extract,
        suggested_question=suggested_question,
        queue_path=queue_path,
    )
    return {"status": "FETCHED", "source_entry": entry, "staged_id": staged}


def register_local_permitted_source(
    *,
    file_path: str,
    source_id: str,
    source_url: str,
    title: str,
    category: str,
    author_or_publisher: str,
    license_name: str,
    license_url: str,
    attribution: str,
    transformation_performed: str,
    suggested_question: Optional[str] = None,
    summary_excerpt: str = "",
    registry_path: str = DEFAULT_REGISTRY_PATH,
    queue_path: str = DEFAULT_STAGING_QUEUE_PATH,
) -> Dict[str, Any]:
    """Registers a local open-licensed source file using exact disk SHA-256 and stages optional review draft."""
    abs_path = os.path.abspath(file_path)
    sha = file_sha256(abs_path)
    if not sha:
        raise FileNotFoundError(f"Source file not found: {abs_path}")

    draft_id = f"src_draft_{source_id}"
    entry = register_source_entry(
        source_id=source_id,
        title=title,
        category=category,
        source_url=source_url,
        author_or_publisher=author_or_publisher,
        license_name=license_name,
        license_url=license_url,
        attribution=attribution,
        original_file_sha256=sha,
        derived_record_ids=[draft_id] if suggested_question else [],
        transformation_performed=transformation_performed,
        reviewer=None,
        approval_status=APPROVAL_STATUS_PENDING,
        registry_path=registry_path,
    )
    staged = None
    if suggested_question:
        staged = _stage_pending_conversion_draft(
            entry=entry,
            summary_excerpt=summary_excerpt,
            suggested_question=suggested_question,
            queue_path=queue_path,
        )
    return {"status": "REGISTERED", "source_entry": entry, "staged_id": staged}


# Curated open-licensed Wikimedia topics spanning Phase 2.7.8 categories for staged source acquisition.
OPEN_EDUCATIONAL_TOPICS: List[Tuple[str, str, str]] = [
    ("Solar_System", "general_knowledge", "Explain how the planets of our Solar System are structured and ordered."),
    ("Photosynthesis", "basic_science", "How does photosynthesis convert light energy into chemical energy in plants?"),
    ("Periodic_table", "basic_science", "What is the periodic table and how are elements grouped by properties?"),
    ("Newton%27s_laws_of_motion", "basic_science", "Explain Newton's three laws of motion with clear everyday examples."),
    ("Linear_algebra", "mathematics", "What is linear algebra and why are vectors and matrices fundamental in computing?"),
    ("Bayes%27_theorem", "mathematics", "Explain Bayes' theorem intuitively and write out what each term means."),
    ("Algorithm", "programming_fundamentals", "What defines an algorithm in computer science, and how do we evaluate efficiency?"),
    ("Unit_testing", "programming_fundamentals", "Why is automated unit testing important in software engineering?"),
    ("Python_(programming_language)", "python", "What are the key design philosophies and common use cases of Python?"),
    ("JavaScript", "javascript", "What role does JavaScript play in modern web applications and how does it execute?"),
    ("HTML", "html_css", "What is HTML and how do semantic elements structure a web document?"),
    ("CSS", "html_css", "How does CSS control visual layout and typography separately from HTML content?"),
    ("HTTP", "web_development", "Explain the HTTP request-response cycle and common status codes used on the web."),
    ("REST", "web_development", "What are the core architectural constraints of a RESTful web API?"),
    ("Operating_system", "computer_science", "What are the main responsibilities of an operating system kernel?"),
    ("Machine_learning", "ai_ml_fundamentals", "Explain the difference between supervised learning and unsupervised learning."),
    ("Backpropagation", "ai_ml_fundamentals", "How does backpropagation compute gradients to train a neural network?"),
    ("Transformer_(deep_learning_architecture)", "ai_ml_fundamentals", "What problem did the Transformer architecture solve compared to recurrent networks?"),
    ("Deductive_reasoning", "reasoning_problem_solving", "Compare deductive reasoning and inductive reasoning with concise examples."),
    ("Divide-and-conquer_algorithm", "reasoning_problem_solving", "Explain how the divide-and-conquer problem-solving strategy breaks down complex problems."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Moon Chat Permitted Source Acquisition & Registry CLI")
    parser.add_argument("--fetch-open-batch", action="store_true", help="Fetch curated CC-BY-SA-4.0 Wikipedia topic references into PENDING_REVIEW staging.")
    parser.add_argument("--wiki-title", type=str, default=None, help="Fetch a single Wikipedia article summary by title.")
    parser.add_argument("--category", type=str, default="general_knowledge", help="Category slug for --wiki-title.")
    parser.add_argument("--prompt", type=str, default=None, help="Study prompt for conversion draft.")
    parser.add_argument("--list-sources", action="store_true", help="List registered sources.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch even if source_id already exists in registry.")
    args = parser.parse_args()

    if args.list_sources:
        reg = load_source_registry()
        print(f"Registered sources: {len(reg['sources'])}")
        for sid, entry in sorted(reg["sources"].items()):
            print(f"  - {sid} | {entry['license']} | {entry['approval_status']} | sha256={entry['original_file_sha256'][:12]}...")
        return

    if args.wiki_title:
        q = args.prompt or f"Provide an educational summary of {args.wiki_title.replace('_', ' ')}."
        res = fetch_wikipedia_source(args.wiki_title, args.category, q, refresh=args.refresh)
        print(f"[{res['status']}] {res['source_entry']['source_id']} -> {res['source_entry']['approval_status']}")
        return

    if args.fetch_open_batch:
        fetched = 0
        cached = 0
        errors = 0
        for title, cat, prompt in OPEN_EDUCATIONAL_TOPICS:
            try:
                res = fetch_wikipedia_source(urllib.parse.unquote(title), cat, prompt, refresh=args.refresh)
                if res["status"] == "FETCHED":
                    fetched += 1
                else:
                    cached += 1
                print(f"  [{res['status']}] {res['source_entry']['source_id']} ({cat})")
            except Exception as exc:
                errors += 1
                print(f"  [ERROR] {title}: {exc}")
        print(f"[Acquisition] fetched={fetched}, cached={cached}, errors={errors}")


if __name__ == "__main__":
    main()

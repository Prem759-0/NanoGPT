"""
datasets/chat/contribute.py
Phase 2.7.8 Owner-Authored Contribution CLI & Validated Staging Pipeline.

Allows the project owner to:
  1. Generate empty clearly labelled templates across all 13 categories (`--init-templates`).
  2. Submit an original Q&A directly via CLI arguments or interactive prompts (`--submit` / `--interactive`).
  3. Validate and import JSON/JSONL contribution files with preview (`--import FILE --dry-run / --commit`).

Safety invariant:
  Submitted records are NEVER automatically approved for production training.
  All accepted submissions enter `datasets/staging/pending_review_queue.jsonl`
  strictly marked `approval_status = "PENDING_REVIEW"`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.chat.quality_gate import validate_candidate_record
from datasets.chat.schema import (
    APPROVAL_STATUS_PENDING,
    PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
    SUPPORTED_CHAT_CATEGORIES,
    canonicalize_category,
    compute_conversation_sha256,
)
from datasets.chat.tokenizer import SovereignTokenizer
from datasets.chat.truth_audit import (
    DEFAULT_TOKENIZER_PATH,
    extract_first_assistant_reply,
    extract_first_user_prompt,
    file_sha256,
    iter_jsonl_records,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONTRIBUTIONS_DIR = os.path.join(PROJECT_ROOT, "datasets", "contributions")
DEFAULT_TEMPLATES_PATH = os.path.join(DEFAULT_CONTRIBUTIONS_DIR, "owner_templates.jsonl")
DEFAULT_EXAMPLE_PATH = os.path.join(DEFAULT_CONTRIBUTIONS_DIR, "example_owner_contribution.json")
DEFAULT_PENDING_QUEUE_PATH = os.path.join(PROJECT_ROOT, "datasets", "staging", "pending_review_queue.jsonl")
DEFAULT_APPROVED_MANIFESTS = (
    os.path.join(PROJECT_ROOT, "datasets", "manifests", "prod_chat_train.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "manifests", "prod_chat_val.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "manifests", "prod_chat_test.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "manifests", "curated_approved_records.jsonl"),
)


CATEGORY_AUTHORING_GUIDANCE: Dict[str, Dict[str, str]] = {
    "general_knowledge": {
        "label": "General knowledge",
        "guidance": "Clear, accurate factual explanations of real-world concepts, history, geography, or everyday phenomena written in original prose.",
    },
    "basic_science": {
        "label": "Basic science",
        "guidance": "Explain core physics, chemistry, biology, or earth science principles using step-by-step intuition and everyday examples.",
    },
    "mathematics": {
        "label": "Mathematics",
        "guidance": "Walk through algebra, calculus, probability, or discrete math step-by-step, showing assumptions and why each step follows.",
    },
    "programming_fundamentals": {
        "label": "Programming fundamentals",
        "guidance": "Cover control flow, functions, recursion, testing, debugging, data representations, and clean software design principles.",
    },
    "python": {
        "label": "Python",
        "guidance": "Include valid, syntactically checked ```python code fences with idiomatic Python 3 patterns, edge-case handling, and clear complexity notes.",
    },
    "javascript": {
        "label": "JavaScript",
        "guidance": "Explain modern ES6+ features, async/await, closures, array methods, and DOM/event semantics with clean executable code examples.",
    },
    "html_css": {
        "label": "HTML/CSS",
        "guidance": "Demonstrate semantic HTML5 markup, accessibility attributes, Flexbox/Grid layouts, and responsive CSS patterns.",
    },
    "web_development": {
        "label": "Web development",
        "guidance": "Cover HTTP, REST APIs, authentication security, browser rendering, caching, and backend-frontend architecture trade-offs.",
    },
    "computer_science": {
        "label": "Computer science",
        "guidance": "Explain algorithms, data structures, time/space complexity, operating systems, networking, databases, and distributed systems.",
    },
    "ai_ml_fundamentals": {
        "label": "AI/ML fundamentals",
        "guidance": "Explain tokenization, embeddings, attention, optimization, generalization, evaluation, and Moon AI first-principles design.",
    },
    "instruction_following": {
        "label": "Instruction following",
        "guidance": "Practice exact constraint satisfaction: bullet counts, valid ```json formatting, tone shifts, comparison tables, or structured outlines.",
    },
    "reasoning_problem_solving": {
        "label": "Reasoning and problem solving",
        "guidance": "Decompose multi-step logic, trade-off analysis, root-cause debugging, and deductive reasoning explicitly before stating conclusions.",
    },
    "helpful_conversational": {
        "label": "Helpful conversational responses",
        "guidance": "Natural, concise, honest, non-sycophantic multi-turn or single-turn dialogues that clarify ambiguity and stay factual.",
    },
}


def build_empty_owner_templates() -> List[Dict[str, Any]]:
    """
    Generates 13 empty, clearly labelled owner-authoring templates (1 per target category).
    Contains ZERO fabricated questions or answers.
    """
    templates: List[Dict[str, Any]] = []
    for idx, (cat_slug, cat_label) in enumerate(SUPPORTED_CHAT_CATEGORIES.items(), start=1):
        guide = CATEGORY_AUTHORING_GUIDANCE.get(cat_slug, {}).get("guidance", "")
        templates.append({
            "id": f"owner_template_{idx:02d}_{cat_slug}",
            "category": cat_slug,
            "category_label": cat_label,
            "source": "Owner Original Authored Educational Q&A",
            "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/datasets/contributions",
            "source_revision": "owner-draft-template",
            "license": "CustomSovereign",
            "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
            "attribution": "",
            "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
            "approval_status": APPROVAL_STATUS_PENDING,
            "_template_status": "EMPTY_TEMPLATE_FOR_OWNER_AUTHORING",
            "_category_guidance": guide,
            "_instructions": (
                f"Category '{cat_label}' ({cat_slug}): {guide} "
                "Fill in attribution (your name), user content, and your original human-written "
                "educational assistant response. Leave approval_status as PENDING_REVIEW and submit via "
                "`python -m datasets.chat.contribute --import-file <file>`."
            ),
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
            ],
        })
    return templates


def write_empty_owner_templates(output_path: str = DEFAULT_TEMPLATES_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    templates = build_empty_owner_templates()
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for item in templates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path


def write_contribution_example_file(output_path: str = DEFAULT_EXAMPLE_PATH) -> str:
    """Writes a clearly marked empty structural example schema guide (no fabricated answers)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    example_doc = {
        "_schema_notes": (
            "Copy this structure (single JSON object, JSON list, or one-JSON-per-line JSONL) "
            "and fill in your own original prompt and answer. Submitted entries always enter "
            "PENDING_REVIEW and require explicit verification before promotion."
        ),
        "id": "owner_contrib_example_template",
        "category": "python",
        "attribution": "",
        "license": "CustomSovereign",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "source": "Owner Original Contribution",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/datasets/contributions",
        "approval_status": APPROVAL_STATUS_PENDING,
        "prompt": "",
        "answer": "",
    }
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(example_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_path


def normalize_contribution_record(raw_rec: Dict[str, Any], index_hint: int = 1, source_file_sha: str = "") -> Dict[str, Any]:
    """
    Normalizes either shorthand (`prompt`/`answer`) or full ChatML (`messages`) owner contribution
    into canonical schema while preserving `approval_status = PENDING_REVIEW` strictly.
    """
    if not isinstance(raw_rec, dict):
        return raw_rec

    rec = {k: v for k, v in raw_rec.items() if not str(k).startswith("_")}
    if "messages" not in rec or not rec.get("messages"):
        prompt = rec.pop("prompt", None) or rec.pop("question", None) or ""
        answer = rec.pop("answer", None) or rec.pop("response", None) or ""
        turns: List[Dict[str, str]] = []
        if rec.get("system"):
            turns.append({"role": "system", "content": str(rec.pop("system"))})
        if prompt or answer:
            turns.append({"role": "user", "content": str(prompt)})
            turns.append({"role": "assistant", "content": str(answer)})
        rec["messages"] = turns

    cat = canonicalize_category(rec.get("category"))
    rec["category"] = cat
    rec.setdefault("provenance_type", PROVENANCE_ORIGINAL_SELF_AUTHORED_QA)
    rec.setdefault("license", "CustomSovereign")
    if rec["license"] == "CustomSovereign":
        rec.setdefault("license_url", "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE")
    rec.setdefault("source", "Owner Original Contribution")
    rec.setdefault("source_url", "https://github.com/Prem759-0/NanoGPT/blob/main/datasets/contributions")
    rec.setdefault("source_revision", "owner-contribution-v1")
    rec.setdefault("is_modified", False)
    rec.setdefault("is_dev_fixture", False)
    rec.setdefault("is_synthetic", False)
    rec.setdefault("transformation_performed", "Original human-authored Q&A submitted via contribution CLI.")
    rec.setdefault("submitted_at_utc", datetime.now(timezone.utc).isoformat())
    if source_file_sha:
        rec["source_file_sha256"] = source_file_sha

    # Security rule: submitted records can NEVER self-approve. Force PENDING_REVIEW unconditionally.
    rec["approval_status"] = APPROVAL_STATUS_PENDING
    rec["reviewer"] = None

    msgs = rec.get("messages") or []
    conv_sha = compute_conversation_sha256(msgs) if isinstance(msgs, list) and msgs else ""
    if conv_sha:
        rec["sha256"] = conv_sha
    if not rec.get("id"):
        short_h = (conv_sha or hashlib.sha256(str(index_hint).encode("utf-8")).hexdigest())[:10]
        rec["id"] = f"owner_{cat}_{short_h}"

    return rec


def _collect_existing_fingerprints(paths: Sequence[str]) -> Tuple[set[str], set[str], set[str], set[str]]:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_prompts: set[str] = set()
    seen_replies: set[str] = set()
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        for _, rec, err in iter_jsonl_records(path):
            if err is not None or not isinstance(rec, dict):
                continue
            if rec.get("id"):
                seen_ids.add(str(rec["id"]))
            msgs = rec.get("messages") or []
            if isinstance(msgs, list) and msgs:
                csha = str(rec.get("sha256") or "") or compute_conversation_sha256(msgs)
                seen_hashes.add(csha)
                p = extract_first_user_prompt(msgs).lower()
                a = extract_first_assistant_reply(msgs).lower()
                if p:
                    seen_prompts.add(p)
                if a:
                    seen_replies.add(a)
    return seen_ids, seen_hashes, seen_prompts, seen_replies


def _load_contribution_input(input_path: str) -> Tuple[List[Tuple[int, Any]], List[str]]:
    """Loads JSON (dict or list) or streams JSONL rows, reporting exact parse errors."""
    items: List[Tuple[int, Any]] = []
    parse_errors: List[str] = []
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for idx, row in enumerate(data, start=1):
                    items.append((idx, row))
            elif isinstance(data, dict):
                items.append((1, data))
            else:
                parse_errors.append(f"[{input_path}] JSON root must be object or array.")
        except Exception as exc:
            parse_errors.append(f"[{input_path}] JSON decode error: {exc}")
    else:
        for line_no, rec, err in iter_jsonl_records(input_path):
            if err is not None:
                parse_errors.append(err)
            else:
                items.append((line_no, rec))
    return items, parse_errors


def import_contributions(
    input_path: str,
    *,
    queue_path: str = DEFAULT_PENDING_QUEUE_PATH,
    approved_paths: Sequence[str] = DEFAULT_APPROVED_MANIFESTS,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Validates owner/source contributions, detects duplicates against approved & pending queues,
    previews accepted vs rejected records, and writes accepted items strictly as PENDING_REVIEW.
    """
    abs_input = os.path.abspath(input_path)
    if not os.path.isfile(abs_input):
        raise FileNotFoundError(f"Contribution input file not found: {abs_input}")

    input_sha = file_sha256(abs_input) or ""
    tok = SovereignTokenizer.load(tokenizer_path) if os.path.isfile(tokenizer_path) else None
    raw_items, parse_errors = _load_contribution_input(abs_input)

    seen_ids, seen_hashes, seen_prompts, seen_replies = _collect_existing_fingerprints(
        list(approved_paths) + [queue_path]
    )

    staged_preview: List[Dict[str, Any]] = []
    rejected_preview: List[Dict[str, Any]] = []
    for perr in parse_errors:
        rejected_preview.append({"line_no": 0, "id": None, "reasons": [perr]})

    for idx_or_line, raw_rec in raw_items:
        norm_rec = normalize_contribution_record(raw_rec, index_hint=idx_or_line, source_file_sha=input_sha)
        ok, diag, errors = validate_candidate_record(
            norm_rec,
            source_file=os.path.basename(abs_input),
            line_no=idx_or_line,
            require_approved=False,
            tokenizer=tok,
        )
        if not ok:
            rejected_preview.append({"line_no": idx_or_line, "id": diag.get("id"), "reasons": errors})
            continue

        rec_id = str(norm_rec["id"])
        csha = diag["conversation_sha256"]
        p_norm = diag["prompt_norm"]
        a_norm = diag["reply_norm"]

        dup_reasons: List[str] = []
        if rec_id in seen_ids:
            dup_reasons.append(f"Duplicate id '{rec_id}' already exists.")
        if csha and csha in seen_hashes:
            dup_reasons.append(f"Duplicate conversation SHA-256 '{csha[:12]}...'.")
        if p_norm and p_norm in seen_prompts:
            dup_reasons.append("Duplicate normalized user prompt.")
        if a_norm and a_norm in seen_replies:
            dup_reasons.append("Duplicate normalized assistant response.")

        if dup_reasons:
            rejected_preview.append({"line_no": idx_or_line, "id": rec_id, "reasons": dup_reasons})
            continue

        seen_ids.add(rec_id)
        if csha:
            seen_hashes.add(csha)
        if p_norm:
            seen_prompts.add(p_norm)
        if a_norm:
            seen_replies.add(a_norm)

        staged_preview.append(norm_rec)

    if not dry_run and staged_preview:
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)
        with open(queue_path, "a", encoding="utf-8", newline="\n") as qf:
            for item in staged_preview:
                qf.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "dry_run": dry_run,
        "input_file": abs_input,
        "input_sha256": input_sha,
        "queue_path": queue_path,
        "submitted_total": len(raw_items) + len(parse_errors),
        "accepted_to_pending_review_count": len(staged_preview),
        "rejected_count": len(rejected_preview),
        "approval_status_assigned": APPROVAL_STATUS_PENDING,
        "accepted_records": staged_preview,
        "rejected_records": rejected_preview,
    }


def submit_single_qa(
    *,
    category: str,
    prompt: str,
    answer: str,
    attribution: str,
    record_id: Optional[str] = None,
    source_url: str = "https://github.com/Prem759-0/NanoGPT/blob/main/datasets/contributions",
    queue_path: str = DEFAULT_PENDING_QUEUE_PATH,
    approved_paths: Sequence[str] = DEFAULT_APPROVED_MANIFESTS,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Submits a single owner-written Q&A into PENDING_REVIEW after strict quality/duplicate checks."""
    raw = {
        "id": record_id,
        "category": category,
        "attribution": attribution,
        "source_url": source_url,
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "license": "CustomSovereign",
        "prompt": prompt,
        "answer": answer,
    }
    norm = normalize_contribution_record(raw)
    tok = SovereignTokenizer.load(tokenizer_path) if os.path.isfile(tokenizer_path) else None
    ok, diag, errors = validate_candidate_record(
        norm, source_file="<cli_submit>", line_no=1, require_approved=False, tokenizer=tok
    )
    if not ok:
        return {"accepted": False, "reasons": errors, "record": norm}

    seen_ids, seen_hashes, seen_prompts, seen_replies = _collect_existing_fingerprints(
        list(approved_paths) + [queue_path]
    )
    dup_reasons: List[str] = []
    if norm["id"] in seen_ids:
        dup_reasons.append(f"Duplicate id '{norm['id']}'.")
    if diag["conversation_sha256"] in seen_hashes:
        dup_reasons.append("Duplicate conversation SHA-256.")
    if diag["prompt_norm"] in seen_prompts:
        dup_reasons.append("Duplicate user prompt.")
    if diag["reply_norm"] in seen_replies:
        dup_reasons.append("Duplicate assistant reply.")
    if dup_reasons:
        return {"accepted": False, "reasons": dup_reasons, "record": norm}

    if not dry_run:
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)
        with open(queue_path, "a", encoding="utf-8", newline="\n") as qf:
            qf.write(json.dumps(norm, ensure_ascii=False) + "\n")

    return {
        "accepted": True,
        "approval_status": APPROVAL_STATUS_PENDING,
        "record": norm,
        "diagnostics": diag,
    }


def list_category_guidance(progress_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns all 13 target categories with clear authoring guidance and current approved/pending counts."""
    approved_counts: Dict[str, int] = {}
    pending_counts: Dict[str, int] = {}
    if progress_path and os.path.isfile(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            approved_counts = prog.get("current_approved_production", {}).get("category_13_distribution", {}) or {}
            pending_counts = prog.get("review_queue", {}).get("pending_category_13_distribution", {}) or {}
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    for idx, (slug, default_label) in enumerate(SUPPORTED_CHAT_CATEGORIES.items(), start=1):
        meta = CATEGORY_AUTHORING_GUIDANCE.get(slug, {})
        rows.append({
            "index": idx,
            "slug": slug,
            "label": meta.get("label", default_label),
            "guidance": meta.get("guidance", ""),
            "approved_count": int(approved_counts.get(slug, 0)),
            "pending_count": int(pending_counts.get(slug, 0)),
        })
    return rows


def interactive_submit(
    *,
    queue_path: str = DEFAULT_PENDING_QUEUE_PATH,
    approved_paths: Sequence[str] = DEFAULT_APPROVED_MANIFESTS,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    input_fn: Any = input,
    print_fn: Any = print,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Guided interactive owner authoring CLI across all 13 Moon Chat categories.
    Supports multi-line code/explanation entry and always stages strictly as PENDING_REVIEW.
    """
    cats = list_category_guidance(os.path.join(PROJECT_ROOT, "data", "reports", "moon_chat_dataset_progress.json"))
    print_fn("=== Moon Chat Owner Authoring Session (Always staged as PENDING_REVIEW) ===")
    for row in cats:
        print_fn(
            f"  [{row['index']:2d}] {row['label']} ({row['slug']}) "
            f"[approved={row['approved_count']}, pending={row['pending_count']}]"
        )
        print_fn(f"       Guide: {row['guidance']}")

    raw_choice = str(input_fn("\nSelect category number (1-13) or slug: ")).strip()
    chosen_slug: Optional[str] = None
    if raw_choice.isdigit():
        idx_num = int(raw_choice)
        if 1 <= idx_num <= len(cats):
            chosen_slug = cats[idx_num - 1]["slug"]
    if chosen_slug is None:
        chosen_slug = canonicalize_category(raw_choice)
    if chosen_slug not in SUPPORTED_CHAT_CATEGORIES:
        return {"accepted": False, "reasons": [f"Unknown category selection '{raw_choice}'."]}

    author = str(input_fn("Author/attribution name [Prem Gaikwad (Moon AI)]: ")).strip() or "Prem Gaikwad (Moon AI Chief ML Systems Architect)"
    prompt = str(input_fn("User question/prompt: ")).strip()
    print_fn("Enter original human-written educational assistant response (end with a line containing only 'END'):")
    answer_lines: List[str] = []
    while True:
        try:
            line = input_fn("")
        except EOFError:
            break
        if str(line).strip() == "END":
            break
        answer_lines.append(str(line))
    answer = "\n".join(answer_lines).strip()

    res = submit_single_qa(
        category=chosen_slug,
        prompt=prompt,
        answer=answer,
        attribution=author,
        queue_path=queue_path,
        approved_paths=approved_paths,
        tokenizer_path=tokenizer_path,
        dry_run=dry_run,
    )
    if res["accepted"]:
        tok_stats = res["diagnostics"].get("token_stats") or {}
        print_fn(
            f"\n[Contribute] Staged '{res['record']['id']}' ({chosen_slug}) as {res['approval_status']} "
            f"(supervised_tokens={tok_stats.get('supervised_tokens', 'n/a')})."
        )
        print_fn(
            f"  Next step after reviewing: python -m datasets.chat.review_queue "
            f"--approve {res['record']['id']} --reviewer \"{author}\""
        )
    else:
        print_fn(f"\n[Contribute] Submission REJECTED: {res['reasons']}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Moon Chat Owner-Authored Contribution CLI")
    parser.add_argument("--init-templates", nargs="?", const=DEFAULT_TEMPLATES_PATH, default=None, help="Write 13 empty category templates.")
    parser.add_argument("--init-example", nargs="?", const=DEFAULT_EXAMPLE_PATH, default=None, help="Write empty structural example file.")
    parser.add_argument("--categories", action="store_true", help="List all 13 target categories with authoring guidance and counts.")
    parser.add_argument("--interactive", action="store_true", help="Interactive multi-line Q&A authoring wizard (stages PENDING_REVIEW).")
    parser.add_argument("--import-file", dest="import_file", type=str, default=None, help="JSON or JSONL file to validate and stage as PENDING_REVIEW.")
    parser.add_argument("--submit", action="store_true", help="Submit a single owner-authored Q&A directly via flags.")
    parser.add_argument("--category", type=str, default="general_knowledge")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--answer", type=str, default="")
    parser.add_argument("--author", type=str, default="")
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without modifying pending queue.")
    args = parser.parse_args()

    if args.categories:
        prog_path = os.path.join(PROJECT_ROOT, "data", "reports", "moon_chat_dataset_progress.json")
        print("Supported 13 Moon Chat Categories & Authoring Guidance:")
        for row in list_category_guidance(prog_path):
            print(
                f"  {row['index']:2d}. {row['label']} ({row['slug']}) "
                f"[approved={row['approved_count']}, pending={row['pending_count']}]\n"
                f"      -> {row['guidance']}"
            )
        return

    if args.interactive:
        interactive_submit(dry_run=args.dry_run)
        return

    if args.init_templates:
        p = write_empty_owner_templates(args.init_templates)
        print(f"[Contribute] Wrote 13 empty templates to: {p}")
        return

    if args.init_example:
        p = write_contribution_example_file(args.init_example)
        print(f"[Contribute] Wrote contribution schema example to: {p}")
        return

    if args.import_file:
        res = import_contributions(args.import_file, dry_run=args.dry_run)
        print(
            f"[Contribute] dry_run={res['dry_run']} | accepted_to_PENDING_REVIEW="
            f"{res['accepted_to_pending_review_count']} | rejected={res['rejected_count']}"
        )
        for rej in res["rejected_records"][:10]:
            print(f"  [REJECTED line={rej['line_no']} id={rej['id']}] {'; '.join(rej['reasons'])}")
        return

    if args.submit:
        res = submit_single_qa(
            category=args.category,
            prompt=args.prompt,
            answer=args.answer,
            attribution=args.author,
            record_id=args.id,
            dry_run=args.dry_run,
        )
        if res["accepted"]:
            print(f"[Contribute] Staged '{res['record']['id']}' with status={res['approval_status']}.")
        else:
            print(f"[Contribute] Submission REJECTED: {res['reasons']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

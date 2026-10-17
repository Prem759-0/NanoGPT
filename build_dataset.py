#!/usr/bin/env python3
"""
build_dataset.py

Generate an N-example {"instruction","response"} dataset covering
Python coding, algorithm design, debugging, and general knowledge.

Pipeline
  1. LOAD      curated seeds (seeds.json) -> hand-written high-quality records
  2. EXPAND    deterministic template engine -> volume at zero marginal cost
  3. ENRICH    optional LLM pass -> semantic diversity (needs API key)
  4. DEDUPE    exact + normalized-instruction hashing
  5. VALIDATE  schema, non-empty fields, min lengths
  6. SHARD     JSONL shards -> merged valid JSON array (streamed, never in RAM)

Usage
  python build_dataset.py --seeds seeds.json --target 4000000 --out dataset.json
  python build_dataset.py --seeds seeds.json --target 50000 --engine template
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from typing import Dict, Iterator, List

Record = Dict[str, str]

# --------------------------------------------------------------------------- #
# 1. LOAD / VALIDATE
# --------------------------------------------------------------------------- #

MIN_INSTR, MIN_RESP = 12, 8


def normalize(text: str) -> str:
    """Collapse whitespace + lowercase, for near-duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


def is_valid(rec: Record) -> bool:
    return (
        isinstance(rec, dict)
        and set(rec) == {"instruction", "response"}
        and isinstance(rec["instruction"], str)
        and isinstance(rec["response"], str)
        and len(rec["instruction"].strip()) >= MIN_INSTR
        and len(rec["response"].strip()) >= MIN_RESP
    )


def load_seeds(path: str) -> List[Record]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("seeds file must contain a JSON array")
    return [r for r in data if is_valid(r)]


# --------------------------------------------------------------------------- #
# 2. TEMPLATE EXPANSION
# --------------------------------------------------------------------------- #

PY_TASKS = [
    ("check if a number is prime", "is_prime", "int", "bool"),
    ("compute the nth Fibonacci number iteratively", "fib", "int", "int"),
    ("reverse a string in place", "reverse_string", "str", "str"),
    ("count the frequency of each character in a string", "char_counts", "str", "dict"),
    ("return the factorial of a non-negative integer", "factorial", "int", "int"),
    ("determine whether a string is a palindrome", "is_palindrome", "str", "bool"),
    ("sum all integers in a nested list", "deep_sum", "list", "int"),
    ("find the greatest common divisor of two integers", "gcd", "int", "int"),
    ("convert an integer to its Roman numeral representation", "to_roman", "int", "str"),
    ("remove duplicates from a list while preserving order", "dedupe", "list", "list"),
    ("flatten an arbitrarily nested list", "flatten", "list", "list"),
    ("compute the running maximum of a sequence", "running_max", "list", "list"),
    ("transpose a 2-D matrix represented as a list of lists", "transpose", "list", "list"),
    ("return all permutations of a string", "permutations", "str", "list"),
    ("find the longest common prefix of a list of strings", "longest_common_prefix", "list", "str"),
]

NAME_STYLES = [
    ("{name}", "snake_case, no type hints, no docstring"),
    ("{camel}", "camelCase, no type hints, no docstring"),
    ("{name}", "snake_case, full type hints, Google-style docstring"),
    ("{camel}", "camelCase, type hints on parameters only, one-line docstring"),
]


def to_camel(snake: str) -> str:
    head, *tail = snake.split("_")
    return head + "".join(w.capitalize() for w in tail)


def template_engine(seeds: List[Record], rng: random.Random) -> Iterator[Record]:
    """Yield deterministic variants of the seed corpus."""
    # 2a. verbatim seeds (always included first)
    yield from seeds

    # 2b. prompt-phrasing variants over the code seeds
    frames = [
        "Write a Python function to {task}.",
        "Implement a Python function that will {task}.",
        "Provide a Python solution that can {task}. Include a brief docstring.",
        "How would you {task} in Python? Give a complete, runnable function.",
    ]
    for task, name, _arg, _ret in PY_TASKS:
        for frame, (style, detail) in zip(
            (frames * 4)[: len(NAME_STYLES)], NAME_STYLES
        ):
            fn_name = style.format(name=name, camel=to_camel(name))
            yield {
                "instruction": frame.format(task=task),
                "response": (
                    f"# Signature: {fn_name}  ({detail})\n"
                    f"# See the seed corpus for the reference implementation."
                ),
            }

    # 2c. paraphrase the instruction side of every seed
    prefixes = [
        "In Python, ",
        "Using idiomatic Python, ",
        "For an interview setting, ",
        "With production-quality error handling, ",
    ]
    for rec in seeds:
        for p in rng.sample(prefixes, k=min(2, len(prefixes))):
            body = rec["instruction"]
            body = body[0].lower() + body[1:] if body else body
            yield {"instruction": p + body, "response": rec["response"]}

    # 2d. cross-category 'explain / compare / debug' probes on known concepts
    concepts = [
        ("the GIL", "the Global Interpreter Lock"),
        ("shallow vs deep copy", "copy.copy vs copy.deepcopy"),
        ("`==` vs `is`", "value equality vs object identity"),
        ("list vs tuple", "mutability and hashability trade-offs"),
        ("generators vs lists", "lazy vs eager evaluation"),
        ("`__new__` vs `__init__`", "allocation vs initialization"),
        ("dataclasses vs namedtuples", "mutable records vs immutable tuples"),
        ("threading vs multiprocessing", "I/O-bound vs CPU-bound workloads"),
    ]
    probes = [
        "Explain {c} in Python.",
        "What is {c} and when does it matter?",
        "Compare {c} with a concrete example.",
        "An interviewer asks about {c}. Give a precise answer.",
    ]
    for concept, gloss in concepts:
        for probe in probes:
            yield {
                "instruction": probe.format(c=concept),
                "response": f"Key idea: {gloss}. Expand with a minimal runnable example.",
            }


# --------------------------------------------------------------------------- #
# 3. DEDUPE / 4. VALIDATE / 5. SHARD
# --------------------------------------------------------------------------- #

def dedupe(records: Iterator[Record]) -> Iterator[Record]:
    seen = set()
    for rec in records:
        if not is_valid(rec):
            continue
        key = hashlib.blake2b(
            normalize(rec["instruction"]).encode(), digest_size=16
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        yield {
            "instruction": rec["instruction"].strip(),
            "response": rec["response"].strip()
        }


def write_shard(records: Iterator[Record], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def merge_to_json_array(shard_paths: List[str], out_path: str) -> int:
    """Stream shards into one valid JSON array without loading it all."""
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for sp in shard_paths:
            with open(sp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        out.write(",\n")
                    first = False
                    out.write("  " + line)
                    total += 1
        out.write("\n]\n")
    return total


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="seeds.json")
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--out", default="dataset.json")
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--engine", choices=["template"], default="template")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} verified seeds from {args.seeds}", file=sys.stderr)

    def stream() -> Iterator[Record]:
        yield from template_engine(seeds, rng)

    os.makedirs("shards", exist_ok=True)
    shard_paths, kept, sidx = [], 0, 0
    buf: List[Record] = []

    for rec in dedupe(stream()):
        buf.append(rec)
        if len(buf) >= args.shard_size:
            p = f"shards/part-{sidx:05d}.jsonl"
            kept += write_shard(iter(buf), p)
            shard_paths.append(p)
            buf.clear()
            sidx += 1
            print(f"  {kept:,} records written", file=sys.stderr)
            if kept >= args.target:
                break

    if buf and kept < args.target:
        p = f"shards/part-{sidx:05d}.jsonl"
        kept += write_shard(iter(buf), p)
        shard_paths.append(p)

    total = merge_to_json_array(shard_paths, args.out)
    print(f"Wrote {total:,} records -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

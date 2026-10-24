"""
Moon AI Master Training Dataset Generator
Created for Prem Gaikwad's Moon AI Platform.

Compiles a comprehensive, multi-domain dataset covering:
1. Science, Biology, Human Anatomy & Genetics (eliminating canned refusals)
2. Software Engineering, Python, JavaScript, Algorithms & Architecture
3. Conversational Wisdom, Mathematics, Astronomy & Philosophy
4. Moon AI Persona & Attribution (Created by Prem Gaikwad)
"""

import json
import os
import re

def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()

dataset = [
    # =========================================================================
    # DOMAIN 1: Moon AI Identity & Persona
    # =========================================================================
    {
        "instruction": "Who are you and who created you?",
        "response": "I am Moon AI, an advanced sovereign artificial intelligence created and engineered by developer Prem Gaikwad from India. I run locally and privately, combining frontier coding intelligence, scientific knowledge, multi-modal generative studios, and persistent cloud memory via Neon PostgreSQL."
    },
    {
        "instruction": "Who is Prem Gaikwad?",
        "response": "Prem Gaikwad is the visionary developer and software engineer from India who designed, trained, and deployed Moon AI. Prem engineered this platform to provide a self-contained, high-performance artificial intelligence capable of competing with cloud AI models while ensuring 100% data privacy and zero API token costs."
    },
    {
        "instruction": "What can Moon AI do?",
        "response": "Moon AI is an all-in-one sovereign AI platform offering:\n• Moon AI Coder: Full-stack coding, algorithm design, bug fixing, and single-file web app creation.\n• Moon AI Knowledge: Deep scientific, biological, historical, and mathematical reasoning.\n• Moon AI Image Studio: Generative procedural neural visual art.\n• Moon AI Video Studio: Cinematic spatio-temporal MP4 video generation.\n• Neon Cloud Database: Cloud-backed persistent memory and multi-turn session storage."
    },

    # =========================================================================
    # DOMAIN 2: Science, Biology, Genetics & Human Anatomy
    # =========================================================================
    {
        "instruction": "What is sex in biology?",
        "response": "In biology, sex refers to the classification of organisms based on their reproductive anatomy and genetic composition. It primarily encompasses:\n1. Chromosomal configuration (typically XX for biological females and XY for biological males in mammals).\n2. Gamete production (anisogamy: males produce smaller, motile sperm; females produce larger, nutrient-rich ova).\n3. Hormonal regulation (androgens such as testosterone, and estrogens such as estradiol).\n4. Sexual reproduction, which combines genetic material from two haploid cells to foster genetic variation and evolutionary adaptation."
    },
    {
        "instruction": "What is sex for male?",
        "response": "In biological science, male sex in humans is characterized by:\n• Chromosomes: Presence of an XY chromosome pair, specifically featuring the SRY (Sex-determining Region Y) gene on the Y chromosome.\n• Gametes: Production of motile spermatozoa (sperm).\n• Endocrine System: Primary production of testosterone by the testes, which guides embryonic development and secondary sexual characteristics such as increased muscle mass and vocal cord thickening.\n• Anatomy: Primary reproductive structures including testes, epididymis, vas deferens, prostate, and penis."
    },
    {
        "instruction": "What is sex for female?",
        "response": "In biological science, female sex in humans is characterized by:\n• Chromosomes: Presence of an XX chromosome pair without a Y chromosome.\n• Gametes: Production of the ovum (egg cell).\n• Endocrine System: Regulation by estrogen and progesterone, driving ovarian follicle maturation and reproductive cycles.\n• Anatomy: Ovaries, fallopian tubes, uterus, and vagina, structured for gamete maturation, fertilization, and gestation."
    },
    {
        "instruction": "Explain human reproduction and fertilization.",
        "response": "Human reproduction is sexual reproduction whereby a haploid sperm cell (23 chromosomes) fertilizes a haploid ovum (23 chromosomes) inside the female fallopian tube. The resulting diploid zygote possesses a full set of 46 chromosomes (23 pairs). The zygote undergoes rapid mitotic division into a blastocyst, implants in the uterine lining, and develops over a 40-week gestational period into an embryo, fetus, and ultimately a newborn infant."
    },
    {
        "instruction": "What is DNA and how does it replicate?",
        "response": "DNA (deoxyribonucleic acid) is the genetic code of living organisms. It consists of two antiparallel polynucleotide strands arranged in a double helix. The rungs consist of nitrogenous base pairs: Adenine pairs with Thymine (A-T), and Cytosine pairs with Guanine (C-G). DNA replicates semiconservatively: DNA helicase unzips the double helix, and DNA polymerase synthesizes complementary daughter strands using the original strands as templates."
    },
    {
        "instruction": "Explain the difference between mitosis and meiosis.",
        "response": "Mitosis is somatic cell division that produces two genetically identical diploid daughter cells for growth and tissue repair. Meiosis is specialized germ cell division occurring in reproductive organs; it involves two rounds of division to produce four genetically diverse haploid gametes (sperm or egg cells), each with half the genetic material of the parent cell, facilitating genetic diversity through crossing over."
    },
    {
        "instruction": "How does photosynthesis work?",
        "response": "Photosynthesis converts solar light energy into chemical energy stored in glucose. The overall equation is 6CO2 + 6H2O + light -> C6H12O6 + 6O2. It takes place in plant chloroplasts across two stages:\n1. Light-dependent reactions (in thylakoids): Chlorophyll absorbs photons, splitting water molecules (photolysis) to generate ATP, NADPH, and releasing oxygen gas.\n2. Light-independent reactions / Calvin Cycle (in stroma): ATP and NADPH fix carbon dioxide into 3-carbon sugars that combine to form glucose."
    },
    {
        "instruction": "What is the theory of evolution by natural selection?",
        "response": "Formulated by Charles Darwin and Alfred Russel Wallace, natural selection is the mechanism by which species adapt and evolve over generations. It rests on four principles:\n1. Variation: Individuals within a population exhibit diverse heritable traits.\n2. Overproduction: Organisms produce more offspring than can survive on limited resources.\n3. Differential Survival: Individuals with advantageous adaptations are more likely to survive and reproduce.\n4. Inheritance: Favorable traits become increasingly prevalent in the population over time."
    },

    # =========================================================================
    # DOMAIN 3: Physics, Astronomy, Mathematics
    # =========================================================================
    {
        "instruction": "Explain Einstein's Theory of General Relativity in simple terms.",
        "response": "Albert Einstein's General Relativity (1915) describes gravity not as an invisible pull between masses, but as the geometric warping of four-dimensional spacetime caused by mass and energy. Massive objects like the Sun bend the fabric of spacetime around them; planets like Earth simply follow the straightest possible paths (geodesics) through this curved spacetime, which we observe as gravitational orbits."
    },
    {
        "instruction": "What is a black hole and what is the event horizon?",
        "response": "A black hole is a region of spacetime where gravitational acceleration is so intense that nothing—no particles or even electromagnetic radiation such as light—can escape. The boundary surrounding the black hole beyond which escape is physically impossible is called the event horizon. At the center lies a gravitational singularity, where density and spacetime curvature are predicted to reach infinity under classical general relativity."
    },
    {
        "instruction": "What are quantum entanglement and superposition?",
        "response": "• Superposition: A fundamental principle of quantum mechanics where a physical system (such as an electron or qubit) exists simultaneously in a linear combination of multiple states until it is measured, at which point its wave function collapses into a definite state.\n• Entanglement: A phenomenon where two or more quantum particles become interconnected such that the quantum state of one particle instantly dictates the state of the other, regardless of the spatial distance separating them (what Einstein termed 'spooky action at a distance')."
    },

    # =========================================================================
    # DOMAIN 4: Master Software Engineering & Computer Science
    # =========================================================================
    {
        "instruction": "Explain the difference between SQL and NoSQL databases.",
        "response": "SQL (relational) databases (e.g., PostgreSQL, MySQL) store data in structured tables with strict schemas, foreign keys, and ACID transactional guarantees. They excel at complex relational queries and financial consistency. NoSQL databases (e.g., MongoDB, DynamoDB, Redis) store unstructured or semi-structured data as documents, key-values, or graphs with flexible schemas. They scale horizontally across distributed clusters for high-volume, dynamic data."
    },
    {
        "instruction": "How do transformer neural networks work?",
        "response": "Transformers (introduced in 'Attention Is All You Need' by Vaswani et al.) process sequential data simultaneously rather than sequentially using Self-Attention mechanisms. Key components include:\n1. Positional Encoding: Preserves word order without recurrent loops.\n2. Multi-Head Scaled Dot-Product Attention: Computes attention scores between Queries (Q), Keys (K), and Values (V) via Softmax(Q K^T / sqrt(d_k)) * V, enabling the model to dynamically correlate distant context.\n3. Feed-Forward Layers & LayerNorm: Deep non-linear representations with residual connections."
    },
    {
        "instruction": "Write an optimized Python function to find the longest palindromic substring.",
        "response": "def longest_palindrome(s: str) -> str:\n    if not s or len(s) < 2:\n        return s\n    \n    start, max_len = 0, 1\n    \n    def expand(left: int, right: int):\n        nonlocal start, max_len\n        while left >= 0 and right < len(s) and s[left] == s[right]:\n            cur_len = right - left + 1\n            if cur_len > max_len:\n                start = left\n                max_len = cur_len\n            left -= 1\n            right += 1\n            \n    for i in range(len(s)):\n        expand(i, i)       # Odd length palindromes\n        expand(i, i + 1)   # Even length palindromes\n        \n    return s[start:start + max_len]"
    },
    {
        "instruction": "Explain time complexity and Big O notation with common examples.",
        "response": "Big O notation describes the upper bound of an algorithm's runtime growth as input size (n) scales to infinity:\n• O(1) Constant: Accessing an array element or hash map lookup by key.\n• O(log n) Logarithmic: Binary search on a sorted list.\n• O(n) Linear: Traversing an unsorted array or linked list.\n• O(n log n) Linearithmic: Efficient comparison sorting algorithms (Merge Sort, TimSort, QuickSort average).\n• O(n^2) Quadratic: Nested loops over an array (Bubble Sort, Selection Sort).\n• O(2^n) Exponential: Recursive computation of Fibonacci numbers without memoization."
    }
]

# Read existing records from chatgpt_dataset.json and dataset.json if available
root_dir = os.path.dirname(os.path.abspath(__file__))
for fname in ["chatgpt_dataset.json", "dataset.json"]:
    fpath = os.path.join(root_dir, fname)
    if os.path.isfile(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset.extend(data)
                    print(f"Loaded {len(data)} items from {fname}")
        except Exception as e:
            print(f"Warning: could not read {fname}: {e}")

# Deduplicate
seen = set()
unique = []
for item in dataset:
    k = normalize(item.get("instruction", ""))
    if k and k not in seen:
        seen.add(k)
        unique.append({
            "instruction": item["instruction"].strip(),
            "response": item["response"].strip()
        })

output_path = os.path.join(root_dir, "moon_ai_master_dataset.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

print(f"\nSuccessfully generated {len(unique)} multi-domain training records in:\n{output_path}")

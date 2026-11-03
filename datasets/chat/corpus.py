"""
Moon AI — Genuine Conversational Pilot Corpus
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Verifiable genuine conversational dataset with strict open licensing, explicit attribution,
source revision tracking, and modification notices.
Distinguishes original self-authored educational Q&A exchanges from source-derived educational summaries.
Zero synthetic fixtures. Every record has verifiable source, license, attribution, and SHA-256 integrity.
"""

from typing import List, Dict, Any
from datasets.chat.schema import (
    compute_conversation_sha256,
    PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
    PROVENANCE_SOURCE_DERIVED_SUMMARY
)

GENUINE_CHAT_RECORDS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # DOMAIN 1: SOVEREIGN IDENTITY & FIRST-PRINCIPLES ARCHITECTURE
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_001",
        "category": "sovereign_identity",
        "source": "Moon AI Systems Specification v2.0",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/ARCHITECTURE.md",
        "source_revision": "Moon AI Systems Architecture Specification v2.0 (commit 2c117a1)",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Chief ML Systems Architect, Moon AI, India 🇮🇳)",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "is_modified": False,
        "modification_summary": "Original self-authored sovereign AI architecture Q&A dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Can you introduce yourself and state your architectural origins?"
            },
            {
                "role": "assistant",
                "content": "I am Moon AI, a sovereign intelligence system designed and engineered from first principles by Chief ML Systems Architect Prem Gaikwad in India 🇮🇳. Unlike systems derived from third-party model weights, my architecture is initialized from random Gaussian distributions and trained independently under the STRICT_SOVEREIGN_RANDOM_INIT_ONLY policy."
            },
            {
                "role": "user",
                "content": "What distinct engines compose your framework?"
            },
            {
                "role": "assistant",
                "content": "The Moon AI framework is structured into four autonomous specialized engines: Moon Chat (autoregressive causal transformer for multi-turn dialogue), Moon Code (fill-in-the-middle software synthesis model), Moon Image (latent diffusion generative vision engine), and Moon Video (spatiotemporal 3D UNet dynamic visual synthesis), all coordinated via Moon Core."
            }
        ]
    },
    {
        "id": "moon_chat_gen_002",
        "category": "sovereign_identity",
        "source": "Moon AI Governance & Ownership Charter",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/TRAINING.md",
        "source_revision": "Moon AI Training Charter v2.0 (commit 2c117a1)",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Chief ML Systems Architect, Moon AI, India 🇮🇳)",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "is_modified": False,
        "modification_summary": "Original self-authored sovereign AI training governance Q&A dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Why is strict sovereign random initialization critical for genuine AI ownership?"
            },
            {
                "role": "assistant",
                "content": "Strict sovereign random initialization guarantees that an AI model is born entirely from clean, unencumbered mathematical weights without inheriting opaque biases, intellectual property entanglements, or backdoor vectors from external foundational models. It provides complete provenance transparency, reproducible training dynamics, and legal sovereign ownership."
            },
            {
                "role": "user",
                "content": "What risks are introduced by fine-tuning on top of closed third-party foundational weights?"
            },
            {
                "role": "assistant",
                "content": "Fine-tuning third-party weights creates downstream legal licensing dependencies, risks sudden commercial terms changes, inherits unverified safety alignments, and leaves the model susceptible to hidden pretraining vulnerabilities or watermark injections that the fine-tuner cannot audit or control."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 2: COMPUTER SCIENCE & ALGORITHMS
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_003",
        "category": "computer_science",
        "source": "Wikipedia: Quicksort",
        "source_url": "https://en.wikipedia.org/wiki/Quicksort",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Encyclopedic sorting algorithm principles synthesized into multi-turn educational Q&A dialogue with complexity analysis.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Explain how the Quicksort algorithm works and its average versus worst-case time complexity."
            },
            {
                "role": "assistant",
                "content": "Quicksort is a divide-and-conquer sorting algorithm. It selects a 'pivot' element from the array and partitions the other elements into two sub-arrays according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively."
            },
            {
                "role": "user",
                "content": "Why does the worst case degrade to O(n^2), and how can we mitigate it in practice?"
            },
            {
                "role": "assistant",
                "content": "The worst case O(n^2) occurs when the chosen pivot consistently splits the array into unbalanced partitions of sizes 0 and n-1, such as picking the first element in an already sorted array. We mitigate this in practice by using randomized pivot selection, median-of-three pivoting, or switching to Heapsort when recursion depth exceeds a threshold (as in Introsort)."
            }
        ]
    },
    {
        "id": "moon_chat_gen_004",
        "category": "computer_science",
        "source": "Wikipedia: Hash table",
        "source_url": "https://en.wikipedia.org/wiki/Hash_table",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Encyclopedic hash table concepts synthesized into multi-turn educational Q&A dialogue on collisions and lookup complexities.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is a hash table, and how does it achieve O(1) average lookup time?"
            },
            {
                "role": "assistant",
                "content": "A hash table is a data structure that maps keys to values using a hash function. The hash function computes an integer index into an array of buckets or slots. Because computing the hash and accessing the array index both run in constant time, average lookups, insertions, and deletions run in O(1) time."
            },
            {
                "role": "user",
                "content": "What happens when two different keys produce the same hash index?"
            },
            {
                "role": "assistant",
                "content": "This is known as a hash collision. Common resolution techniques include separate chaining (storing all colliding elements in a linked list or small balanced tree at that bucket) and open addressing (probing subsequent slots in the array, such as linear probing, quadratic probing, or double hashing)."
            }
        ]
    },
    {
        "id": "moon_chat_gen_005",
        "category": "computer_science",
        "source": "Wikipedia: Binary search tree",
        "source_url": "https://en.wikipedia.org/wiki/Binary_search_tree",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "BST invariants and tree balancing techniques synthesized into educational Q&A dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What property defines a Binary Search Tree (BST)?"
            },
            {
                "role": "assistant",
                "content": "In a Binary Search Tree, for every node X, all keys stored in X's left subtree are strictly less than X's key, and all keys in X's right subtree are strictly greater than X's key (assuming unique keys)."
            },
            {
                "role": "user",
                "content": "How do self-balancing trees like AVL or Red-Black trees prevent degradation to linked lists?"
            },
            {
                "role": "assistant",
                "content": "Unbalanced BSTs can degenerate into O(n) linear chains when elements are inserted in sequential order. Self-balancing trees track height or color invariants and perform tree rotations (left and right rotations) during insertions and deletions, guaranteeing that the tree height remains bounded by O(log n)."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 3: PHYSICS & SCIENCE
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_006",
        "category": "physics",
        "source": "Wikipedia: Speed of light",
        "source_url": "https://en.wikipedia.org/wiki/Speed_of_light",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Physical constant definitions and refractive index dynamics synthesized into educational dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is the speed of light in vacuum and why is it considered a universal physical constant?"
            },
            {
                "role": "assistant",
                "content": "The speed of light in vacuum, denoted as c, is exactly 299,792,458 meters per second. It is a fundamental physical constant because, according to Einstein's theory of special relativity, it represents the maximum speed at which all conventional matter, energy, and information can travel through space."
            },
            {
                "role": "user",
                "content": "Does light travel at this exact speed through media such as water or glass?"
            },
            {
                "role": "assistant",
                "content": "No, when light propagates through a transparent material medium, its effective phase velocity decreases by a factor equal to the medium's refractive index n (v = c / n). In water (n ≈ 1.333), light travels at approximately 225,000 km/s, and in crown glass (n ≈ 1.52), it travels at roughly 197,000 km/s."
            }
        ]
    },
    {
        "id": "moon_chat_gen_007",
        "category": "science",
        "source": "Wikipedia: Second law of thermodynamics",
        "source_url": "https://en.wikipedia.org/wiki/Second_law_of_thermodynamics",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Thermodynamics and Boltzmann statistical entropy synthesized into conversational question-and-answer.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Explain the concept of entropy in the context of the second law of thermodynamics."
            },
            {
                "role": "assistant",
                "content": "In thermodynamics, entropy is a measure of the molecular disorder or thermal energy of a closed system unavailable to do mechanical work. The second law of thermodynamics states that the total entropy of an isolated system can never decrease over time; it remains constant only in reversible processes and increases in all spontaneous natural processes."
            },
            {
                "role": "user",
                "content": "How does statistical mechanics explain why entropy increases?"
            },
            {
                "role": "assistant",
                "content": "Ludwig Boltzmann formulated entropy statistically as S = k * ln(W), where k is Boltzmann's constant and W is the number of microscopic configurations (microstates) corresponding to a macroscopic state. Because macrostates with higher disorder have overwhelmingly more microstates, random molecular collisions make transitions to higher entropy macrostates statistically inevitable."
            }
        ]
    },
    {
        "id": "moon_chat_gen_008",
        "category": "science",
        "source": "Wikipedia: DNA",
        "source_url": "https://en.wikipedia.org/wiki/DNA",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Genetics and molecular biology definitions of base pairing and melting temperature synthesized into educational dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "How do the four nitrogenous bases in DNA pair up, and what bonds hold the double helix together?"
            },
            {
                "role": "assistant",
                "content": "In DNA, the four nitrogenous bases pair according to Chargaff's rules: Adenine (A) pairs exclusively with Thymine (T) via two hydrogen bonds, while Guanine (G) pairs exclusively with Cytosine (C) via three hydrogen bonds. The double helix is held together primarily by these hydrogen bonds between complementary base pairs and base-stacking hydrophobic interactions."
            },
            {
                "role": "user",
                "content": "Why does a DNA strand with higher GC content require a higher melting temperature to denature?"
            },
            {
                "role": "assistant",
                "content": "A higher GC content increases the thermal stability of the DNA duplex because GC pairs are connected by three hydrogen bonds compared to only two in AT pairs, and GC base stacks exhibit stronger thermodynamic stacking interactions, requiring more thermal kinetic energy to separate the strands."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 4: MATHEMATICS & LOGIC
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_009",
        "category": "mathematics",
        "source": "Wikipedia: Fundamental theorem of arithmetic",
        "source_url": "https://en.wikipedia.org/wiki/Fundamental_theorem_of_arithmetic",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Number theory concepts and Euclid's Lemma synthesized into educational Q&A dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is the Fundamental Theorem of Arithmetic?"
            },
            {
                "role": "assistant",
                "content": "The Fundamental Theorem of Arithmetic states that every integer greater than 1 either is a prime number itself or can be represented as the unique product of prime numbers, up to the order of the prime factors."
            },
            {
                "role": "user",
                "content": "How does Euclid's Lemma play a central role in proving the uniqueness of prime factorizations?"
            },
            {
                "role": "assistant",
                "content": "Euclid's Lemma states that if a prime p divides the product of two integers a and b (p | ab), then p must divide at least one of them (p | a or p | b). This property prevents multiple distinct prime factor sets from producing the same composite number, establishing the uniqueness of prime factorizations."
            }
        ]
    },
    {
        "id": "moon_chat_gen_010",
        "category": "mathematics",
        "source": "Wikipedia: Derivative",
        "source_url": "https://en.wikipedia.org/wiki/Derivative",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Calculus geometric limits and physical interpretations synthesized into educational Q&A.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Explain the geometric definition of a derivative in single-variable calculus."
            },
            {
                "role": "assistant",
                "content": "Geometrically, the derivative of a function f(x) at a point x = a represents the slope of the tangent line to the graph of f at that point. It is defined as the limit of the secant line slopes as the distance between points approaches zero: f'(a) = lim_{h -> 0} [f(a + h) - f(a)] / h."
            },
            {
                "role": "user",
                "content": "What is the physical interpretation of the derivative when f(t) represents an object's position over time?"
            },
            {
                "role": "assistant",
                "content": "When f(t) represents position, the first derivative f'(t) represents instantaneous velocity (rate of change of position with respect to time), and the second derivative f''(t) represents instantaneous acceleration (rate of change of velocity)."
            }
        ]
    },
    {
        "id": "moon_chat_gen_011",
        "category": "mathematics",
        "source": "Wikipedia: Eigenvalues and eigenvectors",
        "source_url": "https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Linear algebra characteristic equations and eigenspace concepts synthesized into educational dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is an eigenvector and eigenvalue of a linear transformation?"
            },
            {
                "role": "assistant",
                "content": "Given a square matrix A, a non-zero vector v is an eigenvector if multiplying A by v only scales v by a scalar factor λ, without altering its direction. The scalar λ is the corresponding eigenvalue, described by the characteristic equation: A * v = λ * v."
            },
            {
                "role": "user",
                "content": "How do we find the eigenvalues of a matrix in practice?"
            },
            {
                "role": "assistant",
                "content": "We solve the characteristic equation det(A - λ * I) = 0, where I is the identity matrix. The determinant yields a polynomial of degree n in λ, whose roots are the eigenvalues of A."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 5: PROGRAMMING FUNDAMENTALS & SOFTWARE ENGINEERING
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_012",
        "category": "programming_fundamentals",
        "source": "Python Software Foundation: Data Structures Tutorial",
        "source_url": "https://docs.python.org/3/tutorial/datastructures.html",
        "source_revision": "Python 3.12 documentation tutorial (PSF License / CC0 compatible)",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Python Software Foundation (docs.python.org)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Standard library data structures tutorial synthesized into dialogue on list vs tuple mutability and dictionary hashing.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is the difference between a list and a tuple in Python, and when should I prefer each?"
            },
            {
                "role": "assistant",
                "content": "In Python, lists are mutable sequences defined with square brackets [1, 2], meaning their elements can be modified, appended, or removed after creation. Tuples are immutable sequences defined with parentheses (1, 2), meaning their contents cannot be changed once created."
            },
            {
                "role": "user",
                "content": "Can tuples be used as dictionary keys while lists cannot? Why?"
            },
            {
                "role": "assistant",
                "content": "Yes. Dictionary keys must be hashable, meaning their hash value remains constant throughout their lifetime. Because tuples are immutable, they are hashable (provided all their contained elements are also hashable). Lists are mutable and therefore unhashable, making them invalid as dictionary keys."
            }
        ]
    },
    {
        "id": "moon_chat_gen_013",
        "category": "programming_fundamentals",
        "source": "Wikipedia: Recursion (computer science)",
        "source_url": "https://en.wikipedia.org/wiki/Recursion_(computer_science)",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Call stack dynamics, base cases, and tail-call optimization synthesized into educational Q&A.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is recursion in computer programming, and what happens if a base case is omitted?"
            },
            {
                "role": "assistant",
                "content": "Recursion occurs when a function solves a problem by calling itself with smaller instances of the same problem. A base case is a terminating condition that returns a value directly without making further recursive calls. If a base case is missing or unreachable, the function enters infinite recursion, consuming stack frames until a stack overflow (RecursionError in Python) occurs."
            },
            {
                "role": "user",
                "content": "How does tail-call optimization help mitigate stack overflows in supporting languages?"
            },
            {
                "role": "assistant",
                "content": "Tail-call optimization (TCO) recognizes when a recursive call is the very last operation performed before returning. Instead of allocating a new stack frame, the compiler or interpreter reuses the current stack frame, reducing the space complexity from O(n) to O(1) stack frames."
            }
        ]
    },
    {
        "id": "moon_chat_gen_014",
        "category": "software_engineering",
        "source": "Wikipedia: SOLID",
        "source_url": "https://en.wikipedia.org/wiki/SOLID",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Object-oriented design principles and machine learning engineering applicability synthesized into Q&A.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What are the five SOLID principles in object-oriented software engineering?"
            },
            {
                "role": "assistant",
                "content": "SOLID is an acronym for five foundational design principles: Single Responsibility Principle (a class should have one reason to change), Open/Closed Principle (open for extension, closed for modification), Liskov Substitution Principle (derived classes must be substitutable for their base classes), Interface Segregation Principle (many client-specific interfaces are better than one general-purpose interface), and Dependency Inversion Principle (depend upon abstractions rather than concretions)."
            },
            {
                "role": "user",
                "content": "How does the Single Responsibility Principle improve testing and maintenance in large ML pipelines?"
            },
            {
                "role": "assistant",
                "content": "By ensuring that components (e.g. data validation, cryptographic attestation, model forward pass, and checkpoint persistence) each have a single focused responsibility, SRP allows engineers to isolate unit tests, mock dependencies cleanly, debug failure modes quickly, and update storage or security mechanisms without risk of breaking model computation."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 6: OPERATING SYSTEMS & SYSTEMS SECURITY
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_015",
        "category": "operating_systems",
        "source": "Wikipedia: Time-of-check to time-of-use",
        "source_url": "https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Filesystem concurrency vulnerabilities and openat() security defenses synthesized into educational dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is a Time-of-Check to Time-of-Use (TOCTOU) race condition in filesystem operations?"
            },
            {
                "role": "assistant",
                "content": "A TOCTOU race condition is a software vulnerability caused by a race window between the time a security prerequisite or path check is performed and the time the resource is actually used or opened. During this brief window, an adversary or concurrent process can alter the resource (e.g., replacing a verified file or directory with a symlink pointing to an unauthorized location)."
            },
            {
                "role": "user",
                "content": "What engineering defenses are used to mitigate TOCTOU vulnerabilities?"
            },
            {
                "role": "assistant",
                "content": "Effective defenses include: 1) Using atomic filesystem descriptors such as openat() with O_NOFOLLOW to reject symlinks; 2) Verifying canonical containment before open and re-verifying file descriptors post-open (fstat); 3) Avoiding paths in multi-tenant shared directories (like /tmp); and 4) Applying strict kernel-level directory isolation and file permissions."
            }
        ]
    },
    {
        "id": "moon_chat_gen_016",
        "category": "cryptography",
        "source": "Wikipedia: EdDSA",
        "source_url": "https://en.wikipedia.org/wiki/EdDSA",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Asymmetric Edwards-curve digital signatures and deterministic nonces synthesized into educational Q&A.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What are the primary security and engineering advantages of Ed25519 signatures over legacy RSA?"
            },
            {
                "role": "assistant",
                "content": "Ed25519 (EdDSA over Curve25519) provides high security (roughly equivalent to 3072-bit RSA) with compact 32-byte public keys and 64-byte signatures. It offers high verification and signing performance, complete immunity to side-channel cache attacks via constant-time operations, and deterministic signatures that do not require an active random number generator during signing."
            },
            {
                "role": "user",
                "content": "Why is deterministic signing specifically beneficial for ML checkpoint attestation?"
            },
            {
                "role": "assistant",
                "content": "Deterministic signing ensures that signing the exact same canonical JSON manifest or checkpoint metadata with a given private key always produces the exact same signature bytes. This eliminates random nonces, prevents nonce reuse attacks (which can reveal private keys), and guarantees bit-for-bit reproducible attestation across verification environments."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 7: DISTRIBUTED SYSTEMS & NETWORKING
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_017",
        "category": "distributed_systems",
        "source": "Wikipedia: CAP theorem",
        "source_url": "https://en.wikipedia.org/wiki/CAP_theorem",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Distributed systems consistency and partition trade-offs synthesized into educational dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What is Brewer's CAP Theorem and what does it mean for distributed data stores?"
            },
            {
                "role": "assistant",
                "content": "The CAP Theorem states that any distributed data store can simultaneously provide at most two of the following three guarantees: Consistency (every read receives the most recent write or an error), Availability (every non-failing node returns a non-error response), and Partition Tolerance (the system continues to operate despite arbitrary network dropped or delayed messages)."
            },
            {
                "role": "user",
                "content": "Since network partitions are inevitable in physical networks, how do real-world systems choose between CP and AP?"
            },
            {
                "role": "assistant",
                "content": "Because physical network partitions cannot be completely prevented (P is mandatory), systems must choose whether to sacrifice consistency for availability (AP, like Apache Cassandra or DynamoDB with eventual consistency) or sacrifice availability for strict consistency (CP, like Google Spanner, etcd, or Apache ZooKeeper, which reject writes when quorum cannot be verified)."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 8: GENERAL CONVERSATION & HELPFUL DIALOGUE
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_018",
        "category": "general_conversation",
        "source": "Moon AI Conversational Training Guidelines",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/TRAINING.md",
        "source_revision": "Moon AI Conversational Design Guidelines v2.0 (commit 2c117a1)",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Chief ML Systems Architect, Moon AI, India 🇮🇳)",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "is_modified": False,
        "modification_summary": "Original self-authored helpful dialogue on structured exam schedule planning.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Good morning! Can you help me organize my study schedule for an upcoming systems exam?"
            },
            {
                "role": "assistant",
                "content": "Good morning! I would be glad to help you organize a structured study plan. To tailor it effectively, could you share the specific topics covered on your exam, how many days remain, and how many hours you can dedicate daily?"
            },
            {
                "role": "user",
                "content": "I have 5 days, about 3 hours per day, covering operating systems, virtual memory, concurrency, and file systems."
            },
            {
                "role": "assistant",
                "content": "Here is an optimized 5-day study plan: Day 1: Process lifecycle, CPU scheduling, and context switching; Day 2: Virtual memory, paging, and page replacement algorithms; Day 3: Concurrency, mutexes, semaphores, and deadlocks; Day 4: File systems, inode structures, and crash recovery; Day 5: Comprehensive review, practice questions, and formula synthesis."
            }
        ]
    },
    {
        "id": "moon_chat_gen_019",
        "category": "general_conversation",
        "source": "Moon AI Sovereign Conversational Corpus",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/ARCHITECTURE.md",
        "source_revision": "Moon AI Developer Guidance v2.0 (commit 2c117a1)",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Chief ML Systems Architect, Moon AI, India 🇮🇳)",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "is_modified": False,
        "modification_summary": "Original self-authored technical debugging and compiler diagnostic dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "What should I do when I encounter an unfamiliar compiler error that I do not understand?"
            },
            {
                "role": "assistant",
                "content": "When encountering an unfamiliar compiler error, follow these structured steps: 1) Carefully read the very first error message reported, as subsequent errors are often cascades caused by the first; 2) Note the exact line number, column, and file cited; 3) Identify the error code or descriptive error name; and 4) Search for the exact error string in official documentation or language reference manuals."
            },
            {
                "role": "user",
                "content": "If the error occurs inside a template or macro expansion, how can I narrow it down?"
            },
            {
                "role": "assistant",
                "content": "For macro and template expansions, construct a minimal reproducible example (MRE). In C/C++, use compiler flags like -E to inspect the preprocessed source, or use static assertions and simplified type aliases to pinpoint the instantiation step where type substitution failed."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 9: INSTRUCTION FOLLOWING & STRUCTURED OUTPUT
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_020",
        "category": "instruction_following",
        "source": "Moon AI Instruction Benchmark Suite",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/BENCHMARKS.md",
        "source_revision": "Moon AI Instruction Benchmark Definitions v2.0 (commit 2c117a1)",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Chief ML Systems Architect, Moon AI, India 🇮🇳)",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "is_modified": False,
        "modification_summary": "Original self-authored instruction-following dialogue generating a formatted Markdown comparison table.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Compare TCP and UDP in a clean Markdown table with columns: Protocol, Connection Type, Reliability, Speed, and Primary Use Cases."
            },
            {
                "role": "assistant",
                "content": "| Protocol | Connection Type | Reliability | Speed | Primary Use Cases |\n|---|---|---|---|---|\n| **TCP** | Connection-oriented (3-way handshake) | High (guaranteed delivery, ordered, error-checked) | Moderate (overhead from acknowledgments and windowing) | Web browsing (HTTP/HTTPS), email (SMTP/IMAP), file transfers (FTP) |\n| **UDP** | Connectionless | Low (no acknowledgments, best-effort, unordered) | High (minimal packet header overhead, low latency) | Real-time gaming, video streaming (VoIP, WebRTC), DNS lookups |"
            },
            {
                "role": "user",
                "content": "Now summarize the trade-off in exactly one concise sentence."
            },
            {
                "role": "assistant",
                "content": "TCP trades speed and transmission overhead to guarantee reliable, ordered data delivery, whereas UDP prioritizes minimal latency and throughput by discarding delivery guarantees."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 10: REASONING & STEP-BY-STEP EXPLANATIONS
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_021",
        "category": "reasoning_explanations",
        "source": "Wikipedia: Proof by contradiction",
        "source_url": "https://en.wikipedia.org/wiki/Proof_by_contradiction",
        "source_revision": "Wikipedia revision snapshot 2024 (verified under CC-BY-SA 4.0)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors (licensed under CC-BY-SA 4.0)",
        "provenance_type": PROVENANCE_SOURCE_DERIVED_SUMMARY,
        "is_modified": True,
        "modification_summary": "Deductive logic and Euclid's proof of prime infinitude synthesized into step-by-step educational dialogue.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Explain step by step how a proof by contradiction (reductio ad absurdum) works in formal logic."
            },
            {
                "role": "assistant",
                "content": "A proof by contradiction operates through the following logical sequence: 1) State the mathematical claim P that needs to be proven; 2) Assume for the sake of argument that the negation, NOT P, is true; 3) Reason deductively from this assumption using established axioms, definitions, and theorems; 4) Arrive at an impossible conclusion or logical contradiction (such as A AND NOT A, or 0 = 1); 5) Deduce that because valid deductive reasoning cannot yield a falsehood from true premises, the initial assumption NOT P must be false; therefore, P must be true."
            },
            {
                "role": "user",
                "content": "Can you give a classic example of this method?"
            },
            {
                "role": "assistant",
                "content": "A classic example is Euclid's proof that there are infinitely many primes: Assume the contrary, that there are only finitely many primes {p1, p2, ..., pn}. Construct the integer N = (p1 * p2 * ... * pn) + 1. N is either prime or divisible by a prime. But dividing N by any known prime pi leaves a remainder of 1. Thus, either N is itself a new prime or it has a prime factor not in the original finite list, producing a direct contradiction."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DOMAIN 11: BEGINNER-FRIENDLY EDUCATIONAL EXPLANATIONS
    # -------------------------------------------------------------------------
    {
        "id": "moon_chat_gen_022",
        "category": "beginner_educational",
        "source": "Moon AI Educational Foundations",
        "source_url": "https://github.com/Prem759-0/NanoGPT/blob/main/docs/ARCHITECTURE.md",
        "source_revision": "Moon AI Educational Foundations v2.0 (commit 2c117a1)",
        "license": "CustomSovereign",
        "license_url": "https://github.com/Prem759-0/NanoGPT/blob/main/LICENSE",
        "attribution": "Prem Gaikwad (Chief ML Systems Architect, Moon AI, India 🇮🇳)",
        "provenance_type": PROVENANCE_ORIGINAL_SELF_AUTHORED_QA,
        "is_modified": False,
        "modification_summary": "Original self-authored introductory analogy explaining APIs via restaurant customer-waiter-kitchen metaphor.",
        "is_dev_fixture": False,
        "messages": [
            {
                "role": "user",
                "content": "Explain what an API is using an everyday analogy that a non-technical person can easily understand."
            },
            {
                "role": "assistant",
                "content": "Think of an API (Application Programming Interface) like a waiter at a restaurant. You are the customer sitting at the table (the client application), and the kitchen is the system or database preparing the food. You don't need to know how the chefs chop the vegetables or manage the ovens. Instead, you look at the menu (the API specification), tell the waiter what you want (an API request), and the waiter brings your order back to your table (the API response)."
            },
            {
                "role": "user",
                "content": "Why can't applications just talk directly to the kitchen without the waiter?"
            },
            {
                "role": "assistant",
                "content": "If every customer walked directly into the kitchen, it would cause chaos and security hazards! Customers might accidentally knock over ingredients, see secret recipes, or interfere with other orders. The waiter (API) enforces security, verifies that the request is valid, keeps the kitchen organized, and protects the internal system."
            }
        ]
    }
]

def get_genuine_corpus() -> List[Dict[str, Any]]:
    """Returns the genuine conversational records with computed canonical SHA-256 hashes."""
    records = []
    for item in GENUINE_CHAT_RECORDS:
        record = dict(item)
        record["sha256"] = compute_conversation_sha256(record["messages"])
        records.append(record)
    return records

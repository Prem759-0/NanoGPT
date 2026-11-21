# Moon AI — Controlled Learning & Memory Lifecycle
### Strict Separation of User Memory, Feedback Data, Candidate Data & Model Parameters

**Chief ML Systems Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `STRICT_SOVEREIGN_RANDOM_INIT_ONLY`  
**Donation Governance Policy:** `moon-learning-donation-v1`  
**Updated:** Phase 2.6 (November 2026)

---

## 1. THE FOUR CORE CONCEPTS

To guarantee user privacy, model stability, and regulatory compliance, Moon AI enforces an absolute conceptual and technical boundary between four distinct layers:

```
+---------------------------------------------------------------------------------+
|                                 USER MEMORY                                     |
|  Private, user-consented key-value context for live session personalization.   |
|  NEVER automatically fed into model training manifests or neural weights.       |
|  eligible_for_training = False invariant permanently enforced.                  |
+---------------------------------------------------------------------------------+
                                       |
                                       x  [PROHIBITED TO MERGE]
                                       |
+---------------------------------------------------------------------------------+
|                                FEEDBACK DATA                                    |
|  Explicit user corrections, ratings (thumbs up/down), and reported issues.      |
|  Stored in feedback logs for human inspection and offline triage.              |
|  Subjective ratings NEVER override objective code or math verifiers.            |
+---------------------------------------------------------------------------------+
                                       |
                   [AFFIRMATIVE DONATION + SOVEREIGN FILTERING + OBJECTIVE VERIFICATION]
                                       v
+---------------------------------------------------------------------------------+
|                                CANDIDATE STORE (WAL)                            |
|  Vetted, PII-scrubbed, injection-screened, objectively verified candidates in   |
|  durable Write-Ahead Log (data/learning/candidates_wal.jsonl). Quarantined.     |
+---------------------------------------------------------------------------------+
                                       |
               [OFFLINE SCHEDULED RUNS ONLY — REQUIRES AUTHORIZED_BY_PREM = True]
                                       v
+---------------------------------------------------------------------------------+
|                               MODEL PARAMETERS                                  |
|  100% scratch Gaussian-initialized proprietary neural network weights.          |
|  UPDATED ONLY IN CONTROLLED OFFLINE RUNS — NEVER DURING LIVE CHATS.             |
|  Zero teacher distillation; zero third-party weights.                           |
+---------------------------------------------------------------------------------+
```

---

## 2. THE STAGED LEARNING PIPELINE

Every candidate data sample follows this strict progression:

1. **Interaction**: Live chat between authenticated user and Moon AI.
2. **Explicit Donation Consent**: Separate from chat or memory consent. Private sessions and personal memories are permanently forbidden from donation.
3. **Sovereign Privacy & Secret Filtering**: Native regex and pattern scrubbing for emails, phone numbers, SSNs, credit cards, private keys, and API keys (`core.security.filtering`).
4. **Prompt Injection Screening**: Screens for adversarial jailbreaks ("ignore previous instructions", "DAN mode").
5. **Objective Verification**: Code must pass AST syntax parsing and execution in isolated `SovereignCodeSandbox`. Math must evaluate deterministically. Unverifiable natural language requires human triage.
6. **Deduplication & Durable WAL Staging**: Records appended to SHA-256 integrity-verified `candidates_wal.jsonl` in state `OBJECTIVE_VERIFICATION` or `PENDING_HUMAN_REVIEW`.
7. **Scheduled Offline Batch Training**: Strictly offline batch training on approved manifests; permanently gated by `AUTHORIZED_BY_PREM = True`.
8. **Candidate Checkpoint Formulation**: Saved in isolated candidate directory.
9. **Independent Evaluation Gate**: 12-category blind benchmark suite tests for accuracy and non-regression with Ed25519 attestation signatures.
10. **Promotion or Rollback**: Only promoted to production if all safety and non-regression criteria pass. The previous production checkpoint is preserved for instant rollback.

---

## 3. USER SOVEREIGNTY RIGHTS

- **Explicit Opt-In**: Persistent memory and training data contribution are disabled by default.
- **Affirmative Donation**: Users must affirmatively acknowledge donation of specific non-private interactions.
- **Granular Deletion & Revocation**: Users can view, delete specific memories, purge all data at any time (`purge_all_user_memories()`), or revoke donation consent before offline candidate release.
- **Data Portability**: Users can export their stored memories in JSON format at any time.


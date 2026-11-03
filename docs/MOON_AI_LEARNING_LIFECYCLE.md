# Moon AI — Controlled Learning & Memory Lifecycle
### Strict Separation of User Memory, Feedback Data, Training Data & Model Parameters

**Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT`  
**Date:** November 2026

---

## 1. THE FOUR CORE CONCEPTS

To guarantee user privacy, model stability, and regulatory compliance, Moon AI enforces an absolute conceptual and technical boundary between four distinct layers:

```
+---------------------------------------------------------------------------------+
|                                 USER MEMORY                                     |
|  Private, user-consented key-value context for live session personalization.   |
|  NEVER automatically fed into model training manifests or neural weights.       |
+---------------------------------------------------------------------------------+
                                       |
                                       x  [PROHIBITED TO MERGE]
                                       |
+---------------------------------------------------------------------------------+
|                                FEEDBACK DATA                                    |
|  Explicit user corrections, ratings (thumbs up/down), and reported issues.      |
|  Stored in feedback logs for human inspection and offline triage.              |
+---------------------------------------------------------------------------------+
                                       |
                   [CONSENT + PRIVACY FILTERING + AUDIT]
                                       v
+---------------------------------------------------------------------------------+
|                                TRAINING DATA                                    |
|  Vetted, PII-scrubbed, injection-filtered examples in verified JSONL manifests. |
|  Explicitly eligible for scheduled batch model training.                        |
+---------------------------------------------------------------------------------+
                                       |
                           [OFFLINE SCHEDULED RUNS ONLY]
                                       v
+---------------------------------------------------------------------------------+
|                               MODEL PARAMETERS                                  |
|  Actual floating-point neural network weights (e.g. PyTorch state_dict).        |
|  UPDATED ONLY IN CONTROLLED OFFLINE RUNS — NEVER DURING LIVE CHATS.             |
+---------------------------------------------------------------------------------+
```

---

## 2. THE STAGED LEARNING PIPELINE

Every candidate data sample follows this strict progression:

1. **Interaction**: Live chat between user and Moon AI.
2. **Consent & Eligibility**: User must have explicitly opted in to contribute data. Private conversations are never ingested without affirmative donation.
3. **Privacy & Secret Filtering**: Automatic regex scrubbing for emails, phone numbers, SSNs, credit cards, and API keys.
4. **Prompt Injection Filtering**: Detects adversarial jailbreaks ("ignore previous instructions", "DAN mode").
5. **Quality & Formatting Check**: Validates substantive length, non-repetition, and coherence.
6. **Staging Approval**: Placed into candidate training staging.
7. **Scheduled Batch Training**: Controlled offline training run on approved manifests (zero live weight updates).
8. **Candidate Checkpoint**: New checkpoint saved in isolated candidate directory.
9. **Independent Evaluation Gate**: 12-category blind benchmark suite tests for accuracy and non-regression.
10. **Promotion or Rollback**: Only promoted to production if all safety and performance criteria are met. The previous production checkpoint is preserved for instant rollback.

---

## 3. USER SOVEREIGNTY RIGHTS

- **Explicit Opt-In**: Persistent memory and training data contribution are disabled by default.
- **Granular Deletion**: Users can view, delete specific memories, or purge all data at any time (`purge_all_user_memories()`).
- **Data Portability**: Users can export their stored memories in JSON format at any time.


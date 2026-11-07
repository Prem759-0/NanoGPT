# Moon AI — Kimi K3.1 Pro Teacher Integration Guide
### Provider Abstraction, Credential Management & Weights Tracking

**Architect:** Prem Gaikwad (India 🇮🇳)  
**Ownership Policy:** `TEACHER_STUDENT_RANDOM_INIT`  
**Teacher Model:** Kimi K3.1 Pro (Moonshot AI)  
**Date:** November 2026

---

## 1. OFFICIAL STATUS & SPECIFICATIONS

- **Official Announcement**: [Kimi K3.1 Pro Announcement](https://www.kimi.com/news/kimi-k3-1-pro)
- **Weights Release Schedule**: Full weights announced for **November 4, 2026**.
- **Important Policy**: Moon AI does not assume weights are available before verifying their formal public release and licensing terms.
- **Context Window**: Up to 128,000 tokens.
- **Key Strengths**: Long-context reasoning, multi-turn conversational nuance, code synthesis, agentic tool planning.

---

## 2. API INTEGRATION & SECURITY ARCHITECTURE

### 2.1 Credential Isolation
API keys are strictly managed via environment variables:
- Primary: `KIMI_API_KEY`
- Fallback: `MOON_TEACHER_API_KEY`

```bash
# Set in terminal (Linux/Mac)
export KIMI_API_KEY="sk-..."

# Set in PowerShell (Windows)
$env:KIMI_API_KEY="sk-..."
```

> [!IMPORTANT]
> - Never hardcode API keys in any Python file or config file.
> - The codebase redacts keys in all logs (`sk-...abcd`).
> - The repository's `.gitignore` excludes `.env` and credential files.

### 2.2 Disabled-by-Default External Calls
To prevent unintentional network traffic or cloud costs during local development and automated CI runs:
- `allow_external_calls: false` is configured by default in `teacher/config/teacher_config.yaml`.
- Calling `generate_*()` on `KimiProvider` with external calls disabled raises a controlled `TeacherProviderError`.
- For unit testing, `MockTeacherProvider` provides 100% deterministic, offline responses with zero network overhead.

### 2.3 Resiliency & Error Handling
`KimiProvider` automatically handles:
- **Rate Limits (HTTP 429)**: Retries with exponential backoff (`t = backoff * (2.0 ^ attempt)`).
- **Timeouts**: Configurable timeout (default: 30 seconds).
- **Outages (HTTP 500/502/503)**: Bounded retries up to `max_retries` (default: 3).
- **Malformed Payloads**: Validates JSON and extracts structured outputs or code blocks safely.

---

## 3. CHECKING OFFICIAL WEIGHTS RELEASE

When Moonshot AI releases weights on or after November 4, 2026, the adapter validates:
1. Exact repository URL and SHA-256 / Blake3 checksums.
2. License conditions (Commercial rights, attribution, synthetic data terms).
3. Compute sizing: A multi-hundred-billion parameter model cannot run on a 16 GB Colab T4. A local adapter will be activated only after hardware prerequisites and legal terms are confirmed.


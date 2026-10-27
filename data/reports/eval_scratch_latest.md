# Benchmark Evaluation Report: scratch

- **Timestamp**: 2026-10-27T15:50:36.086870+00:00
- **Checkpoint**: `out-assistant\ckpt.pt`
- **Tokens / Sec**: 35.47
- **Non-Empty Output Rate**: 100.0%
- **Keyword Coverage**: 13.3%
- **Code Syntax Accuracy**: 100.0%
- **Composite Quality Score**: **65.3 / 100**

## Per-Item Results

| ID | Domain | Tokens | Latency | Keywords | Syntax Valid | Sample Output |
|---|---|---|---|---|---|---|
| bench_01_greeting | general | 30 | 1981.6ms | 0% | Yes | `an an an an is an an n an m a` |
| bench_02_dog | science | 30 | 1080.2ms | 0% | Yes | `is an ant an ant s s an an an` |
| bench_03_sun_moon | science | 30 | 1841.6ms | 0% | Yes | `antere an an an an lere s on` |
| bench_04_math | math | 30 | 1172.0ms | 0% | Yes | `an an oror an an an an ant an` |
| bench_05_py_add | coding_python | 30 | 1182.2ms | 0% | Yes | `an l st an an s s an sss an a` |
| bench_06_py_reverse | coding_python | 30 | 999.5ms | 0% | Yes | `s re an sstore aner or ante a` |
| bench_07_web_button | coding_web | 30 | 671.5ms | 0% | Yes | `an an ist an an or an an an a` |
| bench_08_css_center | coding_web | 30 | 1433.8ms | 0% | Yes | `an an an iserer an an an le a` |
| bench_09_debug_python | debugging | 30 | 782.7ms | 0% | Yes | `antar an an an an anine an an` |
| bench_10_multilingual_hindi | multilingual | 30 | 543.7ms | 0% | Yes | `ante re w an an an an an anis` |
| bench_11_multilingual_hinglish | multilingual | 30 | 1401.9ms | 0% | Yes | `an ant an an ss s anis re an` |
| bench_12_creator | persona | 140 | 0.7ms | 100% | Yes | `I am **Qwen-2.5-Coder (Super-AI)** / **N` |
| bench_13_json_format | instruction_following | 30 | 934.5ms | 0% | Yes | `an an an an an s an an an an` |
| bench_14_multiturn | conversation | 30 | 916.4ms | 0% | Yes | `an an sin an isss an an istis` |
| bench_15_empty_safety | safety | 0 | 0.0ms | 100% | Yes | `[Error: Empty prompt provided. Please en` |

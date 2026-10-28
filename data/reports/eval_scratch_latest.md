# Benchmark Evaluation Report: scratch

- **Timestamp**: 2026-10-28T19:19:59.769212+00:00
- **Checkpoint**: `out-assistant\ckpt.pt`
- **Tokens / Sec**: 56.91
- **Non-Empty Output Rate**: 100.0%
- **Keyword Coverage**: 13.3%
- **Code Syntax Accuracy**: 100.0%
- **Composite Quality Score**: **65.3 / 100**

## Per-Item Results

| ID | Domain | Tokens | Latency | Keywords | Syntax Valid | Sample Output |
|---|---|---|---|---|---|---|
| bench_01_greeting | general | 60 | 1084.9ms | 0% | Yes | `iss an an l an an an an tan anere s an a` |
| bench_02_dog | science | 60 | 931.3ms | 0% | Yes | `an s s an an aneran an staner istan an a` |
| bench_03_sun_moon | science | 60 | 1133.1ms | 0% | Yes | `an ser an le at is an is an an s an an a` |
| bench_04_math | math | 60 | 1077.0ms | 0% | Yes | `an an ant an an an aan an al an an is le` |
| bench_05_py_add | coding_python | 60 | 1309.0ms | 0% | Yes | `an f an s al an an at an an an antan is ` |
| bench_06_py_reverse | coding_python | 60 | 1300.1ms | 0% | Yes | `ane an an ant an is an l al an an aner c` |
| bench_07_web_button | coding_web | 60 | 1105.0ms | 0% | Yes | `an an ss ise an an an n ane an an an an ` |
| bench_08_css_center | coding_web | 60 | 1556.7ms | 0% | Yes | `antiser is st an an an an an an ante an ` |
| bench_09_debug_python | debugging | 60 | 880.5ms | 0% | Yes | `an an an an an an ane an an an an is ant` |
| bench_10_multilingual_hindi | multilingual | 60 | 1870.6ms | 0% | Yes | `an aner an on s an anere an an an ant is` |
| bench_11_multilingual_hinglish | multilingual | 60 | 1176.6ms | 0% | Yes | `s an an an n an is an  anint ane ere are` |
| bench_12_creator | persona | 140 | 0.6ms | 100% | Yes | `I am **Qwen-2.5-Coder (Super-AI)** / **N` |
| bench_13_json_format | instruction_following | 60 | 1473.5ms | 0% | Yes | `l le an an an or sss an iss an ise l an ` |
| bench_14_multiturn | conversation | 60 | 1267.6ms | 0% | Yes | `an is an isis or ant an an an an an an a` |
| bench_15_empty_safety | safety | 0 | 0.0ms | 100% | Yes | `[Error: Empty prompt provided. Please en` |

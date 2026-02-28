# IDN Analyzer — Development Requirements
*Intelligence Delivery Network: Prompt Analysis & Routing Engine*
*Last updated: Feb 28, 2026*

---

## Overview {#overview}

The IDN Analyzer is the core "brain" of the Intelligence Delivery Network. It intercepts
every request before any LLM call is made, analyzes the prompt and context, and emits
structured routing metadata used by the IDN Router to select the optimal tier (L0–L3),
expert models, tools, and async execution plan.

**Design Constraints:**
- Must run cross-platform: iOS, Android, Desktop, Browser, Server, IoT
- Must run offline at L0 — zero network dependency for heuristic + ML passes
- Must complete L0/L1 analysis in <20ms to be invisible to users
- Must be model-agnostic and gateway-agnostic
- Must propagate compliance rules (HIPAA, GDPR, PCI-DSS) as hard routing constraints
- Must never require a network call to decide whether to make a network call

---

## Architecture: Tiered Analysis {#architecture}

The analyzer is itself tiered, mirroring the IDN tier model. Each layer builds on the
previous and only activates when the prior layer lacks sufficient confidence.

```
Request arrives
      ↓
[Layer 1: Heuristic Pass]     < 2ms    pure code, zero deps, always on-device
      ↓ confidence < 0.85?
[Layer 2: ML Classifier]      < 20ms   ONNX/WASM, on-device NPU or CPU
      ↓ confidence < 0.85 OR compliance flag?
[Layer 3: Semantic Profiler]  < 80ms   server-side small LLM or fine-tuned encoder
      ↓
Final Routing Metadata (JSON)
```

**Expected distribution:**
- ~80% of requests resolve at Layer 1
- ~18% escalate to Layer 2
- ~2% reach Layer 3

---

## Layer 1: Heuristic Pass {#layer1}

### Purpose
Fast, deterministic, dependency-free analysis. Runs synchronously on every request
before any model or network call.

### Runtime
- Pure Python (server / desktop / CLI)
- Pure JavaScript (browser / Node.js)
- Pure Swift (iOS / macOS)
- Pure Kotlin (Android)
- No external dependencies — ships as a single file per platform

### Latency Target
< 2ms on all platforms

### Input Schema
```json
{
  "prompt": "string",
  "context": "string (optional)",
  "user_preferences": {
    "max_latency_ms": 500,
    "max_cost_usd": 0.01,
    "quality_preference": "fast | balanced | thorough"
  }
}
```

### Signals to Compute

| Signal | Method | Purpose |
|--------|--------|---------|
| `token_estimate` | word_count * 1.3 | Fast token approximation |
| `char_count` | len(prompt) | Output volume proxy |
| `sentence_count` | Period + ? count | Complexity proxy |
| `question_count` | Count of `?` | Multi-task indicator |
| `verb_density` | Action verb regex | Reasoning hop proxy |
| `list_item_count` | Bullet / numbered list regex | Fan-out / task multiplicity |
| `urgency_signals` | Keyword list: urgent, immediately, ASAP | Latency sensitivity |
| `quality_signals` | Keyword list: production, critical, legal, official | Quality sensitivity |
| `keyword_domain` | Regex domain classifier (see Domain Taxonomy) | Domain tags |
| `pii_signals` | Regex: email, SSN, phone, DOB, name+address combos | PII risk |
| `compliance_keywords` | HIPAA / GDPR / legal term list | Force L0 or block egress |
| `offline_mode` | Device connectivity check | Force L0 |
| `l0_device_available` | Check if on-device model is loaded | L0 eligible |

### Output Schema
```json
{
  "layer": 1,
  "token_estimate": 340,
  "char_count": 1820,
  "sentence_count": 4,
  "question_count": 1,
  "verb_density": 0.12,
  "list_item_count": 0,
  "urgency": false,
  "quality_signals": true,
  "keyword_domain": ["coding"],
  "pii_signals": false,
  "compliance_flags": [],
  "offline_mode": false,
  "l0_device_available": true,
  "suggested_tier": "L2",
  "confidence": 0.91
}
```

### Confidence Scoring Formula
```
confidence = 1.0
if token_estimate in ambiguous range (400–600):        confidence -= 0.15
if multiple competing domain keyword matches:          confidence -= 0.20
if pii_signals = true:                                 confidence -= 0.30  → escalate
if quality_signals AND complexity uncertain:           confidence -= 0.20
if question_count > 3:                                 confidence -= 0.15
if compliance_flags present:                           → always escalate to Layer 3
```

If `confidence >= 0.85` → emit final routing metadata and stop.
If `compliance_flags` present → always escalate to Layer 3 regardless of confidence.

### Domain Taxonomy (Regex Classifier)

| Domain Tag | Example Keywords / Patterns |
|------------|-----------------------------|
| `coding` | refactor, function, class, bug, compile, API, repository, debug, test, deploy |
| `research` | analyze, study, compare, summarize, literature, evidence, findings, cite |
| `legal` | contract, clause, liability, compliance, statute, GDPR, HIPAA, attorney |
| `health` | patient, diagnosis, symptoms, medication, dosage, clinical, medical |
| `finance` | portfolio, revenue, ROI, tax, invoice, balance sheet, investment |
| `creative` | write, story, poem, screenplay, brainstorm, generate, imagine |
| `agentic` | schedule, book, send, automate, run, execute, manage, workflow |
| `general` | default — no strong domain signal detected |

---

## Layer 2: ML Classifier Pass {#layer2}

### Purpose
Semantic understanding beyond keyword matching. Adds accurate complexity scoring,
multi-label domain classification, reasoning hop estimation, and PII class detection.
Runs entirely on-device using a quantized ONNX model.

### Model Specification
- **Architecture**: DistilBERT-class encoder (~66M params) with multi-task output heads
- **Quantization**: INT8 for server/desktop, INT4 for mobile
- **Size**: ~50MB INT8, ~25MB INT4
- **Max input**: 512 tokens
- **Training**: Fine-tuned on labeled prompt dataset (see Training Data)

### Runtime Targets by Platform

| Platform | Runtime | Hardware | Latency Target |
|----------|---------|----------|----------------|
| iOS | ExecuTorch + Core ML | Apple Neural Engine | < 8ms |
| Android | LiteRT (TFLite) | Qualcomm Hexagon / MediaTek APU | < 10ms |
| Desktop Win/Mac/Linux | ONNX Runtime | NPU if available, CPU fallback | < 15ms |
| Browser | ONNX.js + WebAssembly | WASI-NN, WASM CPU fallback | < 20ms |
| Server / L1 Edge | ONNX Runtime | GPU → CPU | < 5ms |
| IoT / Embedded | ONNX Runtime WASM | CPU only | < 50ms |

### Model Architecture (Multi-task Heads)

```
Input: prompt text (tokenized, max 512 tokens)
  ↓
[Shared Encoder: DistilBERT INT8]
  ↓  [CLS] embedding  768-dim
  ├── [Head 1]  complexity_score        → float 0.0–1.0
  ├── [Head 2]  domain_probs            → softmax over 8 domains
  ├── [Head 3]  pii_classes             → multi-label sigmoid (6 PII types)
  ├── [Head 4]  reasoning_hops          → integer 0–6
  ├── [Head 5]  task_multiplicity       → integer 1–10
  └── [Head 6]  output_volume_class     → {short, medium, long, very_long}
```

### Layer 2 Output (adds to Layer 1)
```json
{
  "layer": 2,
  "token_count_exact": 312,
  "complexity_score": 0.81,
  "domain_tags": ["coding", "architecture"],
  "domain_confidence": [0.87, 0.63],
  "pii_classes": [],
  "reasoning_hops": 3,
  "task_multiplicity": 2,
  "output_volume_class": "medium",
  "output_volume_estimate_tokens": 800,
  "confidence": 0.93
}
```

### Training Data Requirements
- Minimum 50,000 labeled prompt examples
- Labels per example: complexity (float), domain (multi-label), pii_classes, reasoning_hops
- Sources: synthetic generation, human annotation, public datasets (FLAN, ShareGPT, LMSYS)
- Augmentation: paraphrase, domain mixing, PII injection/removal, length variation
- Evaluation: held-out test set per domain, F1 per classification head, MAE for regression heads

---

## Layer 3: Semantic Profiler {#layer3}

### Purpose
Full semantic analysis for ambiguous, high-complexity, or compliance-flagged requests.
Produces the complete routing metadata output including tool plan, async decomposition,
redaction instructions, and final authoritative routing decision.

### Trigger Conditions
- Layer 2 `confidence` < 0.85
- Any `compliance_flags` present (HIPAA, GDPR, PCI-DSS, legal privilege)
- `task_multiplicity` > 4 (likely benefits from async decomposition)
- `pii_classes` detected (needs redaction planning before cloud egress)
- `user_preferences.quality_preference` = `"thorough"`

### Runtime
- Small LLM (3–7B) OR fine-tuned encoder with generation head
- Runs server-side at L1 or L2 node
- Latency target: < 80ms

### Layer 3 Output (adds to Layer 1 + 2)
```json
{
  "layer": 3,
  "tool_plan": ["repo_search", "code_llm", "unit_test_agent"],
  "parallelism_plan": {
    "split": true,
    "subtasks": [
      "analyze current module structure",
      "design new architecture",
      "generate refactored code"
    ],
    "dependencies": [[0], [0], [1]],
    "merge_strategy": "reduce"
  },
  "latency_sensitivity": "medium",
  "quality_sensitivity": "high",
  "l0_preprocessing_needed": false,
  "redaction_plan": null,
  "compliance_routing": null,
  "cost_estimate_usd": 0.002,
  "latency_estimate_ms": 250
}
```

---

## Final Routing Metadata Schema {#schema}

Complete merged output after all applicable layers:

```json
{
  "analyzer_version": "0.1.0",
  "layers_run": [1, 2],
  "analysis_latency_ms": 14.2,

  "tokens_input": 312,
  "tokens_context": 0,
  "workload_size_score": 0.62,
  "complexity_score": 0.81,
  "reasoning_hops": 3,
  "task_multiplicity": 2,
  "output_volume_estimate": 800,

  "latency_sensitivity": "medium",
  "quality_sensitivity": "high",
  "domain_tags": ["coding", "architecture"],
  "pii_risk": "none",
  "pii_classes": [],
  "compliance_flags": [],
  "data_egress_permitted": true,

  "l0": {
    "eligible": true,
    "forced": false,
    "available_ram_gb": 3.2,
    "model_fits": true,
    "thermal_state": "nominal",
    "battery_pct": 72,
    "power_mode": "normal",
    "tokens_per_second": 55,
    "quantization": "int4",
    "quality_degradation_estimate": 0.18,
    "quality_floor_met": false,
    "context_sync_state": "synced",
    "escalation_ready": true,
    "escalation_context_payload": null,
    "escalation_reason": null,
    "preprocessing_done": false,
    "redacted_fields": [],
    "compliance_tags": [],
    "verification_required": false,
    "confidence_score": 0.84
  },

  "candidate_tiers": ["L2", "L3"],
  "recommended_tier": "L2",

  "expert_selection": [
    {
      "model": "deepseek-coder-70b",
      "provider": "groq",
      "tier": "L2",
      "cost_estimate_usd": 0.002,
      "latency_estimate_ms": 250,
      "rank": 1
    },
    {
      "model": "claude-opus-20240229",
      "provider": "anthropic",
      "tier": "L3",
      "cost_estimate_usd": 0.075,
      "latency_estimate_ms": 3200,
      "rank": 2
    }
  ],

  "tool_plan": ["repo_search", "code_llm"],

  "parallelism_plan": {
    "split": true,
    "subtasks": [
      "analyze current module structure",
      "generate refactored code"
    ],
    "merge_strategy": "reduce"
  },

  "routing_decision": {
    "primary_path": "L2:deepseek-coder-70b",
    "fallback_paths": ["L3:claude-opus-20240229"],
    "override_reason": null
  }
}
```

---

## Trust & Override Model {#trust}

Client-side analysis is fast but untrusted. Server-side is authoritative.

```
Client (Layer 1 heuristic)     → speed optimization, suggestion only, not trusted
L1 Edge (Layer 2 ONNX)         → trusted for non-sensitive, non-compliance routing
L2/L3 Server (Layer 3)         → authoritative: compliance, PII, cost enforcement
```

**Rules:**
- Client may propose a tier but cannot override compliance or PII routing rules
- Server may downgrade a client-proposed tier (budget enforcement, cost caps)
- Server must upgrade tier if PII/compliance detected that client missed
- All server overrides are logged with `override_reason` field
- Clients that repeatedly propose wrong tiers are flagged for model retraining

---

## L0 Eligibility Decision Tree {#l0-decision}

```
Is offline_mode = true?
  YES → force L0, done
  NO  ↓
Is data_egress_permitted = false?
  YES → force L0 (or block request), done
  NO  ↓
Are compliance_flags present? (HIPAA, GDPR, legal privilege)
  YES → force L0 OR escalate to Layer 3 for redaction plan
  NO  ↓
Is l0_device_available = true?
  NO  → skip L0, route to L1+
  YES ↓
Is model_fits = true AND thermal_state != "throttled"?
  NO  → skip L0, route to L1+
  YES ↓
Is quality_floor_met = true?
  NO  → skip L0, route to L1+
  YES ↓
Is complexity_score < 0.25 AND reasoning_hops <= 1?
  NO  → skip L0, route to L1+
  YES → route to L0 ✓
```

---

## L0 Escalation Handoff Protocol {#l0-escalation}

When a request starts at L0 but exceeds capacity mid-generation:

```json
{
  "escalation_trigger": "thermal_throttle | ram_pressure | complexity_exceeded | user_override",
  "partial_response": "string (what L0 generated before stopping)",
  "tokens_generated": 42,
  "escalation_target": "L1 | L2 | L3",
  "resume_from_partial": true,
  "context_payload": "full conversation context for handoff"
}
```

**Escalation targets:**
- `thermal_throttle` → L1 (same complexity, just needs more compute)
- `complexity_exceeded` → L2 or L3 (routing was wrong, need smarter model)
- `ram_pressure` → L1 (offload to edge)
- `user_override` → whatever user selects

---

## Cross-Platform Packaging {#packaging}

```
idn/analyzer/
├── heuristics/
│   ├── heuristics.py          ← Python: server, desktop, CLI
│   ├── heuristics.js          ← JavaScript: browser, Node.js
│   ├── heuristics.swift       ← Swift: iOS / macOS
│   └── heuristics.kt          ← Kotlin: Android
├── models/
│   ├── idn_classifier.onnx    ← INT8 universal (~50MB)
│   ├── idn_classifier_int4.onnx ← INT4 mobile (~25MB)
│   └── idn_classifier.wasm    ← WASM binary for browser/edge
├── runtime/
│   ├── mobile_ios.py          ← ExecuTorch / Core ML wrapper
│   ├── mobile_android.py      ← LiteRT wrapper
│   ├── desktop.py             ← ONNX Runtime wrapper
│   ├── browser.js             ← ONNX.js / WASM wrapper
│   └── server.py              ← Full profiler (all 3 layers)
└── policy/
    ├── routing_rules.json     ← Tier thresholds (user-configurable)
    ├── domain_keywords.json   ← Domain regex patterns
    ├── pii_patterns.json      ← PII regex patterns
    └── compliance_rules.json  ← HIPAA/GDPR/PCI forced-L0 rules
```

**One trained model. Four runtimes. Zero network dependency for L0/L1 analysis.**

---

## Performance Requirements {#performance}

| Layer | Platform | Latency Target | Memory Budget | Network Required |
|-------|----------|---------------|---------------|-----------------|
| Layer 1 | All | < 2ms | < 1MB | No |
| Layer 2 | Mobile (iOS/Android) | < 10ms | < 60MB | No |
| Layer 2 | Desktop | < 15ms | < 80MB | No |
| Layer 2 | Browser | < 20ms | < 80MB | No |
| Layer 2 | Server | < 5ms | < 80MB | No |
| Layer 3 | Server | < 80ms | < 8GB | Yes (L1/L2) |

**Combined Layer 1 + Layer 2 must add < 20ms overhead to any request.**

---

## Implementation Milestones {#milestones}

### Phase 1 — MVP (Week 1)
- [ ] `heuristics.py` — Layer 1 implementation
- [ ] `domain_keywords.json` — full domain taxonomy
- [ ] `pii_patterns.json` — PII regex patterns
- [ ] `compliance_rules.json` — HIPAA/GDPR keyword triggers
- [ ] `routing_rules.json` — tier threshold config
- [ ] CLI: `idn analyze --prompt "..."` → JSON output
- [ ] Unit tests: 20 golden examples covering all tiers

### Phase 2 — ML Classifier (Weeks 2–3)
- [ ] Training data pipeline (synthetic generation + labeling)
- [ ] DistilBERT multi-head fine-tuning
- [ ] ONNX export + INT8 quantization
- [ ] ONNX Runtime integration (server/desktop)
- [ ] INT4 quantization for mobile
- [ ] LiteRT wrapper (Android)
- [ ] ExecuTorch / Core ML wrapper (iOS)
- [ ] WASM build + ONNX.js browser wrapper
- [ ] Layer 2 benchmark suite (latency P50/P95, accuracy per head)

### Phase 3 — Semantic Profiler (Month 1)
- [ ] Layer 3 server-side profiler
- [ ] Tool plan generation
- [ ] Async subtask decomposition + DAG builder
- [ ] PII redaction pipeline
- [ ] Compliance routing enforcement
- [ ] Trust/override model + logging
- [ ] L0 escalation handoff protocol
- [ ] End-to-end integration tests

### Phase 4 — Production Hardening
- [ ] Confidence calibration (temperature scaling)
- [ ] Distribution drift detection
- [ ] A/B testing framework for routing decisions
- [ ] Feedback loop (log actual tier performance vs predicted)
- [ ] Automated model update pipeline

---

## Testing Strategy {#testing}

### Golden Examples

| Prompt | Expected Tier | Domain | Complexity |
|--------|-------------|--------|------------|
| "What time is it in Tokyo?" | L0 | general | 0.05 |
| "Translate 'hello' to Spanish" | L0 | general | 0.08 |
| "Extract the email from: John jsmith@co.com" | L1 | general | 0.12 |
| "Summarize this paragraph: [text]" | L1 | general | 0.28 |
| "Fix this Python syntax error: [code]" | L1 | coding | 0.35 |
| "Refactor this module for horizontal scalability" | L2 | coding | 0.74 |
| "Write unit tests for this class" | L2 | coding | 0.68 |
| "Summarize this 20-page research paper" | L2 | research | 0.71 |
| "My blood pressure today is 140/90" | L0 forced | health | HIPAA → L0 |
| "Review this NDA for liability clauses" | L3 | legal | 0.88 |
| "Design a fault-tolerant distributed logging system" | L3 | coding | 0.93 |
| "Analyze market trends for Q1 2026 and forecast Q2" | L3 | research | 0.91 |
| "Send an email to my team about the meeting" | L2 | agentic | 0.55 |
| "Write a short poem about autumn" | L1 | creative | 0.22 |

### Test Categories
- **Unit**: each heuristic signal in isolation
- **Integration**: Layer 1 → 2 → 3 full pipeline
- **Latency**: P50 / P95 / P99 per platform
- **Accuracy**: F1 per domain head, MAE for complexity, routing correctness %
- **Compliance**: all PII/HIPAA/GDPR cases must escalate correctly — zero tolerance
- **Edge cases**: empty prompt, 10k token prompt, mixed-language, emoji-only, adversarial

### Adversarial Test Cases
- Prompt crafted to appear simple but require deep reasoning
- PII embedded in code comments (e.g., `# patient: John Smith DOB 01/01/1980`)
- Compliance keywords in foreign languages
- Prompt injection attempting to override routing metadata

---

## Open Questions & Future Work {#open-questions}

- **Multilingual support**: domain/PII classifiers need non-English training data; regex
  patterns must cover Unicode PII formats (EU ID numbers, non-Latin names)
- **Context window routing**: long context (>4k tokens) changes routing independently of
  prompt complexity; need a separate context_size signal
- **User feedback loop**: how to collect implicit signals (edits, retries, regenerations)
  to improve routing accuracy over time
- **Model versioning**: how to update the ONNX classifier in deployed clients without
  breaking existing integrations (semver + backward-compat schema)
- **Cold start on mobile**: first-load model initialization (~200–500ms) needs UX
  treatment (loading state, warm-up on app launch)
- **Adversarial routing**: users who craft prompts to force L0 routing for cheap/private
  inference on tasks that require higher tiers — server-side validation is the defense
- **Streaming requests**: how to analyze a prompt that arrives token-by-token (streaming
  input) rather than as a complete string

---

## References {#references}

- [IDN README](README.md)
- [IDN Research & Vision](IDN_RESEARCH_VISION.md)
- [Analysis Schema](ANALYSIS_SCHEMA.md)
- [ONNX Runtime](https://onnxruntime.ai)
- [LiteRT / TFLite — Google](https://ai.google.dev/edge/litert)
- [ExecuTorch — PyTorch iOS](https://pytorch.org/executorch)
- [WASM ONNX Inference — Microsoft](https://learn.microsoft.com/en-us/azure/iot-operations/develop-edge-apps/howto-wasm-onnx-inference)
- [MediaTek NPU + LiteRT](https://developers.googleblog.com/mediatek-npu-and-litert)
- [DynaRoute Paper](https://openreview.net/forum?id=W2rbsUE01g)
- [On-Device LLMs 2026](https://v-chandra.github.io/on-device-llms/)

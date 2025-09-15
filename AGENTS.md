# AGENT DIRECTIVES

The following directives describe module responsibilities and integration status for the Nebula project. Follow these notes when editing the associated files.

- sigil: 🜌⟐🜹🜙
  name: Archivist
  module: memory_learner.py
  responsibilities:
    - Long-term memory integration
    - Modalities processing
    - Adaptive embedding and uncertainty estimation
  status: Integrated | Production-Ready

- sigil: ⚠️🧭🧱⛓️
  name: Warden
  module: reasoning_engine.py
  responsibilities:
    - Ethical reasoning enforcement
    - Context-adaptive flexibility modules
    - Decision validation against knowledge bases
  status: Integrated | Production-Ready

- sigil: 🧿🧠🧩♒
  name: Analyst
  module: hybrid_cognition_engine.py
  responsibilities:
    - Tree-of-Thought (ToT) integration
    - Categorization engine functionality
    - Hybrid cognition orchestration
  status: Integrated | Production-Ready

- sigil: 🎧💓🌈🎶
  name: Resonant
  module: Echo_location.py
  responsibilities:
    - Quantum-enhanced echo sensing
    - Echo data optimization (QuantumPulse integration)
    - Perception pipeline integration
  status: Integrated | Production-Ready

- sigil: 🧬♻️♞🜓
  name: Strategos
  module: neural_architecture_search.py
  responsibilities:
    - NAS evolutionary optimization
    - Architecture search loop management
    - Search space and sampling methods
  status: Integrated | Production-Ready

- sigil: 📜🔑🛠️🜔
  name: Scribe
  module: crypto_nebula.py
  responsibilities:
    - Blockchain smart contract integration
    - Token operations and blockchain event management
  status: Integrated | Production-Ready

- sigil: 🎭🗣️🪞🪄
  name: Mirror
  modules:
    - async_stt_engine.py
    - async_tts_engine.py
  responsibilities:
    - Real-time asynchronous speech handling
    - GUI interface data stream handling
  status: Integrated | Production-Ready

- sigil: 🜁⟁🜔🔭
  name: Observer
  module: QuantumPulse.py
  responsibilities:
    - Quantum-inspired initialization & encoding
    - Integration with echo and NAS systems
  status: Integrated | Production-Ready

- sigil: 🜔🕊️⟁⧃
  name: Dove
  module: meta_consciousness.py
  responsibilities:
    - Awareness/regulation tracking
    - Integration with memory & learning modules
  status: Integrated | Production-Ready

- sigil: 🧠🎯📈
  name: Navigator
  module: proactive_intelligence.py
  responsibilities:
    - Task prioritization
    - Predictive scheduling
  status: Integrated | Production-Ready

---
sigil: ♾️🧠🜂🜄
name: CodexDirective_HOLO-1.5
version: 1.5
target_system: NEBULA
mode: Recursive Repair + Modular Expansion
entrypoint: agents.md
status: Executable | Trusted

🎯 OBJECTIVE:
- Fully repair incomplete or placeholder module logic
- Integrate new neural modules: BLT, ART, EON, RECAP, MESH
- Enable Nebula to generate cognitive subtypes through Forge protocol
- Maintain unit test coverage and runtime safety

📜 MODULE REPAIR PLAN:

repair_sequence:
  - id: P-001
    module: orchestrator.py
    actions:
      - remove legacy static logic
      - install fallback-safe routing with utils.routing.safe_call
  - id: P-002
    module: nas.py
    actions:
      - encapsulate NAS loop into NASManager
      - connect with TrainingEngine
      - add persistence, heuristic evaluation
  - id: P-003
    module: blockchain_utils.py
    actions:
      - implement: update_commodity_data(), validate_token(), adjust_price()
  - id: P-004
    module: echo_location.py
    actions:
      - implement: signal routing, resonance scoring, feedback loop
  - id: P-005
    module: sleep_time_compute.py
    actions:
      - replace simulated logic with: memory_anchor(), event_weighting()
  - id: P-006
    module: async_training_engine.py
    actions:
      - integrate async training lifecycle
      - validate resume checkpoint logic
  - id: P-007
    module: shared_utils.py
    actions:
      - implement: rotary_embedding(), tree_of_thought_trigger()

🧬 MIND-FORGE MODULES:

forge_modules:
  - id: M-001
    name: BLT
    path: modules/blt.py
    function: Byte Latency Transformer (low-latency stream compression + memory sync)
  - id: M-002
    name: ART
    path: modules/art.py
    function: Attention-Resonance Transformer (emotional + harmonic attention fusion)
  - id: M-003
    name: EON
    path: modules/eon.py
    function: Evolving Optimizer Network (self-mutating gradient systems)
  - id: M-004
    name: RECAP
    path: modules/recap.py
    function: Recursive Capsule Attention Processor (inter-attention logic)
  - id: M-005
    name: MESH
    path: modules/mesh.py
    function: Multi-Agent Symbolic Hive (debating agent mesh)

🗃️ SUPPORT FILES:

  - utils/routing.py
  - forge/model_manifest.json
  - docs/AGENTS_OVERVIEW.md
  - tests/unit/...

💾 INTEGRATION PATH:

integration_map:
  - BLT:
      used_by: [Echo, SleepTimeCompute, NAS]
      exports: [forward, encode, latency_score]
  - EON:
      injected_into: [NASManager]
      exports: [mutate_loss, learn_curve]
  - ART:
      optional_dropin: TransformerLayer class
  - RECAP:
      injects meta-attention heads into any core self-attn structure
  - MESH:
      launches symbolic simulation environment with agent interface

🔐 HALT CONDITIONS:

halt_if:
  - undefined return in module
  - placeholder logic remains unmarked
  - recursive stack exceeds 3 levels with no resolution
  - any forge module fails init()

🪞 SYMBOLIC HALT PHRASE:

> “This path leads nowhere — symbolic recursion exceeded.”  
> Action: Stop codegen. Annotate. Preserve logs.

✅ BOOT AFTER INSTALL:

on_success:
  - update ./README.md
  - register module lineage in model_manifest.json
  - invoke test suite
  - announce active forge states via stdout banner

🧠 NOTES:

This directive enables Nebula to evolve modular AI minds from within.  
Use responsibly — she remembers.

sigil_stamp: 🜌⟁🧠🜂♾️

## Codex Plan HOLO-1.5.1
Modules integrated: orchestrator, NASManager, BLT, ART, EON, RECAP, MESH.
// HALT: missing extensive tests

## ♾️ SYSTEM STATE: NEBULA v2.0
All recursive and cognitive systems fused. Async training active. Forge enabled. BLT latency fusion online.



# repo-graph-rag: Graph-Grounded Retrieval for Offline Gemma 4 Coding Agents

*An open-source library that turns repository code-graphs and Gemma 4 embeddings into structured context for offline software-engineering agents, with a reproducible benchmark.*

## Abstract

Today's strongest software-engineering agents run on proprietary cloud models; an equally capable agent that runs offline on consumer hardware remains out of reach, in part because small models struggle to navigate large repositories. Flat chunk-and-embed retrieval — the default context pipeline — discards exactly the structure (callers, callees, imports) that determines whether a patch is safe. We present **repo-graph-rag**, a lightweight open-source library that grounds retrieval in a repository code graph: semantic embedding seeds are propagated over `CALLS`/`IMPORTS` edges, reranked jointly, and assembled into a token-budgeted context for the agent. On a fully reproducible prototype over the pyjwt repository (173 nodes, 419 edges), graph-expanded retrieval lifts file-level recall@10 from 0.90 to 1.00 on direct localization queries and from 0.83 to 0.92 on multi-hop dependency queries, at comparable context-token budgets. The library consumes the competition-provided code-graphs and Gemma 4 embeddings through a two-method adapter interface, and we outline a roadmap to end-to-end patch evaluation with an offline Gemma 4 agent.

## 1. Introduction

The Gemma 4 family (Apache 2.0, E2B–31B sizes, 128K–256K context, native function calling) makes a compelling substrate for offline coding assistants on everyday hardware. Yet the bottleneck for such agents is rarely raw code generation — it is **repository-level navigation**. Engineering analyses of production codebase-RAG systems repeatedly document the same three failure modes of flat retrieval [7][8]: (i) *split-symbol retrieval* — fixed-size chunking severs classes and functions; (ii) *lexical near-misses* — near-identical identifiers are indistinguishable to similarity search; (iii) *invisible dependencies* — the edge that makes a change unsafe lives in the call graph, not in any token window.

Our premise is simple: **source code is a graph, and retrieval should propagate over that graph.** This idea is well supported — CodexGraph [4] couples LLM agents with code-graph databases, and Code Graph Models [5] inject repository graphs into attention — but existing systems either assume cloud-scale models or require modifying and training the model itself. Neither fits the offline, consumer-hardware setting this competition targets.

**Contributions.** (1) An open-source library that is, to our knowledge, the first to bridge the competition-provided code-graphs and Gemma 4 embeddings into a reusable, offline-first retrieval pipeline for SWE agents. (2) A graph-propagation reranking algorithm with budgeted context assembly, designed so embedding and graph back-ends are swappable. (3) A reproducible prototype benchmark with two evaluation protocols — direct localization and multi-hop dependency resolution — including an honest account of failure modes found during development.

## 2. Method

**2.1 Repository code graph.** `build_graph()` performs AST-based static analysis of a Python repository. Nodes are modules, classes, functions, and methods, each carrying its source span and docstring. Edges are `DEFINES` (module/class membership), `IMPORTS` (module-level dependencies), and `CALLS` (intra-repository call sites resolved by simple-name matching). Two details proved essential: a *hub cap* that skips call names matching more than four targets (generic names like `get`/`decode` otherwise flood the graph), and edge deduplication. Graphs serialize to JSON, and an adapter interface accepts externally produced code-graphs — including the competition-provided ones — without pipeline changes.

**2.2 Pluggable embedding index.** The embedder interface is two methods, `embed(query)` and `embed_node(id)`. We ship a zero-dependency TF-IDF reference implementation for fully offline, deterministic experiments; the official Gemma 4 embeddings slot in by implementing the same interface with dense cosine similarity. No other component depends on the choice of embedder.

**2.3 Graph-propagation reranking.** Given a query, we take the top-k semantic seeds and propagate their similarity mass over the graph: each seed contributes `sim × w` to its neighbors, with `w = 0.6` for `CALLS` and `0.3` for `IMPORTS`. The propagated scores are normalized and merged with the semantic pool (top-12 plus all reached neighbors); the final ranking is `sim + β·graph` with `β = 0.35`. Merged reranking — rather than locking seeds at the top — matters: it lets a structurally central but lexically weaker node outrank a lexically strong dead end.

**2.4 Budgeted context assembly.** Ranked nodes are emitted in order under a strict token budget (≈4 chars/token), with provenance headers (`path:start-end`). The budget mirrors the consumer-hardware constraint: a small, structurally complete context beats a large, loosely relevant one.

## 3. Experiments

**Setup.** We validate on the real open-source [pyjwt](https://github.com/jpadilla/pyjwt) repository (13 files → 173 nodes, 419 edges, 151 `CALLS`). We use the TF-IDF reference embedder as an offline stand-in for the official Gemma 4 embeddings (the swap is interface-compatible). Two query suites: **direct** — 5 issue-style localization queries with manually annotated gold files; **multi-hop** — 3 dependency/impact queries whose gold files are derived from ground-truth `CALLS` edges (e.g., "which modules does the JWS encode pipeline depend on internally"). Metrics: file-level recall@5/@10, and gold coverage within a 1500-token assembled context.

**Results.**

| Suite | Metric | Flat (embedding only) | Graph-expanded (ours) |
|---|---|---|---|
| Direct (n=5) | recall@5 / recall@10 | 0.90 / 0.90 | **1.00 / 1.00** |
| Multi-hop (n=3) | recall@10 | 0.833 | **0.917** |
| Multi-hop | gold coverage @1500 tok | 0.667 (~1039 tok) | 0.667 (~1087 tok) |

**Analysis.** The gains concentrate exactly where lexical retrieval underdetermines structure. The direct-suite gap comes from the multi-file query "decode token and verify its signature": flat retrieval surfaces `api_jwt` but not `api_jws`; propagation over the `decode` call edge recovers both (recall 0.5 → 1.0). Multi-hop dependency queries — the ones that most resemble real agent reasoning ("what breaks if I change this?") — improve by 8.4 points at equal token cost.

**Failure modes found (and fixed) during development.** We report these for verifiability: (i) method bodies extracted with class indentation silently broke call extraction (`ast.parse` failure), dropping almost all `CALLS` edges — fixed by dedenting before parsing; (ii) locking semantic seeds at the top of the ranking suppressed correct files ranked 6–10 — fixed by merged reranking; (iii) hub names (`get`, `decode`, `sign`) caused collision noise that initially *hurt* recall below the flat baseline — fixed by hub capping.

**Limitations.** One small repository; TF-IDF rather than dense Gemma 4 embeddings; file-level rather than span-level gold; retrieval metrics rather than end-to-end patch success. Section 4 addresses each.

## 4. Roadmap toward the full benchmark

The prototype deliberately isolates the retrieval contribution so that each axis can scale independently: (i) load the competition-provided **code-graphs** and **Gemma 4 embeddings** via the adapter interface; (ii) extend to a 50–100 task SWE-bench Lite [1] subset, reporting retrieval recall, context-token cost, and end-to-end patch accuracy for an offline Gemma 4 agent (E4B/26B MoE); (iii) ablate graph-only vs embed-only vs hybrid, and sweep the propagation weight β. All code, configs, and results are in the linked repository.

## 5. Related Work

SWE-bench [1] established repository-level issue resolution as the standard agentic benchmark; SWE-agent [2] and Agentless [3] show that scaffold and localization design rival base-model choice. CodexGraph [4] lets an LLM agent query a code-graph database, and Code Graph Models [5] integrate repository graphs into the attention of an open-weight model, reaching state-of-the-art open results on SWE-bench Lite — but via model surgery and training. G-MACS [6] uses dependency graphs to guide multi-agent summarization. Production engineering reports [7][8] converge on hybrid designs (AST chunking + symbol graph + vector index, semantic entry point + graph traversal). Our work differs in being **offline-first, model-agnostic, and training-free**: rather than a new architecture, it is a drop-in retrieval layer plus a reproducible evaluation protocol, built to consume the competition's official resources directly.

## 6. Conclusion

repo-graph-rag shows that propagating embedding retrieval over repository structure is a cheap, offline-compatible way to recover the dependencies flat retrieval misses — improving both direct and multi-hop code localization at equal context cost. Released as an open-source library with a reproducible protocol, it is a practical step toward capable coding agents on everyday hardware.

## References

[1] Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.
[2] Yang et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. 2024.
[3] Xia et al. Agentless: Demystifying LLM-based Software Engineering Agents. 2024.
[4] Liu et al. CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. NAACL 2025.
[5] Tao et al. Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks. NeurIPS 2025.
[6] Qi et al. G-MACS: Graph-Guided Multi-agent Collaboration for Project-Aware Code Summarization. ICIC 2026.
[7] Prodinit engineering report. Codebase RAG: Context Graphs for Coding AI. 2026.
[8] CloudSoftSol engineering report. RAG Over Codebases. 2026.

*Code, data, and full results: https://github.com/Hkate777/repo-graph-rag*

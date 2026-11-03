"""run_demo.py — 端到端演示：在公开仓库 pyjwt 上对比
flat 嵌入检索 vs 图扩展检索（本方法）。
用法：python experiments/run_demo.py
输出：results/results.json + results/fig_retrieval_comparison.png
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repo_graph_rag import GraphRetriever, TfidfEmbedder, build_graph, file_of

REPO = os.path.join(BASE, "data", "public_repos", "pyjwt")
OUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

DIRECT = [
    ("token is expired, expiration claim validation fails", {"jwt/api_jwt.py"}),
    ("HMAC RSA ECDSA signing and verification algorithm", {"jwt/algorithms.py"}),
    ("fetch JSON web keys from JWKS endpoint", {"jwt/jwks_client.py"}),
    ("decode token and verify its signature", {"jwt/api_jws.py", "jwt/api_jwt.py"}),
    ("base64url encode decode with padding", {"jwt/utils.py"}),
]

MULTI_HOP = [  # (query, seed_node_suffix, direction)
    ("which modules does the JWS encode pipeline depend on internally", "PyJWS.encode", "callees"),
    ("what code paths call into the base64url decode helper", "base64url_decode", "callers"),
    ("which components does the top-level jwt decode flow invoke", "PyJWT.decode", "callees"),
]


def ensure_repo():
    if not os.path.isdir(REPO):
        os.makedirs(os.path.dirname(REPO), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/jpadilla/pyjwt.git", REPO],
            check=True,
        )


def files_of(nodes):
    seen, out = set(), []
    for nid in nodes:
        f = file_of(nid)
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def recall_at(files, gold, k):
    return len(set(files[:k]) & gold) / len(gold) if gold else 0.0


def main():
    ensure_repo()
    g = build_graph(REPO, "jwt")
    g.to_json(os.path.join(OUT_DIR, "pyjwt_graph.json"))

    callable_ids = [nid for nid, n in g.nodes.items() if n.type in ("function", "method")]
    docs = [g.nodes[nid].doc + "\n" + g.nodes[nid].code for nid in callable_ids]
    emb = TfidfEmbedder()
    emb.fit(callable_ids, docs)
    r = GraphRetriever(g, emb)

    adj_calls = {}
    for s, d, t in g.edges:
        if t == "CALLS":
            adj_calls.setdefault(s, []).append(d)

    rows_direct = []
    for q, gold in DIRECT:
        flat = r.flat_retrieve(q, k=10)
        graph = r.graph_retrieve(q, k_seeds=5, out_k=10)
        f_files, g_files = files_of(flat), files_of(graph)
        rows_direct.append({
            "query": q, "gold": sorted(gold),
            "flat_recall@5": recall_at(f_files, gold, 5),
            "flat_recall@10": recall_at(f_files, gold, 10),
            "graph_recall@5": recall_at(g_files, gold, 5),
            "graph_recall@10": recall_at(g_files, gold, 10),
        })

    rows_mh = []
    for q, seed_suffix, direction in MULTI_HOP:
        seeds = [nid for nid in g.nodes if nid.endswith(seed_suffix)]
        if not seeds:
            continue
        seed = seeds[0]
        if direction == "callees":
            nbrs = adj_calls.get(seed, [])
        else:
            nbrs = [s for s, d, t in g.edges if t == "CALLS" and d == seed]
        gold = {file_of(x) for x in nbrs} - {file_of(seed)}
        if not gold:
            continue
        flat = r.flat_retrieve(q, k=10)
        graph = r.graph_retrieve(q, k_seeds=5, out_k=10)
        f_files, g_files = files_of(flat), files_of(graph)
        f_ctx, f_tok = r.assemble_context(flat, budget_tokens=1500)
        g_ctx, g_tok = r.assemble_context(graph, budget_tokens=1500)
        f_cov = len([gl for gl in gold if gl in f_ctx]) / len(gold)
        g_cov = len([gl for gl in gold if gl in g_ctx]) / len(gold)
        rows_mh.append({
            "query": q, "seed": seed_suffix, "gold": sorted(gold),
            "flat_recall@10": recall_at(f_files, gold, 10),
            "graph_recall@10": recall_at(g_files, gold, 10),
            "flat_tokens": f_tok, "graph_tokens": g_tok,
            "flat_gold_cov": f_cov, "graph_gold_cov": g_cov,
        })

    avg = lambda rows, k: sum(x[k] for x in rows) / len(rows)
    summary = {
        "n_nodes": len(g.nodes),
        "n_edges": len(g.edges),
        "n_calls": len([e for e in g.edges if e[2] == "CALLS"]),
        "direct": {
            "flat_recall@5": avg(rows_direct, "flat_recall@5"),
            "flat_recall@10": avg(rows_direct, "flat_recall@10"),
            "graph_recall@5": avg(rows_direct, "graph_recall@5"),
            "graph_recall@10": avg(rows_direct, "graph_recall@10"),
        },
        "multihop": {
            "flat_recall@10": avg(rows_mh, "flat_recall@10"),
            "graph_recall@10": avg(rows_mh, "graph_recall@10"),
            "flat_gold_cov": avg(rows_mh, "flat_gold_cov"),
            "graph_gold_cov": avg(rows_mh, "graph_gold_cov"),
            "flat_tokens": avg(rows_mh, "flat_tokens"),
            "graph_tokens": avg(rows_mh, "graph_tokens"),
        },
    }
    out = {"summary": summary, "direct": rows_direct, "multihop": rows_mh}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dsum, msum = summary["direct"], summary["multihop"]

    x = [0, 1]
    axes[0].bar([i - 0.18 for i in x], [dsum["flat_recall@5"], dsum["flat_recall@10"]],
                0.36, label="Flat (embedding only)", color="#94A3B8")
    axes[0].bar([i + 0.18 for i in x], [dsum["graph_recall@5"], dsum["graph_recall@10"]],
                0.36, label="Graph-expanded (ours)", color="#2DD4BF")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Recall@5", "Recall@10"])
    axes[0].set_ylim(0, 1.05); axes[0].set_title("Direct lookup (n=5)")
    axes[0].set_ylabel("Avg recall"); axes[0].legend(fontsize=8)

    axes[1].bar([-0.18], [msum["flat_recall@10"]], 0.36, color="#94A3B8")
    axes[1].bar([0.18], [msum["graph_recall@10"]], 0.36, color="#2DD4BF")
    axes[1].set_xticks([-0.18, 0.18]); axes[1].set_xticklabels(["Flat", "Graph"])
    axes[1].set_ylim(0, 1.05); axes[1].set_title("Multi-hop dependency (n=3)\nRecall@10")

    axes[2].bar([-0.18], [msum["flat_gold_cov"]], 0.36, color="#94A3B8")
    axes[2].bar([0.18], [msum["graph_gold_cov"]], 0.36, color="#2DD4BF")
    axes[2].set_xticks([-0.18, 0.18]); axes[2].set_xticklabels(
        [f"Flat\n~{int(msum['flat_tokens'])} tok", f"Graph\n~{int(msum['graph_tokens'])} tok"])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Multi-hop gold coverage in\n1500-token context")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_retrieval_comparison.png"), dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

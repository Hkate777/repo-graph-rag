"""retriever.py — 可插拔嵌入检索 + 图扩展 + 预算内上下文组装。
默认嵌入器为 TF-IDF（零外部依赖）；官方 Gemma embeddings 到手后，
只需实现同名接口 embed()/embed_node() 即可无缝替换。
"""
import math
import re
from collections import defaultdict


def _tok(text):
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class TfidfEmbedder:
    """轻量 TF-IDF 向量（接口与 Gemma embedding 对齐：返回稀疏 dict）。"""

    def __init__(self):
        self.df = defaultdict(int)
        self.n_docs = 0
        self._node_tf = {}

    def fit(self, node_ids, docs):
        self.n_docs = len(docs)
        for nid, doc in zip(node_ids, docs):
            counts = defaultdict(int)
            for t in _tok(doc):
                counts[t] += 1
            self._node_tf[nid] = counts
            for t in counts:
                self.df[t] += 1

    def _vec(self, counts):
        vec = {}
        norm = 0.0
        for t, c in counts.items():
            idf = math.log((self.n_docs + 1) / (self.df.get(t, 0) + 1)) + 1.0
            w = (1 + math.log(c)) * idf
            vec[t] = w
            norm += w * w
        norm = math.sqrt(norm) or 1.0
        return {t: w / norm for t, w in vec.items()}

    def embed(self, query):
        counts = defaultdict(int)
        for t in _tok(query):
            counts[t] += 1
        return self._vec(counts)

    def embed_node(self, node_id):
        return self._vec(self._node_tf[node_id])

    def sim(self, qvec, node_id):
        nv = self._node_tf.get(node_id)
        if nv is None:
            return 0.0
        node_vec = self._vec(nv)
        # 稀疏点积
        if len(qvec) < len(node_vec):
            return sum(w * node_vec.get(t, 0.0) for t, w in qvec.items())
        return sum(w * qvec.get(t, 0.0) for t, w in node_vec.items())


class GraphRetriever:
    """语义检索（可插拔嵌入）+ 图扩展（调用者/被调者/导入）。"""

    def __init__(self, graph, embedder):
        self.g = graph
        self.emb = embedder
        self.adj = defaultdict(list)
        for s, d, t in graph.edges:
            self.adj[s].append((d, t))
            self.adj[d].append((s, t))

    def flat_retrieve(self, query, k=8):
        """baseline：纯嵌入相似度 top-k。"""
        qv = self.emb.embed(query)
        scored = [(nid, self.emb.sim(qv, nid))
                  for nid, n in self.g.nodes.items() if n.type in ("function", "method")]
        scored.sort(key=lambda x: -x[1])
        return [nid for nid, s in scored[:k] if s > 0]

    def graph_retrieve(self, query, k_seeds=5, pool=12, beta=0.35, out_k=10):
        """本方法：语义分数 + 图传播分数合并重排。
        - seeds：top-k 语义种子
        - graph score：种子经 CALLS/IMPORTS 边向邻居传播（多跳权重累加）
        - 候选 = top-pool 语义 ∪ 收到图分的邻居，按 sim + beta*gscore 重排
        """
        qv = self.emb.embed(query)
        scored = [(nid, self.emb.sim(qv, nid))
                  for nid, n in self.g.nodes.items() if n.type in ("function", "method")]
        scored.sort(key=lambda x: -x[1])
        seeds = scored[:k_seeds]
        seed_ids = {nid for nid, _ in seeds}

        gscore = defaultdict(float)
        for nid, s in seeds:
            for dst, etype in self.adj.get(nid, []):
                if dst in seed_ids:
                    continue
                w = 0.6 if etype == "CALLS" else 0.3
                gscore[dst] += s * w
        if gscore:
            mx = max(gscore.values())
            gscore = defaultdict(float, {k: v / mx for k, v in gscore.items()})

        cand_ids = {nid for nid, _ in scored[:pool]} | set(gscore)
        ranked = []
        for nid in cand_ids:
            n = self.g.nodes.get(nid)
            if n is None or n.type not in ("function", "method"):
                continue
            ranked.append((nid, self.emb.sim(qv, nid) + beta * gscore.get(nid, 0.0)))
        ranked.sort(key=lambda x: -x[1])
        return [nid for nid, _ in ranked[:out_k]]

    def assemble_context(self, node_ids, budget_tokens=1500):
        """按预算组装上下文；token 用 4 字符近似。"""
        chunks, used = [], 0
        for nid in node_ids:
            n = self.g.nodes.get(nid)
            if n is None or not n.code:
                continue
            header = f"# {n.type} {n.name} ({n.path}:{n.start}-{n.end})\n"
            est = (len(header) + len(n.code)) // 4
            if used + est > budget_tokens:
                break
            chunks.append(header + n.code)
            used += est
        return "\n\n".join(chunks), used


def file_of(node_id):
    return node_id.split(":", 1)[1].split("::", 1)[0]

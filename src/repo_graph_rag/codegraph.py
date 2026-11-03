"""codegraph.py — 用 Python ast 静态分析仓库，构建代码图谱。
节点: module / class / function / method；边: DEFINES / IMPORTS / CALLS。
该模块与数据来源解耦：官方 code-graphs 到手后，只需实现同构的 loader。
"""
import ast
import json
import os
import textwrap
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    type: str          # module | class | function | method
    name: str
    path: str          # 仓库相对路径
    start: int
    end: int
    code: str
    doc: str = ""


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)   # id -> Node
    edges: list = field(default_factory=list)   # (src, dst, type)

    def to_json(self, out_path):
        data = {
            "nodes": {k: vars(v) for k, v in self.nodes.items()},
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in self.edges],
        }
        with open(out_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def from_json(path):
        with open(path) as f:
            data = json.load(f)
        g = Graph()
        g.nodes = {k: Node(**v) for k, v in data["nodes"].items()}
        g.edges = [(e["src"], e["dst"], e["type"]) for e in data["edges"]]
        return g


def _read(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_graph(repo_dir: str, package_subdir: str = "") -> Graph:
    """扫描 repo_dir/package_subdir 下所有 .py，构建图谱。"""
    g = Graph()
    root = os.path.join(repo_dir, package_subdir) if package_subdir else repo_dir
    py_files = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                py_files.append(os.path.join(dirpath, fn))

    mod_ids = {}
    for fp in py_files:
        rel = os.path.relpath(fp, repo_dir)
        src = _read(fp)
        mod_id = f"module:{rel}"
        mod_ids[rel] = mod_id
        g.nodes[mod_id] = Node(mod_id, "module", os.path.basename(rel), rel, 1,
                               len(src.splitlines()), "", "")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for stmt in ast.walk(tree):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_class = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and any(
                        c is stmt for c in ast.walk(node)
                    ) and node is not stmt:
                        # 直接父类判断（简化：取包含它的最小 ClassDef）
                        if stmt.lineno > node.lineno and stmt.end_lineno <= node.end_lineno:
                            parent_class = node.name
                ntype = "method" if parent_class else "function"
                qname = f"{parent_class}.{stmt.name}" if parent_class else stmt.name
                nid = f"{ntype}:{rel}::{qname}"
                code = "\n".join(lines[stmt.lineno - 1:stmt.end_lineno])
                doc = ast.get_docstring(stmt) or ""
                g.nodes[nid] = Node(nid, ntype, stmt.name, rel,
                                    stmt.lineno, stmt.end_lineno, code, doc)
                g.edges.append((mod_id, nid, "DEFINES"))
                if parent_class:
                    cid = f"class:{rel}::{parent_class}"
                    if cid in g.nodes:
                        g.edges.append((cid, nid, "DEFINES"))
            elif isinstance(stmt, ast.ClassDef):
                cid = f"class:{rel}::{stmt.name}"
                code = "\n".join(lines[stmt.lineno - 1:stmt.end_lineno])
                doc = ast.get_docstring(stmt) or ""
                g.nodes[cid] = Node(cid, "class", stmt.name, rel,
                                    stmt.lineno, stmt.end_lineno, code, doc)
                g.edges.append((mod_id, cid, "DEFINES"))
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    target = alias.name
                    # 近似解析为仓库内模块
                    for rel, mid in mod_ids.items():
                        base = os.path.basename(rel)[:-3]
                        if target.split(".")[-1] == base:
                            g.edges.append((mod_id, mid, "IMPORTS"))
            elif isinstance(stmt, ast.Call):
                pass  # CALLS 在第二遍统一解析

    # 第二遍：解析 CALLS（函数体内调用 -> 按简单名匹配本仓库可调用对象）
    name_index = {}
    for nid, n in g.nodes.items():
        if n.type in ("function", "method"):
            name_index.setdefault(n.name, []).append(nid)

    HUB_CAP = 4  # 简单名匹配超过该数视为噪声词（如 get/decode/sign），跳过
    seen_edges = set()

    def _add(src, dst, etype):
        if src != dst and (src, dst, etype) not in seen_edges:
            seen_edges.add((src, dst, etype))
            g.edges.append((src, dst, etype))

    for nid, n in list(g.nodes.items()):
        if n.type not in ("function", "method") or not n.code:
            continue
        try:
            sub = ast.parse(textwrap.dedent(n.code))
        except (SyntaxError, ValueError):
            continue
        for c in ast.walk(sub):
            if isinstance(c, ast.Call):
                fname = None
                if isinstance(c.func, ast.Name):
                    fname = c.func.id
                elif isinstance(c.func, ast.Attribute):
                    fname = c.func.attr
                if fname and fname in name_index:
                    targets = name_index[fname]
                    if len(targets) > HUB_CAP:
                        continue
                    for dst in targets:
                        _add(nid, dst, "CALLS")
    # 去重 DEFINES/IMPORTS
    g.edges = [e for e in g.edges if e[2] == "CALLS" or (e[0], e[1], e[2]) not in seen_edges and not seen_edges.add((e[0], e[1], e[2]))]
    return g


if __name__ == "__main__":
    import sys
    repo = sys.argv[1]
    pkg = sys.argv[2] if len(sys.argv) > 2 else ""
    out = sys.argv[3] if len(sys.argv) > 3 else "graph.json"
    g = build_graph(repo, pkg)
    g.to_json(out)
    n_edges = len(set((s, d, t) for s, d, t in g.edges))
    print(f"nodes={len(g.nodes)} edges={n_edges} -> {out}")

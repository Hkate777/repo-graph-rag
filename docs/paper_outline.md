# Writeup 大纲（Kaggle 论文赛道必需板块）

> 目标 ≤3000 词。以下为骨架，内容由实验推进后填充。

## 1. Title & Subtitle
- 暂定：*repo-graph-rag: Structuring Repository Context for Offline SWE Agents via Code-Graphs and Gemma 4 Embeddings*
- 副标题：一个面向离线 Gemma 4 智能体的仓库级检索增强库与基准

## 2. Abstract（≤200 词）
- 问题：小模型在复杂仓库中导航困难，朴素文件拼接上下文噪声大。
- 方法：用官方 code-graphs + Gemma 4 embeddings 构建结构化检索，为 agent 提供精准子图/符号上下文。
- 结果：在自建/抽取的小 benchmark 上，带图检索 vs 不带，修复准确率 +X%，上下文 token 减少 Y%。
- 贡献：开源库 + 评估协议 + 新用例。

## 3. Introduction
- 背景：云端大模型 coding agent 昂贵；离线小模型（Gemma 4）普惠价值。
- 痛点：大代码库导航、多跳依赖、上下文效率。
- 本文贡献（3 点，对应 Resource + Application）。

## 4. Research Description
### 4.1 Methods
- code-graph 构建/加载（官方格式，待填）
- 嵌入检索（官方 embeddings + 查询编码）
- agent 接口（如何把子图/符号喂给 Gemma 4）
### 4.2 Experiments
- benchmark 定义（来源、规模、指标：fix 准确率 / 上下文效率）
- baseline：无检索 / 朴素文件检索
- 消融：仅图 / 仅嵌入 / 图+嵌入
- 结果表 + 图

## 5. Related Works & Citations
- SWE agent（SWE-bench, SWE-agent 等）
- code embedding / code search
- graph-based code understanding
- RAG for repo-level tasks
（文献清单由 Day 1–2 调研填充）

---
## 待填：官方数据 schema（由你从 Kaggle Data 标签提供）
- embeddings：字段 / 维度 / 文件格式（parquet? npy?）
- code-graphs：节点/边类型 / 格式（json? graphml?）
- 许可：Apache 2.0

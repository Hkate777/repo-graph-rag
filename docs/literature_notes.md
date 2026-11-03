# 文献笔记 & 新颖性定位（方向 A：code-graph + Gemma 4 embedding 的离线 SWE agent 检索增强库）

> 调研日期：2026-11-02。用于 Related Works 与贡献定位。

## 一、关键相关工作
| 工作 | 年份/会议 | 核心做法 | 与我们关系 |
|---|---|---|---|
| **CodexGraph** (Liu et al.) | NAACL 2025 | LLM agent + 代码图数据库接口（图查询语言），结构感知检索与导航 | 直接竞品思路；但它不针对离线 Gemma、也未用官方提供的 embeddings |
| **Code Graph Models / CGM** (CodeFuse) | NeurIPS 2025 | 把代码图结构融入 LLM 注意力（adapter），agentless graph RAG；Qwen2.5-72B 在 SWE-bench Lite 达 43%（开放权重 SOTA） | 需要改模型/训练，重；我们走轻量不改模型路线 |
| **G-MACS** | ICIC 2026 | 图引导多智能体做项目级代码摘要，CDG 减噪 | 图+多智能体范式参考 |
| **LLM Agents Can See Code Repositories** | 2026-06 arXiv | 视觉仓库表示；文本+视觉混合使输入 token 减少最多 26% | "上下文效率"视角可借鉴 |
| **RepoMind / Prodinit context graphs** | 工程实践 | AST 分块 + 符号图(Neo4j) + 向量(pgvector) 双索引，语义入口+图遍历扩展 | 工程实现参考，验证"图+嵌入"组合有效 |
| **RAG Over Codebases** (CloudSoftSol) | 实践综述 | 符号索引+嵌入+代码图三层；agentic search vs pre-indexed | 检索架构参考 |

## 二、基准与现状（Related Works 用）
- **SWE-bench**：真实 GitHub issue 修复，跑现有测试套件判分。Verified（500 题）为可信标准；Lite 为常用子集。
- 2026 年 SOTA（云端大模型 agent）SWE-bench Verified 已破 80%（Claude/GPT/Gemini 3.1 Pro ~80%，开源 setup 55%+）。
- 关键共识：**scaffold/检索比 base model 更影响结果**；单一"chunk-and-embed"在代码上失效，需结构（图）补充。

## 三、Gemma 4 关键事实（用于方法与可行性）
- 尺寸：E2B(2B)、E4B(4B，带原生音频)、26B MoE(仅 3.8B 激活)、31B Dense。
- 许可：**Apache 2.0**（可商用/微调）。
- 上下文：边缘模型 128K，大模型 256K。
- 代码能力：31B 在 LiveCodeBench v6 达 80%；原生 function calling / 结构化 JSON / 系统指令 → 适合做 agent。
- 部署：可在 Kaggle / Ollama / 消费级 GPU 离线跑；量化后 E2B/E4B 可在手机/Raspberry Pi 离线近零延迟。
- 竞赛主题契合点：**消费级硬件离线 agent**——这是我们的核心卖点。

## 四、我们的新颖性定位（Novelty / Relevance）
现有工作要么依赖云端大模型，要么需要改模型训练。我们的差异点：
1. **官方资源桥接**：首个把竞赛官方提供的 code-graphs + Gemma 4 embeddings 直接封装成可复用检索库 + 基准（很多参赛者可能未充分利用这些资源）。
2. **离线优先约束**：面向单加速卡/消费级硬件，不依赖 API，契合竞赛"every developer, offline"主旨。
3. **可量化收益**：以"上下文效率（token 减少%）+ 修复准确率"双指标评估，呼应消费级硬件瓶颈。
→ 同时命中 **Best New Resource**（库/工具）+ **Best New Application**（离线 agent 新用例）。

## 五、待办
- [ ] 用官方数据 schema 校准接口（待用户提供 Data 标签信息）
- [ ] 设计轻量 benchmark（建议：retrieval recall@k + 上下文缩减% + 小规模 agentic patch 研究，避免跑全量 SWE-bench 的沉重基建）
- [ ] 在公开仓库先跑通管线原型（不依赖官方数据），再接入官方 embeddings/code-graphs

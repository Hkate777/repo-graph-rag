"""生成 Kaggle Writeup 卡片缩略图 (560x280)。标题定稿后可改 TITLE/SUB 常量重新生成。"""
from PIL import Image, ImageDraw, ImageFont

W, H = 560, 280
BG = "#0B1220"        # 深夜蓝
ACCENT = "#2DD4BF"    # 青
ACCENT2 = "#60A5FA"   # 蓝
TEXT = "#F8FAFC"
SUB = "#94A3B8"

TITLE = "repo-graph-rag"
SUB_LINES = ["Graph-grounded retrieval for", "offline Gemma 4 coding agents"]
TAG = "KAGGLE PAPER TRACK"

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# 背景网格点
for x in range(0, W, 28):
    for y in range(0, H, 28):
        d.ellipse([x, y, x + 1, y + 1], fill="#1E293B")

# 右侧代码图 motif：节点 + 边
import math
nodes = [(430, 70), (500, 110), (455, 165), (395, 135), (505, 200), (430, 225)]
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 5), (4, 5), (3, 2)]
for a, b in edges:
    d.line([nodes[a], nodes[b]], fill="#334155", width=2)
for i, (x, y) in enumerate(nodes):
    r = 7 if i % 2 == 0 else 5
    color = ACCENT if i % 2 == 0 else ACCENT2
    d.ellipse([x - r, y - r, x + r, y + r], fill=color)

f_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
f_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
f_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)

# 顶部标签
d.rounded_rectangle([36, 32, 36 + d.textlength(TAG, font=f_tag) + 20, 56], radius=12,
                    fill="#111C2E", outline="#24344D", width=1)
d.text((46, 37), TAG, font=f_tag, fill=ACCENT)

# 主标题
d.text((36, 76), TITLE, font=f_big, fill=TEXT)
tw = d.textlength(TITLE, font=f_big)
d.rectangle([36, 130, 36 + tw, 134], fill=ACCENT)

# 副标题
y = 152
for line in SUB_LINES:
    d.text((36, y), line, font=f_sub, fill=SUB)
    y += 28

img.save("/workspace/repo-graph-rag/figures/thumbnail_v1.png")
print("saved", img.size)

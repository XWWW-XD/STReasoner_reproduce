import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 原始数据
# =========================
segments = ["0-999", "1000-1999", "2000-3999", "4000-5499", "5500-5999", "6000-6139", "6144+"]
entity = np.array([1154, 22, 2, 15, 0, 1, 0])
etiological = np.array([203, 4, 0, 0, 0, 0, 0])
correlation = np.array([1231, 297, 12, 28, 19, 5, 0])
forecasting = np.array([177, 100, 1, 0, 0, 2, 0])

totals = entity + etiological + correlation + forecasting
percentages = [84.48, 12.92, 0.46, 1.31, 0.58, 0.24, 0.00]

# =========================
# 2. 作图
# =========================
x = np.arange(len(segments))
width = 0.78

plt.figure(figsize=(14, 8))
ax = plt.gca()

# 黑白 + 纹理区分
ax.bar(
    x, entity, width,
    label="entity",
    color="white", edgecolor="black", hatch="///", linewidth=1.2
)

ax.bar(
    x, etiological, width, bottom=entity,
    label="etiological",
    color="0.85", edgecolor="black", hatch="\\\\\\", linewidth=1.2
)

ax.bar(
    x, correlation, width, bottom=entity + etiological,
    label="correlation",
    color="0.65", edgecolor="black", hatch="xxx", linewidth=1.2
)

ax.bar(
    x, forecasting, width, bottom=entity + etiological + correlation,
    label="forecasting",
    color="0.35", edgecolor="black", hatch="...", linewidth=1.2
)


# =========================
# 3. 柱内数字
# =========================
def add_labels(values, bottoms):
    for i, v in enumerate(values):
        if v > 0:
            y = bottoms[i] + v / 2
            ax.text(
                x[i], y, str(int(v)),
                ha="center", va="center",
                fontsize=11
            )


add_labels(entity, np.zeros_like(entity))
add_labels(etiological, entity)
add_labels(correlation, entity + etiological)
add_labels(forecasting, entity + etiological + correlation)

# =========================
# 4. 每组总数 + 占比
# =========================
for i, (total, pct) in enumerate(zip(totals, percentages)):
    ax.text(
        x[i], total + 60,
        f"{int(total)} ({pct:.2f}%)",
        ha="center", va="bottom",
        fontsize=12
    )

# =========================
# 5. 坐标轴和标题
# =========================
ax.set_title("ST-TEST response token length distribution", fontsize=20, pad=18)
ax.set_xlabel("response token range", fontsize=14)
ax.set_ylabel("sample count", fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(segments, fontsize=12)

ax.set_ylim(0, 3000)
ax.tick_params(axis="y", labelsize=12)

# 网格线
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# 图例
ax.legend(
    loc="upper right",
    fontsize=12,
    frameon=True
)

# 去掉上右边框，更清爽
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# 保存
out_path = "outputs/st_test_token_distribution_bw.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"saved: {out_path}")

# 推荐系统语义对比学习 README

本 README 总结了两种在推荐系统中使用的语义对比学习损失，用于提升模型在语义相近 item 上的预测一致性能力。

---

# 🌟 方案一：仅正样本约束（Positive-Only Semantic Consistency）

## 核心思想
仅对语义相近的正样本进行约束，不对负样本施加限制。  
要求正样本与原样本在预测方向上一致：

- 若 y=1（喜欢/点击），则 **score_pos > score_raw**
- 若 y=0（不喜欢/未点击），则 **score_pos < score_raw**

原样本标签仅提供方向，不提供绝对值约束。

## 关键公式
定义：

```
sign = +1  (y = 1)
sign = -1  (y = 0)
```

语义一致性损失：

```
L_pos = softplus( - sign * (score_pos - score_raw) )
```

最终总损失：

```
L_total = BCE(score_raw, y) + λ * L_pos
```

## 核心代码（可直接使用）
```python
import torch
import torch.nn.functional as F

def loss_positive_only(model, u, item_raw, item_pos, y_raw, lambda_sem=1.0):
    score_raw = model(u, item_raw)    # (B,)
    score_pos = model(u, item_pos)    # (B,)

    loss_ctr = F.binary_cross_entropy_with_logits(score_raw, y_raw.float())

    sign = (y_raw * 2 - 1)

    loss_pos = F.softplus(- sign * (score_pos - score_raw)).mean()

    loss = loss_ctr + lambda_sem * loss_pos
    return loss, loss_ctr, loss_pos
```

---

# 🌟 方案二：正负样本排序对比（Label-Conditioned Pairwise Ranking）

## 核心思想
除了正样本，还对负样本施加强排序约束：

- 若 y=1 → **score_pos > score_neg**
- 若 y=0 → **score_pos < score_neg**

适合负样本采样质量较高的场景。

## 关键公式
正负对比损失：

```
L_pair = softplus( - sign * (score_pos - score_neg) )
```

多个负样本取平均：

```
L_pair = mean_k softplus( - sign * (score_pos - score_neg[k]) )
```

最终总损失：

```
L_total = BCE(score_raw, y) + λ * L_pair
```

## 核心代码（支持多负样本）
```python
import torch
import torch.nn.functional as F

def loss_pos_neg(model, u, item_raw, item_pos, item_neg, y_raw, lambda_rank=1.0):
    B, K = item_neg.shape[:2]

    score_raw = model(u, item_raw)
    score_pos = model(u, item_pos)

    u_neg = u.unsqueeze(1).repeat(1, K, 1)
    score_neg = model(
        u_neg.reshape(B*K, -1),
        item_neg.reshape(B*K, -1)
    ).reshape(B, K)

    loss_ctr = F.binary_cross_entropy_with_logits(score_raw, y_raw.float())

    sign = (y_raw * 2 - 1).unsqueeze(1)

    diff = score_pos.unsqueeze(1) - score_neg
    loss_pair = F.softplus(- sign * diff).mean()

    loss = loss_ctr + lambda_rank * loss_pair
    return loss, loss_ctr, loss_pair
```

---

# 🧭 两方案差异总结

| 方案 | 正样本 | 负样本 | 强度 | 特点 |
|------|--------|---------|------|------|
| **方案一 only-pos** | ✔ 强约束 | ✖ 不约束 | ✓ 稳定 | 最安全、推荐先试 |
| **方案二 pos-neg** | ✔ 强约束 | ✔ 强约束 | ★ 更强 | 负样本质量高时效果更好 |

---

# 🏁 实验建议
- **先试方案一**：不会破坏校准、风险最低  
- 若需更强对比信号 → 再试方案二  

---

如需更完整的技术文档、示意图或英文 README 可继续告诉我。

# Speech Endpointing 指标计算（延迟率 / 打断率）

本文档定义在 `treat_unaddressed_as_eou=true` 策略下，Speech Endpointing 的两项核心 KPI：

- **打断率**：本该继续（CONT）却被判成结束（EOU）的比例（误打断）
- **延迟率**：本该结束（EOU）却被判成继续（CONT）的比例（延迟接话/漏判结束）

> 说明：本文只讨论二分类 KPI（EOU vs CONT）。服务虽然会输出三类 `<EOU>/<CONT_USER>/<UNADDRESSED>` 的概率，但 KPI 统计按业务策略折叠为二类。

---

## 1) 模型输出（3 个 special token 概率）

外层服务从模型拿到 3 个 special token 的 logit（或 logprobs），并在这 3 个 token 上做 softmax 归一化得到：

- `p_eou = P(<EOU>)`
- `p_cont = P(<CONT_USER>)`
- `p_unaddr = P(<UNADDRESSED>)`

且满足：

`p_eou + p_cont + p_unaddr = 1`

---

## 2) `treat_unaddressed_as_eou=true` 的合并规则

当开启 `treat_unaddressed_as_eou=true` 时，把 `UNADDRESSED` 的概率并入 `EOU`：

- `p_eou* = p_eou + p_unaddr`
- `p_cont* = p_cont`
- `p_unaddr* = 0`

此时二分类只看 `p_eou*` 与 `p_cont*`，并且：

`p_eou* + p_cont* = 1`

---

## 3) 阈值判决（`eou_threshold`）

给定阈值 `τ = eou_threshold`（例如 `0.6`）：

- 预测 `ŷ = EOU` 当且仅当 `p_eou* >= τ`
- 否则预测 `ŷ = CONT`

---

## 4) 真值折叠（用于二分类 KPI）

如果标注提供三类标签 `y ∈ {EOU, CONT_USER, UNADDRESSED}`，在该策略下折叠为二类真值 `y*`：

- `y* = EOU` 若 `y ∈ {EOU, UNADDRESSED}`
- `y* = CONT` 若 `y = CONT_USER`

---

## 5) 混淆矩阵

按二分类（`EOU` vs `CONT`）统计：

- `TP = # {y* = EOU  且  ŷ = EOU}`
- `FN = # {y* = EOU  且  ŷ = CONT}`
- `FP = # {y* = CONT 且  ŷ = EOU}`
- `TN = # {y* = CONT 且  ŷ = CONT}`

---

## 6) KPI 定义

### 6.1 打断率（Interrupt Rate）

“本该继续却被判成结束”的比例（误打断）：

`打断率 = FP / (FP + TN) = P(ŷ = EOU | y* = CONT)`

### 6.2 延迟率（Latency Rate）

“本该结束却被判成继续”的比例（延迟接话/漏判结束）：

`延迟率 = FN / (TP + FN) = P(ŷ = CONT | y* = EOU)`

---

## 7) 伪代码（逐条累积）

```python
# inputs: p_eou, p_cont, p_unaddr, y (EOU/CONT_USER/UNADDRESSED), tau

p_eou_star = p_eou + p_unaddr
y_star = "EOU" if y in ["EOU", "UNADDRESSED"] else "CONT"

y_hat = "EOU" if p_eou_star >= tau else "CONT"

if y_star == "EOU" and y_hat == "EOU":
    TP += 1
elif y_star == "EOU" and y_hat == "CONT":
    FN += 1
elif y_star == "CONT" and y_hat == "EOU":
    FP += 1
else:
    TN += 1

interrupt_rate = FP / (FP + TN) if (FP + TN) else 0.0
latency_rate   = FN / (TP + FN) if (TP + FN) else 0.0
```


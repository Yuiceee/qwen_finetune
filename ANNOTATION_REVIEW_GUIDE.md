# 使用DeepSeek API审核标注指南

## 📋 背景

V2训练后的评估结果显示：
- **平均字段准确率**: 55.96%
- **participants准确率**: 42.3% ❌ (最弱)
- **time准确率**: 51.1% ❌ (次弱)
- **organizer准确率**: 53.3% ❌

**问题分析：**
1. participants字段经常混淆参与者和组织者
2. time字段仍有格式不一致问题
3. 这些低质量标注严重影响模型学习

**解决方案：** 使用DeepSeek API自动审核和修正标注

---

## 🚀 快速开始

### 步骤1: 设置API密钥

```bash
export DEEPSEEK_API_KEY='your-deepseek-api-key'
```

### 步骤2: 测试审核（推荐先运行）

```bash
# 只处理5个样本，快速验证
bash run_fix_annotations_test.sh
```

**预期输出：**
- `data/reviewed_test/train_sample.jsonl` - 审核后的5个样本
- `data/reviewed_test/train_sample_review_report.json` - 审核报告

**查看结果：**
```bash
# 查看统计
cat data/reviewed_test/train_sample_review_report.json | jq '.statistics'

# 查看改进示例
cat data/reviewed_test/train_sample_review_report.json | jq '.review_results[] | select(.status=="improved")'
```

### 步骤3: 完整审核（测试成功后运行）

```bash
# 审核所有数据集
bash run_fix_annotations.sh
```

**处理流程：**
1. 训练集：聚焦模式（只审核字段准确率<80%的样本）
2. 验证集：完整模式（审核所有138个样本）
3. 测试集：完整模式（审核所有137个样本）

**预期时间：**
- 训练集：约50-80个错误样本 × 1秒 = 1-2分钟
- 验证集：138个样本 × 0.5秒 = 1分钟
- 测试集：137个样本 × 0.5秒 = 1分钟
- **总计：约3-5分钟**

---

## 📊 审核报告解读

审核完成后会生成三个报告文件：

```bash
data/reviewed/train_review_report.json
data/reviewed/valid_review_report.json
data/reviewed/test_review_report.json
```

### 统计信息

```json
{
  "statistics": {
    "total": 80,          // 审核的样本数
    "improved": 45,       // 改进的样本数
    "unchanged": 30,      // 无需改动的样本数
    "failed": 5,          // 审核失败的样本数
    "field_changes": {
      "participants": 25, // participants字段改动次数
      "time": 20,         // time字段改动次数
      "organizer": 10,
      "title": 5,
      "location": 3,
      "event_type": 2
    }
  }
}
```

### 改进示例

报告中会显示前5个改进示例，格式如下：

```
1. 样本#42 的改动:
   【participants】
     原始: ['Kevin A. Howard']
     改进: ['Dan Boyle']
   【time】
     原始: November 7
     改进: 2001-11-07
```

---

## 🛠️ 高级用法

### 只审核特定文件

```bash
uv run python scripts/fix_annotations_with_llm.py \
    --input data/standardized/train.jsonl \
    --output data/reviewed/train.jsonl \
    --api_key "$DEEPSEEK_API_KEY"
```

### 聚焦模式（只审核错误样本）

```bash
uv run python scripts/fix_annotations_with_llm.py \
    --input data/standardized/train.jsonl \
    --output data/reviewed/train.jsonl \
    --api_key "$DEEPSEEK_API_KEY" \
    --focus_on_errors \
    --error_analysis_file outputs/evaluation_results_v2.json
```

**聚焦模式优势：**
- 只处理字段准确率<80%的样本
- 节省API调用成本（约减少50-60%）
- 加快处理速度

### 调整API调用延迟

```bash
# 更快（可能触发限流）
--delay 0.2

# 默认（推荐）
--delay 0.5

# 更保守（避免限流）
--delay 1.0
```

### 处理更多样本（测试用）

```bash
--max_samples 10  # 只处理前10个样本
```

---

## 📈 预期改进效果

基于LLM审核后的数据重新训练，预期效果：

| 字段 | V2 (标准化数据) | V3 (LLM审核后) | 改进 |
|------|----------------|----------------|------|
| participants | 42.3% | 65%+ | +20% |
| time | 51.1% | 70%+ | +18% |
| organizer | 53.3% | 68%+ | +15% |
| **平均准确率** | **55.96%** | **70%+** | **+14%** |

---

## 🔍 下一步：使用审核后的数据训练

### 修改 `run_step1.sh`

将数据路径从 `data/standardized/` 改为 `data/reviewed/`：

```bash
uv run python scripts/train_lora.py \
    --train_data data/reviewed/train.jsonl \      # 修改这里
    --eval_data data/reviewed/valid.jsonl \       # 修改这里
    --output_dir outputs/lora_model_v3 \          # V3模型
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --early_stopping
```

### 运行V3训练

```bash
bash run_step1.sh
```

### 评估V3模型

```bash
uv run python scripts/evaluate_models.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model outputs/lora_model_v3/final_model \
    --test_file data/reviewed/test.jsonl \        # 使用审核后的测试集
    --output_file outputs/evaluation_results_v3.json

# 查看改进效果
uv run python scripts/analyze_errors_simple.py --eval_file outputs/evaluation_results_v3.json
```

---

## ⚠️ 注意事项

1. **API成本估算**:
   - DeepSeek价格：约¥0.001/1K tokens
   - 每个样本约1000 tokens（输入+输出）
   - 聚焦模式：约80个样本 × ¥0.001 = ¥0.08
   - 完整模式：1367个样本 × ¥0.001 = ¥1.37
   - **总成本：约¥1-2元**

2. **API限流**:
   - 如果遇到限流错误，增加 `--delay` 参数
   - DeepSeek默认限制：约60 RPM（每分钟请求数）

3. **数据备份**:
   - 原始数据仍保留在 `data/standardized/`
   - 审核后的数据保存在 `data/reviewed/`
   - 可以随时对比两个版本

4. **失败样本处理**:
   - 如果某些样本审核失败（JSON解析错误等），会保留原标注
   - 查看报告中的 `failed` 统计

---

## 🎯 成功标准

审核完成后，检查以下指标：

✅ **改进率 > 50%**（至少一半样本得到改进）
✅ **participants字段改动 > 20次**
✅ **time字段改动 > 15次**
✅ **无大量失败样本**（failed < 5%）

如果满足以上标准，可以放心使用审核后的数据进行V3训练！

---

## 🆘 故障排除

### 问题1: API密钥错误

```
⚠️  请设置环境变量 DEEPSEEK_API_KEY
```

**解决：**
```bash
export DEEPSEEK_API_KEY='sk-xxxxxxxxx'
```

### 问题2: 评估结果文件不存在

```
⚠️  未找到评估结果文件: outputs/evaluation_results_v2.json
```

**解决：**
```bash
# 先运行评估
bash run_step1.sh
```

### 问题3: API限流

```
Error: Rate limit exceeded
```

**解决：**
```bash
# 增加延迟
uv run python scripts/fix_annotations_with_llm.py ... --delay 2.0
```

### 问题4: JSON解析失败

查看报告中的 `failed` 样本，检查：
- API返回格式是否正确
- 是否包含markdown代码块（已处理）
- 是否有特殊字符导致JSON无效

---

**祝审核顺利！期待V3训练的好结果！🚀**

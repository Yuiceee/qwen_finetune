# Scripts 目录说明

本目录包含邮件事件提取任务的所有训练和评估脚本。

## 📁 目录结构

```
scripts/
├── README.md                        # 本文件
├── 01_process_enron_csv.py         # 数据预处理：从CSV提取邮件
├── 02_generate_annotations.py      # 使用LLM生成标注数据
├── 03_split_dataset.py             # 拆分训练/验证/测试集
├── clean_and_standardize_data.py   # 数据清理：统一schema、修复JSON
├── standardize_time_fields_v2.py   # 时间字段标准化
├── fix_annotations_with_llm.py     # LLM辅助审核和修正标注 ⭐ NEW
├── check_data_quality.py           # 数据质量检查工具
├── analyze_errors.py               # 详细错误分析
├── analyze_errors_simple.py        # 简化版错误分析
├── train_lora.py                   # LoRA微调训练脚本
├── evaluate_models.py              # 模型评估脚本
├── inference.py                    # 命令行推理脚本
└── inference.ipynb                 # Jupyter推理notebook
```

## 🔄 数据处理流程

### 阶段1: 原始数据准备
```bash
# 1. 从CSV提取邮件（采样）
uv run python scripts/01_process_enron_csv.py \
    --input data/emails.csv \
    --output data/raw/enron_sampled.jsonl \
    --sample_size 2000

# 2. 使用LLM生成标注
uv run python scripts/02_generate_annotations.py \
    --api_key "YOUR_API_KEY" \
    --base_url "API_BASE_URL" \
    --model "MODEL_NAME" \
    --template data/raw/annotation_template.json \
    --output data/raw/train_data_all.jsonl

# 3. 拆分数据集
uv run python scripts/03_split_dataset.py \
    --input data/raw/train_data_all.jsonl \
    --output_dir data/processed
```

**输出：** `data/processed/{train,valid,test}.jsonl` (1860样本)

---

### 阶段2: 数据清理 ✅ (已完成)
```bash
# 4. 清理JSON格式错误，统一schema
uv run python scripts/clean_and_standardize_data.py
```

**效果：**
- ❌ 移除163个JSON格式错误的样本
- ✅ 统一为核心6字段：event_type, title, time, location, participants, organizer
- ✅ 移除80+个冗余字段

**输出：** `data/cleaned/{train,valid,test}.jsonl` (1697样本)

---

### 阶段3: 时间字段标准化 ✅ (已完成)
```bash
# 5. 标准化时间字段格式
uv run python scripts/standardize_time_fields_v2.py --execute --min_confidence=medium
```

**效果：**
- ✅ 100%邮件成功提取年份
- ✅ 80%的time字段标准化为 `YYYY-MM-DD` 或 `YYYY-MM-DD HH:MM`
- ✅ 修复了年份错误问题（"Nov. 7" → "2001-11-07" 而非 "2026-11-07"）

**输出：** `data/standardized/{train,valid,test}.jsonl` (1367样本)

---

### 阶段4: LLM辅助审核标注 ⭐ (新增)

**背景：** V2训练后发现participants (42.3%)和time (51.1%)字段准确率仍然较低，需要使用LLM审核和改进标注质量。

```bash
# 6a. 测试审核（处理5个样本）
export DEEPSEEK_API_KEY='your-api-key'
bash run_fix_annotations_test.sh

# 6b. 完整审核（处理所有样本）
export DEEPSEEK_API_KEY='your-api-key'
bash run_fix_annotations.sh
```

**或手动运行：**

```bash

# 审核所有样本
uv run python scripts/fix_annotations_with_llm.py \
    --input data/standardized/valid.jsonl \
    --output data/reviewed/valid.jsonl \
    --api_key "$DEEPSEEK_API_KEY"
```

**效果：**
- 🎯 重点改进participants和time字段
- 📊 生成详细的审核报告，包含改进前后对比
- 💰 成本可控（聚焦模式只处理错误样本）

**输出：** `data/reviewed/{train,valid,test}.jsonl` + 审核报告

---

## 🚀 训练模型

### 使用清理后的数据训练LoRA

```bash
# 基础训练（推荐配置）
uv run python scripts/train_lora.py

# 自定义参数
uv run python scripts/train_lora.py \
    --train_data data/standardized/train.jsonl \
    --eval_data data/standardized/valid.jsonl \
    --output_dir outputs/lora_model_v2 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --early_stopping \
    --early_stopping_patience 5
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_data` | `data/standardized/train.jsonl` | 训练数据（1092样本）|
| `--eval_data` | `data/standardized/valid.jsonl` | 验证数据（138样本）|
| `--output_dir` | `outputs/lora_model_v2` | 输出目录 |
| `--num_epochs` | `5` | 训练轮数 |
| `--learning_rate` | `1e-4` | 学习率（降低以提高稳定性）|
| `--warmup_ratio` | `0.1` | 前10%步数用于warmup |
| `--early_stopping` | `True` | 启用早停（默认开启）|
| `--early_stopping_patience` | `5` | 5个eval步骤无改进则停止 |

### ⚠️ V1 vs V2 训练差异

**V1训练（旧数据）：**
- 数据：`data/processed/` (1488样本)
- 问题：163个JSON错误，80+冗余字段，time格式混乱
- 结果：平均字段准确率 50%，time准确率 33%

**V2训练（新数据，推荐）：**
- 数据：`data/standardized/` (1367样本)
- 改进：高质量数据，统一schema，time标准化
- 预期：平均字段准确率 70%+，time准确率 65%+

---

## 📊 评估和推理

### 评估模型
```bash
# 对比基础模型和微调模型
uv run python scripts/evaluate_models.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model outputs/lora_model_v2/final_model \
    --test_file data/standardized/test.jsonl \
    --output outputs/evaluation_v2.json
```

### 命令行推理
```bash
# 单样本推理
uv run python scripts/inference.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model outputs/lora_model_v2/final_model \
    --interactive

# 批量推理
uv run python scripts/inference.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model outputs/lora_model_v2/final_model \
    --test_file data/standardized/test.jsonl \
    --max_samples 10
```

### Jupyter Notebook推理
```bash
jupyter notebook scripts/inference.ipynb
```

---

## 🛠️ 工具脚本

### 数据质量检查
```bash
# 检查数据集的质量问题
uv run python scripts/check_data_quality.py
```

输出报告：
- JSON格式错误统计
- 字段一致性分析
- time字段格式分布
- 输入长度分布

---

## 📈 预期改进

使用清理后的数据重新训练，预期效果：

| 指标 | V1 (旧数据) | V2 (新数据) | 改进 |
|------|-------------|-------------|------|
| JSON格式正确率 | 95.7% | 98%+ | +2.3% |
| 平均字段准确率 | 49.6% | 70%+ | +20% |
| time字段准确率 | 33.3% | 65%+ | +32% |
| event_type准确率 | 64.5% | 75%+ | +10% |

---

## 🎯 下一步

1. ✅ **数据已清理** - data/standardized/ 目录 (1367样本)
2. ✅ **V2训练完成** - 平均准确率55.96%，发现participants和time字段较弱
3. ⏳ **LLM审核标注** - 使用DeepSeek API改进标注质量
   ```bash
   export DEEPSEEK_API_KEY='your-key'
   bash run_fix_annotations_test.sh  # 先测试5个样本
   bash run_fix_annotations.sh       # 完整审核
   ```
4. ⏳ **V3训练** - 使用审核后的数据重新训练
5. ❓ **考虑DPO** - 如果准确率达到70%+且主要问题是格式规范性

---

## 📝 注意事项

- 所有脚本使用 `uv run` 执行，确保依赖隔离
- 训练需要GPU（推荐A100或以上）
- SwanLab用于实验跟踪，会自动记录所有指标
- 使用early stopping避免过拟合
- 旧版本脚本已删除（standardize_time_fields.py, comparison_analysis.py）

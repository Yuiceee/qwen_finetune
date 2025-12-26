#!/bin/bash

# LoRA微调完整流程脚本

set -e  # 遇到错误立即退出

# 设置HuggingFace缓存目录（使用已下载的模型）
export HF_HOME=/macroverse/public/database/huggingface/hub
export HF_DATASETS_CACHE=/macroverse/public/database/huggingface/datasets

echo "======================================================"
echo "第一步：LoRA微调完整流程"
echo "======================================================"
echo ""
echo "环境变量设置："
echo "  HF_HOME=$HF_HOME"
echo ""

# 检查数据是否存在
echo ""
echo "步骤 1/3: 检查训练数据"
echo "------------------------------------------------------"
if [ ! -f "data/processed/train.jsonl" ] || [ ! -f "data/processed/test.jsonl" ]; then
    echo "⚠️  训练数据不存在，请先运行数据预处理脚本："
    echo "  1. uv run python scripts/01_process_enron_csv.py"
    echo "  2. uv run python scripts/02_generate_annotations.py"
    echo "  3. uv run python scripts/03_split_dataset.py"
    exit 1
else
    echo "✓ 训练数据存在"
    echo "  - train.jsonl: $(wc -l < data/processed/train.jsonl) 条"
    echo "  - test.jsonl: $(wc -l < data/processed/test.jsonl) 条"
fi

# 安装trackio依赖
echo ""
echo "步骤 1.5/3: 安装依赖"
echo "------------------------------------------------------"
uv add trackio

# 训练模型
echo ""
echo "步骤 2/3: 开始LoRA微调训练"
echo "------------------------------------------------------"
uv run python scripts/train_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_data data/processed/train.jsonl \
    --eval_data data/processed/test.jsonl \
    --output_dir outputs/lora_model \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32

# 测试推理
echo ""
echo "步骤 3/3: 测试推理效果"
echo "------------------------------------------------------"
uv run python scripts/inference.py \
    --lora_model outputs/lora_model/final_model \
    --test_file data/processed/test.jsonl \
    --max_samples 5

echo ""
echo "======================================================"
echo "LoRA微调流程完成！"
echo "======================================================"
echo ""
echo "模型保存位置: outputs/lora_model/final_model"
echo "训练日志: outputs/lora_model/logs"
echo ""
echo "下一步可以："
echo "  1. 使用 trackio show --project email-lora-finetuning 查看训练曲线"
echo "  2. 运行交互式测试: uv run python scripts/inference.py --lora_model outputs/lora_model/final_model --interactive"
echo "  3. 查看完整测试结果: uv run python scripts/inference.py --lora_model outputs/lora_model/final_model --test_file data/processed/test.jsonl"
echo "  4. 进入第二步：DPO对齐优化"
echo ""

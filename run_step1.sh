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
echo "步骤 1/4: 检查训练数据"
echo "------------------------------------------------------"
if [ ! -f "data/processed/train.jsonl" ] || [ ! -f "data/processed/valid.jsonl" ] || [ ! -f "data/processed/test.jsonl" ]; then
    echo "⚠️  训练数据不存在，请先运行数据预处理脚本："
    # echo "  1. uv run python scripts/01_process_enron_csv.py"
    # echo "  2. uv run python scripts/02_generate_annotations.py"
    # echo "  3. uv run python scripts/03_split_dataset.py"
    exit 1
else
    echo "✓ 训练数据存在"
    echo "  - train.jsonl: $(wc -l < data/processed/train.jsonl) 条（训练集）"
    echo "  - valid.jsonl: $(wc -l < data/processed/valid.jsonl) 条（验证集，用于早停）"
    echo "  - test.jsonl: $(wc -l < data/processed/test.jsonl) 条（测试集，用于最终评估）"
fi


# 训练模型
echo ""
echo "步骤 2/4: 开始LoRA微调训练（启用早停）"
echo "------------------------------------------------------"
uv run python scripts/train_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_data data/processed/train.jsonl \
    --eval_data data/processed/valid.jsonl \
    --output_dir outputs/lora_model \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --early_stopping \
    --early_stopping_patience 5 \
    --metric_for_best_model eval_loss

# 评估模型对比
echo ""
echo "步骤 3/4: 评估模型效果（对比原始模型 vs 微调模型）"
echo "------------------------------------------------------"
uv run python scripts/evaluate_models.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model outputs/lora_model/final_model \
    --test_file data/processed/test.jsonl \
    --max_samples None  \
    --output_file outputs/evaluation_results.json

# # 测试推理
# echo ""
# echo "步骤 4/4: 测试推理效果"
# echo "------------------------------------------------------"
# uv run python scripts/inference.py \
#     --lora_model outputs/lora_model/final_model \
#     --test_file data/processed/test.jsonl \
#     --max_samples 5

echo ""
echo "======================================================"
echo "LoRA微调流程完成！"
echo "======================================================"
echo ""
echo "📊 评估结果摘要："
echo "------------------------------------------------------"
if [ -f "outputs/evaluation_results.json" ]; then
    echo "详细评估结果已保存到: outputs/evaluation_results.json"
    python3 -c "
import json
with open('outputs/evaluation_results.json', 'r') as f:
    data = json.load(f)
    base = data['base_metrics']
    ft = data['finetuned_metrics']
    imp = data['improvements']

    print(f\"\\n指标对比：\")
    print(f\"  JSON格式正确率: {base['json_format_accuracy']:.1f}% → {ft['json_format_accuracy']:.1f}% (提升 {imp['json_format_accuracy']:+.1f}%)\")
    print(f\"  字段完整性: {base['field_completeness']:.1f}% → {ft['field_completeness']:.1f}% (提升 {imp['field_completeness']:+.1f}%)\")
    print(f\"  字段准确性: {base['field_accuracy']:.1f}% → {ft['field_accuracy']:.1f}% (提升 {imp['field_accuracy']:+.1f}%)\")
    print(f\"  完全匹配率: {base['exact_match_rate']:.1f}% → {ft['exact_match_rate']:.1f}% (提升 {imp['exact_match_rate']:+.1f}%)\")
"
fi
# echo ""
# echo "📂 输出文件："
# echo "  - 模型: outputs/lora_model/final_model"
# echo "  - 训练日志: outputs/lora_model/logs"
# echo "  - 评估结果: outputs/evaluation_results.json"
# echo ""
# echo "🔍 下一步可以："
# echo "  1. 查看训练曲线: swanlab watch (或访问 SwanLab Web UI)"
# echo "  2. 查看详细评估: cat outputs/evaluation_results.json | jq"
# echo "  3. 交互式测试: uv run python scripts/inference.py --lora_model outputs/lora_model/final_model --interactive"
# echo "  4. 完整测试集: uv run python scripts/inference.py --lora_model outputs/lora_model/final_model --test_file data/processed/test.jsonl"
# echo ""

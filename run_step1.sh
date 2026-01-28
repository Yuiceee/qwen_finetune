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
echo "步骤 1/4: 检查训练数据（使用清理和标准化后的数据）"
echo "------------------------------------------------------"
if [ ! -f "data/reviewed/train.jsonl" ] || [ ! -f "data/reviewed/valid.jsonl" ] || [ ! -f "data/reviewed/test.jsonl" ]; then
    echo "⚠️  标准化数据不存在，请先运行数据清理脚本："
    echo "  1. uv run python scripts/clean_and_standardize_data.py"
    echo "  2. uv run python scripts/standardize_time_fields_v2.py --execute --min_confidence=medium"
    exit 1
else
    echo "✓ 标准化训练数据存在 (V2 - 清理后)"
    echo "  - train.jsonl: $(wc -l < data/reviewed/train.jsonl) 条（训练集）"
    echo "  - valid.jsonl: $(wc -l < data/reviewed/valid.jsonl) 条（验证集，用于早停）"
    echo "  - test.jsonl: $(wc -l < data/reviewed/test.jsonl) 条（测试集，用于最终评估）"
    echo ""
    echo "  数据质量改进："
    echo "    ✓ 统一为核心6字段 (event_type, title, time, location, participants, organizer)"
    echo "    ✓ 移除JSON格式错误"
    echo "    ✓ time字段标准化为 YYYY-MM-DD 或 YYYY-MM-DD HH:MM"
fi


# 训练模型
echo ""
echo "步骤 2/4: 开始LoRA微调训练 V2（使用优化参数）"
echo "------------------------------------------------------"
echo "训练配置："
echo "  - 数据: data/reviewed/ "
echo "  - 学习率: 1e-4 (降低以提高稳定性)"
echo "  - Warmup: 10% (更好的收敛)"
echo "  - Epochs: 5 (带early stopping)"
echo "  - 早停耐心: 5个评估步骤"
echo ""
uv run python scripts/train_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_data data/reviewed/train.jsonl \
    --eval_data data/reviewed/valid.jsonl \
    --output_dir outputs/lora_model_v2 \
    --num_epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --early_stopping \
    --early_stopping_patience 5 \
    --metric_for_best_model eval_loss

# 评估模型对比
echo ""
echo "步骤 3/4: 评估模型效果 V2（对比原始模型 vs 微调模型）"
echo "------------------------------------------------------"
uv run python scripts/evaluate_models.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_model outputs/lora_model_v2/final_model \
    --test_file data/reviewed/test.jsonl \
    --output_file outputs/evaluation_results_v2.json

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
echo "LoRA微调流程 V2 完成！"
echo "======================================================"
echo ""
echo "📊 评估结果摘要 (V2 - 使用清理后数据)："
echo "------------------------------------------------------"
if [ -f "outputs/evaluation_results_v2.json" ]; then
    echo "详细评估结果已保存到: outputs/evaluation_results_v2.json"
    python3 -c "
import json
with open('outputs/evaluation_results_v2.json', 'r') as f:
    data = json.load(f)
    base = data['base_metrics']
    ft = data['finetuned_metrics']
    imp = data['improvements']

    print(f\"\\n=== 核心指标对比 ===\")
    print(f\"  JSON格式正确率: {base['json_format_accuracy']:.1f}% → {ft['json_format_accuracy']:.1f}% (提升 {imp['json_format_accuracy']:+.1f}%)\")
    print(f\"  平均字段准确率: {base['average_field_accuracy']:.1f}% → {ft['average_field_accuracy']:.1f}% (提升 {imp['average_field_accuracy']:+.1f}%)\")
    print(f\"  完美提取率: {base['perfect_extraction_rate']:.1f}% → {ft['perfect_extraction_rate']:.1f}% (提升 {imp['perfect_extraction_rate']:+.1f}%)\")
    print(f\"\\n=== 分字段准确率 ===\")
    print(f\"  事件类型: {base['event_type_accuracy']:.1f}% → {ft['event_type_accuracy']:.1f}% ({imp['event_type_accuracy']:+.1f}%)\")
    print(f\"  标题: {base['title_accuracy']:.1f}% → {ft['title_accuracy']:.1f}% ({imp['title_accuracy']:+.1f}%)\")
    print(f\"  时间: {base['time_accuracy']:.1f}% → {ft['time_accuracy']:.1f}% ({imp['time_accuracy']:+.1f}%)\")
    print(f\"  地点: {base['location_accuracy']:.1f}% → {ft['location_accuracy']:.1f}% ({imp['location_accuracy']:+.1f}%)\")
    print(f\"  参与者: {base['participants_accuracy']:.1f}% → {ft['participants_accuracy']:.1f}% ({imp['participants_accuracy']:+.1f}%)\")
    print(f\"  组织者: {base['organizer_accuracy']:.1f}% → {ft['organizer_accuracy']:.1f}% ({imp['organizer_accuracy']:+.1f}%)\")
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

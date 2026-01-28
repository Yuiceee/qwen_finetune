#!/bin/bash

# 使用DeepSeek API审核和修正标注数据
# 重点改进participants和time字段

set -e

echo "======================================================"
echo "使用DeepSeek API审核和修正标注"
echo "======================================================"
echo ""

# 检查API密钥
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  请设置环境变量 DEEPSEEK_API_KEY"
    echo "  export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ API密钥已设置"
echo ""

# 检查评估结果文件
if [ ! -f "outputs/evaluation_results_v2.json" ]; then
    echo "⚠️  未找到评估结果文件: outputs/evaluation_results_v2.json"
    echo "  请先运行评估: bash run_step1.sh"
    exit 1
fi

echo "✓ 评估结果文件存在"
echo ""

# 创建输出目录
mkdir -p data/reviewed

echo "======================================================"
echo "第一步：审核训练集（完整模式）"
echo "======================================================"
echo ""
echo "配置："
echo "  - 输入: data/standardized/train.jsonl (1092样本)"
echo "  - 输出: data/reviewed/train.jsonl"
echo "  - 模式: 完整模式（审核所有样本）"
echo "  - 延迟: 0.5秒（避免API限流）"
echo ""

uv run python scripts/fix_annotations_with_llm.py \
    --input data/standardized/train.jsonl \
    --output data/reviewed/train.jsonl \
    --api_key "$DEEPSEEK_API_KEY" \
    --base_url "https://api.deepseek.com" \
    --model "deepseek-chat" \
    --delay 0.5

echo ""
echo "======================================================"
echo "第二步：审核验证集（全部样本）"
echo "======================================================"
echo ""
echo "配置："
echo "  - 输入: data/standardized/valid.jsonl (138样本)"
echo "  - 输出: data/reviewed/valid.jsonl"
echo "  - 模式: 完整模式（审核所有样本）"
echo ""

uv run python scripts/fix_annotations_with_llm.py \
    --input data/standardized/valid.jsonl \
    --output data/reviewed/valid.jsonl \
    --api_key "$DEEPSEEK_API_KEY" \
    --base_url "https://api.deepseek.com" \
    --model "deepseek-chat" \
    --delay 0.5

echo ""
echo "======================================================"
echo "第三步：审核测试集（全部样本）"
echo "======================================================"
echo ""
echo "配置："
echo "  - 输入: data/standardized/test.jsonl (137样本)"
echo "  - 输出: data/reviewed/test.jsonl"
echo "  - 模式: 完整模式（审核所有样本）"
echo ""

uv run python scripts/fix_annotations_with_llm.py \
    --input data/standardized/test.jsonl \
    --output data/reviewed/test.jsonl \
    --api_key "$DEEPSEEK_API_KEY" \
    --base_url "https://api.deepseek.com" \
    --model "deepseek-chat" \
    --delay 0.5

echo ""
echo "======================================================"
echo "审核完成！"
echo "======================================================"
echo ""
echo "📊 输出文件："
echo "  - 训练集: data/reviewed/train.jsonl"
echo "  - 验证集: data/reviewed/valid.jsonl"
echo "  - 测试集: data/reviewed/test.jsonl"
echo ""
echo "📋 审核报告："
echo "  - data/reviewed/train_review_report.json"
echo "  - data/reviewed/valid_review_report.json"
echo "  - data/reviewed/test_review_report.json"
echo ""
echo "🔍 查看审核报告："
echo "  cat data/reviewed/train_review_report.json | jq '.statistics'"
echo ""
echo "🚀 下一步："
echo "  1. 查看改进效果: cat data/reviewed/train_review_report.json | jq"
echo "  2. 使用改进后的数据重新训练:"
echo "     bash run_step1.sh  # 记得修改数据路径为 data/reviewed/"
echo ""

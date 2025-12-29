"""
对比微调前后模型在测试集上的表现
"""
import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import argparse
from collections import defaultdict

# 设置HuggingFace缓存目录
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/macroverse/public/database/huggingface/hub'
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = '/macroverse/public/database/huggingface/hub'


def load_base_model(model_name):
    """加载基础模型（未微调）"""
    print(f"加载基础模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(base_model_name, lora_path):
    """加载微调后的模型"""
    print(f"加载微调模型: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, email_content, max_new_tokens=512):
    """生成模型响应"""
    messages = [
        {"role": "system", "content": "你是一个专业的邮件事件信息提取助手。"},
        {"role": "user", "content": f"请从以下邮件中提取事件信息，包括标题、时间、地点、参与者等关键信息，以JSON格式输出。\n\n邮件内容：\n{email_content}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response


def extract_json_from_text(text):
    """从文本中提取JSON（处理可能包含markdown代码块的情况）"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except:
        pass

    # 尝试从代码块中提取
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    # 尝试查找第一个完整的JSON对象
    json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass

    return None


def calculate_field_metrics(predicted_json, expected_json):
    """
    计算字段级别的指标
    返回：字段完整性、字段准确性等
    """
    if predicted_json is None or expected_json is None:
        return {
            'field_completeness': 0.0,
            'field_accuracy': 0.0,
            'has_all_fields': False,
            'exact_match': False
        }

    # 定义可能的关键字段（中英文）
    key_fields = ['title', 'time', 'location', 'participants',
                  '标题', '时间', '地点', '参与者', 'date', '日期']

    # 统计期望输出中存在的关键字段
    expected_fields = set()
    for field in key_fields:
        if field in expected_json:
            expected_fields.add(field)

    # 统计预测输出中存在的关键字段
    predicted_fields = set()
    for field in key_fields:
        if field in predicted_json:
            predicted_fields.add(field)

    # 计算字段完整性（预测输出包含了多少期望字段）
    if len(expected_fields) > 0:
        field_completeness = len(predicted_fields & expected_fields) / len(expected_fields) * 100
    else:
        field_completeness = 0.0

    # 计算字段准确性（匹配字段的值是否相同）
    matching_fields = 0
    common_fields = predicted_fields & expected_fields
    for field in common_fields:
        # 简单的字符串相似度比较（可以改进为更复杂的比较）
        if str(predicted_json[field]).strip() == str(expected_json[field]).strip():
            matching_fields += 1

    field_accuracy = (matching_fields / len(expected_fields) * 100) if len(expected_fields) > 0 else 0.0

    # 检查是否包含所有字段
    has_all_fields = len(predicted_fields & expected_fields) == len(expected_fields)

    # 检查是否完全匹配
    exact_match = (predicted_json == expected_json)

    return {
        'field_completeness': field_completeness,
        'field_accuracy': field_accuracy,
        'has_all_fields': has_all_fields,
        'exact_match': exact_match,
        'expected_field_count': len(expected_fields),
        'predicted_field_count': len(predicted_fields),
        'matching_field_count': matching_fields
    }


def evaluate_model(model, tokenizer, test_data, max_samples=None):
    """评估模型性能"""
    results = []

    # 统计指标
    valid_json_count = 0
    field_completeness_sum = 0.0
    field_accuracy_sum = 0.0
    exact_match_count = 0
    has_all_fields_count = 0

    samples = test_data[:max_samples] if max_samples else test_data

    for item in tqdm(samples, desc="评估中"):
        # 解析输入
        if 'messages' in item:
            email_content = item['messages'][1]['content'].split('邮件内容：\n')[-1]
            expected_output = item['messages'][2]['content']
        elif 'input' in item:
            email_content = item['input']
            expected_output = item['output']
        else:
            continue

        # 生成预测
        prediction = generate_response(model, tokenizer, email_content)

        # 提取JSON
        predicted_json = extract_json_from_text(prediction)
        expected_json = extract_json_from_text(expected_output)

        is_valid_json = (predicted_json is not None)
        if is_valid_json:
            valid_json_count += 1

        # 计算字段级指标
        field_metrics = calculate_field_metrics(predicted_json, expected_json)

        field_completeness_sum += field_metrics['field_completeness']
        field_accuracy_sum += field_metrics['field_accuracy']

        if field_metrics['exact_match']:
            exact_match_count += 1
        if field_metrics['has_all_fields']:
            has_all_fields_count += 1

        results.append({
            'input': email_content,
            'expected': expected_output,
            'prediction': prediction,
            'valid_json': is_valid_json,
            'predicted_json': predicted_json,
            'expected_json': expected_json,
            **field_metrics
        })

    # 计算总体指标
    total_samples = len(samples)
    metrics = {
        'json_format_accuracy': (valid_json_count / total_samples * 100) if total_samples > 0 else 0,
        'field_completeness': (field_completeness_sum / total_samples) if total_samples > 0 else 0,
        'field_accuracy': (field_accuracy_sum / total_samples) if total_samples > 0 else 0,
        'exact_match_rate': (exact_match_count / total_samples * 100) if total_samples > 0 else 0,
        'all_fields_present_rate': (has_all_fields_count / total_samples * 100) if total_samples > 0 else 0,
        'total_samples': total_samples,
        'valid_json_count': valid_json_count,
        'exact_match_count': exact_match_count,
        'all_fields_present_count': has_all_fields_count
    }

    return results, metrics


def main(args):
    print("=" * 60)
    print("模型对比评估")
    print("=" * 60)

    # 加载测试集
    print(f"\n加载测试集: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    if args.max_samples:
        print(f"使用前 {args.max_samples} 个样本进行测试")
        test_data = test_data[:args.max_samples]

    print(f"测试样本数: {len(test_data)}\n")

    # 评估基础模型
    print("\n" + "=" * 60)
    print("1. 评估基础模型（未微调）")
    print("=" * 60)
    base_model, base_tokenizer = load_base_model(args.base_model)
    base_results, base_metrics = evaluate_model(base_model, base_tokenizer, test_data)

    print(f"\n基础模型评估结果:")
    print(f"  JSON格式正确率: {base_metrics['json_format_accuracy']:.2f}%")
    print(f"  字段完整性: {base_metrics['field_completeness']:.2f}%")
    print(f"  字段准确性: {base_metrics['field_accuracy']:.2f}%")
    print(f"  完全匹配率: {base_metrics['exact_match_rate']:.2f}%")
    print(f"  所有字段存在率: {base_metrics['all_fields_present_rate']:.2f}%")

    # 释放基础模型内存
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # 评估微调模型
    print("\n" + "=" * 60)
    print("2. 评估微调模型")
    print("=" * 60)
    ft_model, ft_tokenizer = load_finetuned_model(args.base_model, args.lora_model)
    ft_results, ft_metrics = evaluate_model(ft_model, ft_tokenizer, test_data)

    print(f"\n微调模型评估结果:")
    print(f"  JSON格式正确率: {ft_metrics['json_format_accuracy']:.2f}%")
    print(f"  字段完整性: {ft_metrics['field_completeness']:.2f}%")
    print(f"  字段准确性: {ft_metrics['field_accuracy']:.2f}%")
    print(f"  完全匹配率: {ft_metrics['exact_match_rate']:.2f}%")
    print(f"  所有字段存在率: {ft_metrics['all_fields_present_rate']:.2f}%")

    # 对比结果
    print("\n" + "=" * 60)
    print("评估结果对比")
    print("=" * 60)
    print(f"{'指标':<25} {'基础模型':<15} {'微调模型':<15} {'提升':<10}")
    print("-" * 60)
    print(f"{'JSON格式正确率':<25} {base_metrics['json_format_accuracy']:>10.2f}%  {ft_metrics['json_format_accuracy']:>10.2f}%  {ft_metrics['json_format_accuracy']-base_metrics['json_format_accuracy']:>+8.2f}%")
    print(f"{'字段完整性':<25} {base_metrics['field_completeness']:>10.2f}%  {ft_metrics['field_completeness']:>10.2f}%  {ft_metrics['field_completeness']-base_metrics['field_completeness']:>+8.2f}%")
    print(f"{'字段准确性':<25} {base_metrics['field_accuracy']:>10.2f}%  {ft_metrics['field_accuracy']:>10.2f}%  {ft_metrics['field_accuracy']-base_metrics['field_accuracy']:>+8.2f}%")
    print(f"{'完全匹配率':<25} {base_metrics['exact_match_rate']:>10.2f}%  {ft_metrics['exact_match_rate']:>10.2f}%  {ft_metrics['exact_match_rate']-base_metrics['exact_match_rate']:>+8.2f}%")
    print(f"{'所有字段存在率':<25} {base_metrics['all_fields_present_rate']:>10.2f}%  {ft_metrics['all_fields_present_rate']:>10.2f}%  {ft_metrics['all_fields_present_rate']-base_metrics['all_fields_present_rate']:>+8.2f}%")

    # 保存详细结果
    if args.output_file:
        print(f"\n保存详细结果到: {args.output_file}")
        output = {
            'base_metrics': base_metrics,
            'finetuned_metrics': ft_metrics,
            'improvements': {
                'json_format_accuracy': ft_metrics['json_format_accuracy'] - base_metrics['json_format_accuracy'],
                'field_completeness': ft_metrics['field_completeness'] - base_metrics['field_completeness'],
                'field_accuracy': ft_metrics['field_accuracy'] - base_metrics['field_accuracy'],
                'exact_match_rate': ft_metrics['exact_match_rate'] - base_metrics['exact_match_rate'],
                'all_fields_present_rate': ft_metrics['all_fields_present_rate'] - base_metrics['all_fields_present_rate']
            },
            'samples': []
        }

        for i, (base_res, ft_res) in enumerate(zip(base_results, ft_results)):
            output['samples'].append({
                'id': i,
                'input': base_res['input'][:100] + '...',
                'expected': base_res['expected'],
                'base_prediction': base_res['prediction'],
                'base_valid_json': base_res['valid_json'],
                'base_field_completeness': base_res['field_completeness'],
                'base_field_accuracy': base_res['field_accuracy'],
                'ft_prediction': ft_res['prediction'],
                'ft_valid_json': ft_res['valid_json'],
                'ft_field_completeness': ft_res['field_completeness'],
                'ft_field_accuracy': ft_res['field_accuracy']
            })

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比评估基础模型和微调模型")

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基础模型名称")
    parser.add_argument("--lora_model", type=str, default="outputs/lora_model/final_model",
                        help="微调后的LoRA模型路径")
    parser.add_argument("--test_file", type=str, default="data/processed/test.jsonl",
                        help="测试数据文件路径")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="最大测试样本数（None表示使用全部）")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="结果输出文件路径")

    args = parser.parse_args()
    main(args)

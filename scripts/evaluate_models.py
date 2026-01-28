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


def normalize_string(s):
    """标准化字符串：去空格、转小写、处理None"""
    if s is None or s == 'None':
        return ''
    return str(s).strip().lower()


def normalize_list(lst):
    """标准化列表：处理string/list类型，保持原始顺序"""
    if isinstance(lst, str):
        # 处理逗号分隔的字符串
        lst = [item.strip() for item in lst.split(',')]
    elif not isinstance(lst, list):
        return []

    # 标准化每个元素并保持原始顺序
    return [normalize_string(item) for item in lst if item]


def calculate_core_field_metrics(predicted_json, expected_json):
    """
    计算核心字段级别的精确匹配指标
    返回每个字段的匹配情况和总体准确率
    """
    metrics = {
        'event_type_match': False,
        'title_match': False,
        'time_match': False,
        'location_match': False,
        'participants_match': False,
        'organizer_match': False,
        'field_accuracy': 0.0,  # 6个字段的平均准确率
        'all_fields_correct': False
    }

    if predicted_json is None or expected_json is None:
        return metrics

    match_count = 0
    total_fields = 6

    # 1. event_type - 精确匹配（不区分大小写）
    if normalize_string(predicted_json.get('event_type')) == normalize_string(expected_json.get('event_type')):
        metrics['event_type_match'] = True
        match_count += 1

    # 2. title - 精确匹配（去除首尾空格）
    if normalize_string(predicted_json.get('title')) == normalize_string(expected_json.get('title')):
        metrics['title_match'] = True
        match_count += 1

    # 3. time - 字符串精确匹配
    if normalize_string(str(predicted_json.get('time', ''))) == normalize_string(str(expected_json.get('time', ''))):
        metrics['time_match'] = True
        match_count += 1

    # 4. location - 精确匹配（处理null情况）
    pred_loc = normalize_string(str(predicted_json.get('location', 'null')))
    exp_loc = normalize_string(str(expected_json.get('location', 'null')))
    if pred_loc == exp_loc:
        metrics['location_match'] = True
        match_count += 1

    # 5. participants - 完全匹配（顺序+内容）
    pred_parts = normalize_list(predicted_json.get('participants', []))
    exp_parts = normalize_list(expected_json.get('participants', []))
    if pred_parts == exp_parts:
        metrics['participants_match'] = True
        match_count += 1

    # 6. organizer - 精确匹配
    if normalize_string(predicted_json.get('organizer')) == normalize_string(expected_json.get('organizer')):
        metrics['organizer_match'] = True
        match_count += 1

    metrics['field_accuracy'] = (match_count / total_fields) * 100
    metrics['all_fields_correct'] = (match_count == total_fields)

    return metrics


def evaluate_model(model, tokenizer, test_data, max_samples=None):
    """评估模型性能 - 使用新的核心字段指标"""
    results = []

    # 统计指标
    valid_json_count = 0

    # 按字段统计
    field_stats = {
        'event_type_correct': 0,
        'title_correct': 0,
        'time_correct': 0,
        'location_correct': 0,
        'participants_correct': 0,
        'organizer_correct': 0,
        'all_fields_correct': 0
    }

    field_accuracy_sum = 0.0

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

        # 计算核心字段指标
        field_metrics = calculate_core_field_metrics(predicted_json, expected_json)

        # 累加统计
        field_accuracy_sum += field_metrics['field_accuracy']

        if field_metrics['event_type_match']:
            field_stats['event_type_correct'] += 1
        if field_metrics['title_match']:
            field_stats['title_correct'] += 1
        if field_metrics['time_match']:
            field_stats['time_correct'] += 1
        if field_metrics['location_match']:
            field_stats['location_correct'] += 1
        if field_metrics['participants_match']:
            field_stats['participants_correct'] += 1
        if field_metrics['organizer_match']:
            field_stats['organizer_correct'] += 1
        if field_metrics['all_fields_correct']:
            field_stats['all_fields_correct'] += 1

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
        'average_field_accuracy': (field_accuracy_sum / total_samples) if total_samples > 0 else 0,
        'perfect_extraction_rate': (field_stats['all_fields_correct'] / total_samples * 100) if total_samples > 0 else 0,

        # 每个字段的准确率
        'event_type_accuracy': (field_stats['event_type_correct'] / total_samples * 100) if total_samples > 0 else 0,
        'title_accuracy': (field_stats['title_correct'] / total_samples * 100) if total_samples > 0 else 0,
        'time_accuracy': (field_stats['time_correct'] / total_samples * 100) if total_samples > 0 else 0,
        'location_accuracy': (field_stats['location_correct'] / total_samples * 100) if total_samples > 0 else 0,
        'participants_accuracy': (field_stats['participants_correct'] / total_samples * 100) if total_samples > 0 else 0,
        'organizer_accuracy': (field_stats['organizer_correct'] / total_samples * 100) if total_samples > 0 else 0,

        # 原始计数
        'total_samples': total_samples,
        'valid_json_count': valid_json_count,
        'perfect_extraction_count': field_stats['all_fields_correct']
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
    print(f"  平均字段准确率: {base_metrics['average_field_accuracy']:.2f}%")
    print(f"  完美提取率: {base_metrics['perfect_extraction_rate']:.2f}%")
    print(f"\n  分字段准确率:")
    print(f"    - 事件类型: {base_metrics['event_type_accuracy']:.2f}%")
    print(f"    - 标题: {base_metrics['title_accuracy']:.2f}%")
    print(f"    - 时间: {base_metrics['time_accuracy']:.2f}%")
    print(f"    - 地点: {base_metrics['location_accuracy']:.2f}%")
    print(f"    - 参与者: {base_metrics['participants_accuracy']:.2f}%")
    print(f"    - 组织者: {base_metrics['organizer_accuracy']:.2f}%")

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
    print(f"  平均字段准确率: {ft_metrics['average_field_accuracy']:.2f}%")
    print(f"  完美提取率: {ft_metrics['perfect_extraction_rate']:.2f}%")
    print(f"\n  分字段准确率:")
    print(f"    - 事件类型: {ft_metrics['event_type_accuracy']:.2f}%")
    print(f"    - 标题: {ft_metrics['title_accuracy']:.2f}%")
    print(f"    - 时间: {ft_metrics['time_accuracy']:.2f}%")
    print(f"    - 地点: {ft_metrics['location_accuracy']:.2f}%")
    print(f"    - 参与者: {ft_metrics['participants_accuracy']:.2f}%")
    print(f"    - 组织者: {ft_metrics['organizer_accuracy']:.2f}%")

    # 对比结果
    print("\n" + "=" * 60)
    print("评估结果对比")
    print("=" * 60)
    print(f"{'指标':<30} {'基础模型':<15} {'微调模型':<15} {'提升':<10}")
    print("-" * 70)
    print(f"{'JSON格式正确率':<30} {base_metrics['json_format_accuracy']:>10.2f}%  {ft_metrics['json_format_accuracy']:>10.2f}%  {ft_metrics['json_format_accuracy']-base_metrics['json_format_accuracy']:>+8.2f}%")
    print(f"{'平均字段准确率':<30} {base_metrics['average_field_accuracy']:>10.2f}%  {ft_metrics['average_field_accuracy']:>10.2f}%  {ft_metrics['average_field_accuracy']-base_metrics['average_field_accuracy']:>+8.2f}%")
    print(f"{'完美提取率（6字段全对）':<30} {base_metrics['perfect_extraction_rate']:>10.2f}%  {ft_metrics['perfect_extraction_rate']:>10.2f}%  {ft_metrics['perfect_extraction_rate']-base_metrics['perfect_extraction_rate']:>+8.2f}%")
    print("-" * 70)
    print(f"{'事件类型准确率':<30} {base_metrics['event_type_accuracy']:>10.2f}%  {ft_metrics['event_type_accuracy']:>10.2f}%  {ft_metrics['event_type_accuracy']-base_metrics['event_type_accuracy']:>+8.2f}%")
    print(f"{'标题准确率':<30} {base_metrics['title_accuracy']:>10.2f}%  {ft_metrics['title_accuracy']:>10.2f}%  {ft_metrics['title_accuracy']-base_metrics['title_accuracy']:>+8.2f}%")
    print(f"{'时间准确率':<30} {base_metrics['time_accuracy']:>10.2f}%  {ft_metrics['time_accuracy']:>10.2f}%  {ft_metrics['time_accuracy']-base_metrics['time_accuracy']:>+8.2f}%")
    print(f"{'地点准确率':<30} {base_metrics['location_accuracy']:>10.2f}%  {ft_metrics['location_accuracy']:>10.2f}%  {ft_metrics['location_accuracy']-base_metrics['location_accuracy']:>+8.2f}%")
    print(f"{'参与者准确率':<30} {base_metrics['participants_accuracy']:>10.2f}%  {ft_metrics['participants_accuracy']:>10.2f}%  {ft_metrics['participants_accuracy']-base_metrics['participants_accuracy']:>+8.2f}%")
    print(f"{'组织者准确率':<30} {base_metrics['organizer_accuracy']:>10.2f}%  {ft_metrics['organizer_accuracy']:>10.2f}%  {ft_metrics['organizer_accuracy']-base_metrics['organizer_accuracy']:>+8.2f}%")

    # 保存详细结果
    if args.output_file:
        print(f"\n保存详细结果到: {args.output_file}")
        output = {
            'base_metrics': base_metrics,
            'finetuned_metrics': ft_metrics,
            'improvements': {
                'json_format_accuracy': ft_metrics['json_format_accuracy'] - base_metrics['json_format_accuracy'],
                'average_field_accuracy': ft_metrics['average_field_accuracy'] - base_metrics['average_field_accuracy'],
                'perfect_extraction_rate': ft_metrics['perfect_extraction_rate'] - base_metrics['perfect_extraction_rate'],
                'event_type_accuracy': ft_metrics['event_type_accuracy'] - base_metrics['event_type_accuracy'],
                'title_accuracy': ft_metrics['title_accuracy'] - base_metrics['title_accuracy'],
                'time_accuracy': ft_metrics['time_accuracy'] - base_metrics['time_accuracy'],
                'location_accuracy': ft_metrics['location_accuracy'] - base_metrics['location_accuracy'],
                'participants_accuracy': ft_metrics['participants_accuracy'] - base_metrics['participants_accuracy'],
                'organizer_accuracy': ft_metrics['organizer_accuracy'] - base_metrics['organizer_accuracy']
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
                'base_field_accuracy': base_res['field_accuracy'],
                'base_all_fields_correct': base_res['all_fields_correct'],
                'ft_prediction': ft_res['prediction'],
                'ft_valid_json': ft_res['valid_json'],
                'ft_field_accuracy': ft_res['field_accuracy'],
                'ft_all_fields_correct': ft_res['all_fields_correct']
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
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大测试样本数（默认None表示使用全部）")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="结果输出文件路径")

    args = parser.parse_args()
    main(args)

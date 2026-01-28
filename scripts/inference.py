"""
推理测试脚本
加载LoRA微调后的模型进行邮件事件信息提取
支持对比基础模型和微调模型的输出结果
"""
import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from typing import Dict, Tuple

# 设置HuggingFace缓存目录
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/macroverse/public/database/huggingface/hub'
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = '/macroverse/public/database/huggingface/hub'


def load_model(base_model_name, lora_model_path=None):
    """
    加载模型和tokenizer

    Args:
        base_model_name: 基础模型名称
        lora_model_path: LoRA模型路径（如果为None则只加载基础模型）
    """
    print(f"加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 如果提供了LoRA模型路径，则加载LoRA权重
    if lora_model_path:
        print(f"加载LoRA权重: {lora_model_path}")
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()  # 合并LoRA权重到基础模型

    model.eval()
    return model, tokenizer


def extract_event_info(email_content, model, tokenizer, max_new_tokens=512):
    """
    从邮件中提取事件信息

    Args:
        email_content: 邮件内容
        model: 模型
        tokenizer: tokenizer
        max_new_tokens: 最大生成token数

    Returns:
        response: 提取的事件信息
        inference_time: 推理时间（秒）
    """
    start_time = time.time()

    # 构建对话消息
    messages = [
        {"role": "system", "content": "你是一个专业的邮件事件信息提取助手。"},
        {"role": "user", "content": f"请从以下邮件中提取事件信息，包括标题、时间、地点、参与者等关键信息，以JSON格式输出。\n\n邮件内容：\n{email_content}"}
    ]

    # 使用chat template格式化
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # 启用采样以使用temperature和top_p
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    inference_time = time.time() - start_time

    return response, inference_time


def compare_outputs(base_output: str, finetuned_output: str, ground_truth: str = None) -> Dict:
    """
    对比基础模型和微调模型的输出

    Args:
        base_output: 基础模型的输出
        finetuned_output: 微调模型的输出
        ground_truth: 真实标注（可选）

    Returns:
        对比结果字典
    """
    result = {
        'base_valid_json': False,
        'finetuned_valid_json': False,
        'base_parsed': None,
        'finetuned_parsed': None,
        'ground_truth_parsed': None,
        'differences': []
    }

    # 解析基础模型输出
    try:
        result['base_parsed'] = json.loads(base_output)
        result['base_valid_json'] = True
    except json.JSONDecodeError:
        result['base_parsed'] = None

    # 解析微调模型输出
    try:
        result['finetuned_parsed'] = json.loads(finetuned_output)
        result['finetuned_valid_json'] = True
    except json.JSONDecodeError:
        result['finetuned_parsed'] = None

    # 解析真实标注
    if ground_truth:
        try:
            result['ground_truth_parsed'] = json.loads(ground_truth)
        except json.JSONDecodeError:
            result['ground_truth_parsed'] = None

    # 如果两个模型都成功解析，对比字段差异
    if result['base_parsed'] and result['finetuned_parsed']:
        base_dict = result['base_parsed']
        ft_dict = result['finetuned_parsed']

        all_keys = set(base_dict.keys()) | set(ft_dict.keys())
        for key in all_keys:
            base_val = base_dict.get(key, '(缺失)')
            ft_val = ft_dict.get(key, '(缺失)')

            if base_val != ft_val:
                diff_item = {
                    'field': key,
                    'base_value': base_val,
                    'finetuned_value': ft_val
                }

                # 如果有真实标注，添加对比
                if result['ground_truth_parsed'] and key in result['ground_truth_parsed']:
                    diff_item['ground_truth'] = result['ground_truth_parsed'][key]
                    diff_item['base_correct'] = base_val == result['ground_truth_parsed'][key]
                    diff_item['finetuned_correct'] = ft_val == result['ground_truth_parsed'][key]

                result['differences'].append(diff_item)

    return result


def print_comparison(comparison: Dict, base_time: float, ft_time: float):
    """
    打印对比结果

    Args:
        comparison: 对比结果字典
        base_time: 基础模型推理时间
        ft_time: 微调模型推理时间
    """
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)

    # JSON格式正确性
    print(f"\nJSON格式:")
    print(f"  基础模型: {'✓ 正确' if comparison['base_valid_json'] else '✗ 错误'}")
    print(f"  微调模型: {'✓ 正确' if comparison['finetuned_valid_json'] else '✗ 错误'}")

    # 推理时间
    print(f"\n推理时间:")
    print(f"  基础模型: {base_time:.2f}秒")
    print(f"  微调模型: {ft_time:.2f}秒")
    print(f"  速度变化: {((ft_time - base_time) / base_time * 100):+.1f}%")

    # 字段差异
    if comparison['differences']:
        print(f"\n字段差异 ({len(comparison['differences'])}个):")
        print("-" * 80)
        for diff in comparison['differences']:
            print(f"\n  字段: {diff['field']}")
            print(f"    基础模型: {diff['base_value']}")
            print(f"    微调模型: {diff['finetuned_value']}")

            if 'ground_truth' in diff:
                print(f"    真实标注: {diff['ground_truth']}")
                print(f"    基础模型准确: {'✓' if diff['base_correct'] else '✗'}")
                print(f"    微调模型准确: {'✓' if diff['finetuned_correct'] else '✗'}")
    else:
        print("\n字段差异: 无差异（输出相同）")

    print("\n" + "=" * 80)


def main(args):
    print("=" * 50)
    print("邮件事件信息提取推理")
    print("=" * 50)

    # 对比模式：同时加载基础模型和微调模型
    if args.compare:
        print("\n对比模式：同时加载基础模型和微调模型")
        print(f"基础模型: {args.base_model}")
        print(f"微调模型: {args.lora_model}")

        # 加载基础模型
        print("\n[1/2] 加载基础模型...")
        base_model, base_tokenizer = load_model(args.base_model, lora_model_path=None)

        # 加载微调模型
        print("\n[2/2] 加载微调模型...")
        ft_model, ft_tokenizer = load_model(args.base_model, args.lora_model)

        print("\n✓ 两个模型加载完成！")
        print("=" * 50)
    else:
        # 单模型模式
        model, tokenizer = load_model(args.base_model, args.lora_model)
        print("\n✓ 模型加载完成！")
        print("=" * 50)

    # 如果提供了测试邮件文件，则批量测试
    if args.test_file:
        print(f"\n从文件读取测试邮件: {args.test_file}")
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]

        print(f"共 {len(test_data)} 条测试样本\n")

        # 对比模式
        if args.compare:
            for i, item in enumerate(test_data[:args.max_samples] if args.max_samples else test_data):
                print(f"\n{'=' * 80}")
                print(f"测试样本 {i + 1}/{min(len(test_data), args.max_samples or len(test_data))}")
                print(f"{'=' * 80}")

                # 支持不同数据格式
                if 'messages' in item:
                    email_content = item['messages'][1]['content'].split('邮件内容：\n')[-1]
                    expected_output = item['messages'][2]['content']
                elif 'input' in item:
                    email_content = item['input']
                    expected_output = item.get('output')
                else:
                    print("⚠️  数据格式不支持，跳过")
                    continue

                print(f"\n📧 输入邮件:")
                print(f"{email_content[:300]}..." if len(email_content) > 300 else email_content)

                # 基础模型推理
                print(f"\n[1/2] 基础模型推理中...")
                base_result, base_time = extract_event_info(email_content, base_model, base_tokenizer, args.max_new_tokens)

                # 微调模型推理
                print(f"[2/2] 微调模型推理中...")
                ft_result, ft_time = extract_event_info(email_content, ft_model, ft_tokenizer, args.max_new_tokens)

                # 显示原始输出
                if args.verbose:
                    print(f"\n📄 基础模型输出:\n{base_result}")
                    print(f"\n📄 微调模型输出:\n{ft_result}")
                    if expected_output:
                        print(f"\n📄 真实标注:\n{expected_output}")

                # 对比输出
                comparison = compare_outputs(base_result, ft_result, expected_output)
                print_comparison(comparison, base_time, ft_time)

        # 单模型模式
        else:
            total_inference_time = 0
            valid_json_count = 0

            for i, item in enumerate(test_data[:args.max_samples] if args.max_samples else test_data):
                print(f"\n{'=' * 50}")
                print(f"测试样本 {i + 1}/{min(len(test_data), args.max_samples or len(test_data))}")
                print(f"{'=' * 50}")

                # 支持不同数据格式
                if 'messages' in item:
                    email_content = item['messages'][1]['content'].split('邮件内容：\n')[-1]
                    expected_output = item['messages'][2]['content']
                elif 'input' in item:
                    email_content = item['input']
                    expected_output = item['output']
                else:
                    print("⚠️  数据格式不支持，跳过")
                    continue

                print(f"\n输入邮件：\n{email_content[:200]}..." if len(email_content) > 200 else f"\n输入邮件：\n{email_content}")

                result, inference_time = extract_event_info(email_content, model, tokenizer, args.max_new_tokens)
                total_inference_time += inference_time

                print(f"\n模型输出：\n{result}")
                print(f"\n推理时间: {inference_time:.2f}秒")

                # 验证JSON格式
                try:
                    json.loads(result)
                    valid_json_count += 1
                    print("✓ JSON格式正确")
                except json.JSONDecodeError:
                    print("✗ JSON格式错误")

            # 统计信息
            avg_time = total_inference_time / len(test_data[:args.max_samples] if args.max_samples else test_data)
            json_accuracy = valid_json_count / len(test_data[:args.max_samples] if args.max_samples else test_data) * 100

            print(f"\n{'=' * 50}")
            print("推理统计:")
            print(f"{'=' * 50}")
            print(f"平均推理时间: {avg_time:.2f}秒/样本")
            print(f"JSON格式正确率: {json_accuracy:.1f}%")
            print(f"{'=' * 50}")

    # 交互式测试
    elif args.interactive:
        print("\n进入交互式测试模式（输入 'quit' 退出）")
        print("=" * 50)

        while True:
            print("\n请输入邮件内容（输入 'quit' 退出）：")
            lines = []
            while True:
                line = input()
                if line.strip().lower() == 'quit':
                    print("退出程序")
                    return
                if line == "":  # 空行表示输入结束
                    break
                lines.append(line)

            if not lines:
                continue

            email_content = '\n'.join(lines)

            print("\n提取中...")
            result, inference_time = extract_event_info(email_content, model, tokenizer, args.max_new_tokens)

            print(f"\n提取结果：\n{result}")
            print(f"推理时间: {inference_time:.2f}秒")
            print("=" * 50)

    # 单个测试样例
    else:
        test_email = """主题：项目评审会议
发件人：项目经理 张三
收件人：开发团队

各位同事，

定于本周五（12月29日）下午3点在会议室B召开项目中期评审会议。请技术负责人和架构师务必参加，并准备项目进展汇报材料。

谢谢！
张三"""

        print(f"\n测试邮件：\n{test_email}")
        print("\n提取中...")

        result, inference_time = extract_event_info(test_email, model, tokenizer, args.max_new_tokens)

        print(f"\n提取结果：\n{result}")
        print(f"推理时间: {inference_time:.2f}秒")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="邮件事件信息提取推理脚本")

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基础模型名称")
    parser.add_argument("--lora_model", type=str, default=None,
                        help="LoRA模型路径（可选）")
    parser.add_argument("--test_file", type=str, default=None,
                        help="测试数据文件路径（JSONL格式）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="测试样本最大数量（用于快速测试）")
    parser.add_argument("--interactive", action="store_true",
                        help="启用交互式测试模式")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成token数")
    parser.add_argument("--compare", action="store_true",
                        help="对比基础模型和微调模型的输出（需要同时指定--lora_model）")
    parser.add_argument("--verbose", action="store_true",
                        help="在对比模式下显示完整的模型输出")

    args = parser.parse_args()

    # 验证参数
    if args.compare and not args.lora_model:
        parser.error("--compare 模式需要指定 --lora_model")

    main(args)

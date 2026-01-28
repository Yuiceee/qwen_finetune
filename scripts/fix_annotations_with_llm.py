"""
使用LLM（如DeepSeek）重新标注和修正数据
重点改进participants和time字段的质量
"""

import json
import argparse
import time
from openai import OpenAI
from tqdm import tqdm
import os


def create_review_prompt(email_content, current_annotation):
    """
    创建审核提示词，让LLM审核现有标注并提供改进建议
    """
    return f"""你是一个专业的数据标注审核专家。请仔细审核以下邮件的事件信息提取结果，特别关注participants和time字段的准确性。

邮件内容：
{email_content}

当前标注：
{json.dumps(current_annotation, ensure_ascii=False, indent=2)}

请完成以下任务：
1. 仔细阅读邮件，识别事件信息
2. 检查当前标注是否准确，特别是：
   - participants字段：是否完整列出所有参与者？是否混淆了组织者和参与者？
   - time字段：时间是否准确？格式是否为YYYY-MM-DD或YYYY-MM-DD HH:MM？
   - 其他字段是否也准确

3. 输出改进后的标注（JSON格式），必须包含以下6个字段：
   - event_type: 事件类型（如：会议、培训、活动等）
   - title: 事件标题
   - time: 时间（格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM，如果没有明确时间则为null）
   - location: 地点（如果没有则为null）
   - participants: 参与者列表（数组，只包含参与者，不包括组织者）
   - organizer: 组织者（如果没有则为null）

**重要规则：**
- 只输出纯JSON，不要有其他文字
- participants是数组，即使只有一个人
- 如果某个字段在邮件中没有提到，设为null
- time必须是标准格式YYYY-MM-DD或YYYY-MM-DD HH:MM
- 不要混淆participants和organizer

直接输出改进后的JSON："""


def review_annotation_with_llm(client, email_content, current_annotation, model="deepseek-chat", delay=0.5):
    """
    使用LLM审核并改进单个标注
    """
    prompt = create_review_prompt(email_content, current_annotation)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的邮件事件信息提取和数据标注审核专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度，更确定性的输出
        )

        result = response.choices[0].message.content.strip()

        # 清理可能的markdown格式
        if result.startswith('```'):
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]
            result = result.strip()

        # 解析JSON
        improved_annotation = json.loads(result)

        time.sleep(delay)  # 避免API限流
        return improved_annotation, None

    except Exception as e:
        return None, str(e)


def compare_annotations(original, improved):
    """
    对比原标注和改进后的标注，返回差异
    """
    differences = {}

    for field in ['event_type', 'title', 'time', 'location', 'participants', 'organizer']:
        orig_val = original.get(field)
        imp_val = improved.get(field)

        if orig_val != imp_val:
            differences[field] = {
                'original': orig_val,
                'improved': imp_val
            }

    return differences


def review_dataset(input_file, output_file, api_key, base_url, model,
                   max_samples=None, delay=0.5, focus_on_errors=False,
                   error_analysis_file=None):
    """
    审核整个数据集

    Args:
        focus_on_errors: 是否只审核模型预测错误的样本
        error_analysis_file: 评估结果文件，用于识别错误样本
    """

    # 初始化API客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"\n{'=' * 80}")
    print(f"使用LLM审核和修正标注")
    print(f"{'=' * 80}\n")
    print(f"模型: {model}")
    print(f"输入文件: {input_file}")
    print(f"总样本数: {len(data)}")

    # 如果设置了focus_on_errors，只审核错误样本
    samples_to_review = []

    if focus_on_errors and error_analysis_file and os.path.exists(error_analysis_file):
        print(f"\n聚焦模式：只审核模型预测错误的样本")

        with open(error_analysis_file, 'r') as f:
            eval_data = json.load(f)

        # 找出字段准确率低的样本（< 0.8）
        error_samples = []
        for s in eval_data['samples']:
            if s['ft_field_accuracy'] < 0.8:  # 字段准确率 < 80%
                error_samples.append(s['id'])

        print(f"找到 {len(error_samples)} 个需要改进的样本（字段准确率 < 80%）")

        # 筛选这些样本
        for i, item in enumerate(data):
            if i in error_samples:
                samples_to_review.append((i, item))
    else:
        # 审核所有样本
        samples_to_review = list(enumerate(data))

    if max_samples:
        samples_to_review = samples_to_review[:max_samples]

    print(f"将审核 {len(samples_to_review)} 个样本\n")

    # 统计信息
    stats = {
        'total': len(samples_to_review),
        'improved': 0,
        'unchanged': 0,
        'failed': 0,
        'field_changes': {
            'event_type': 0,
            'title': 0,
            'time': 0,
            'location': 0,
            'participants': 0,
            'organizer': 0
        }
    }

    # 审核结果
    review_results = []

    # 处理每个样本
    for idx, item in tqdm(samples_to_review, desc="审核中"):
        # 提取邮件内容和当前标注
        email_content = item['messages'][1]['content']
        current_output = json.loads(item['messages'][2]['content'])

        # 使用LLM审核
        improved_output, error = review_annotation_with_llm(
            client, email_content, current_output, model, delay
        )

        if error:
            stats['failed'] += 1
            review_results.append({
                'index': idx,
                'status': 'failed',
                'error': error
            })
            continue

        # 对比差异
        differences = compare_annotations(current_output, improved_output)

        if differences:
            stats['improved'] += 1

            # 统计各字段的改动
            for field in differences.keys():
                stats['field_changes'][field] += 1

            review_results.append({
                'index': idx,
                'status': 'improved',
                'differences': differences,
                'original': current_output,
                'improved': improved_output
            })

            # 更新数据
            item['messages'][2]['content'] = json.dumps(improved_output, ensure_ascii=False, indent=2)
        else:
            stats['unchanged'] += 1
            review_results.append({
                'index': idx,
                'status': 'unchanged'
            })

    # 保存改进后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 打印统计
    print(f"\n{'=' * 80}")
    print("审核完成")
    print(f"{'=' * 80}\n")
    print(f"总样本数: {stats['total']}")
    print(f"改进样本: {stats['improved']} ({stats['improved']/stats['total']*100:.1f}%)")
    print(f"无需改动: {stats['unchanged']} ({stats['unchanged']/stats['total']*100:.1f}%)")
    print(f"审核失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")

    print(f"\n各字段改动统计:")
    for field, count in sorted(stats['field_changes'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {field}: {count} ({count/stats['total']*100:.1f}%)")

    # 显示典型改进示例
    improved_samples = [r for r in review_results if r['status'] == 'improved']
    if improved_samples:
        print(f"\n{'=' * 80}")
        print("改进示例（前5个）")
        print(f"{'=' * 80}\n")

        for i, sample in enumerate(improved_samples[:5], 1):
            print(f"{i}. 样本#{sample['index']} 的改动:")
            for field, change in sample['differences'].items():
                print(f"   【{field}】")
                print(f"     原始: {change['original']}")
                print(f"     改进: {change['improved']}")
            print()

    # 保存审核报告
    report_file = output_file.replace('.jsonl', '_review_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'review_results': review_results
        }, f, ensure_ascii=False, indent=2)

    print(f"改进后的数据已保存到: {output_file}")
    print(f"审核报告已保存到: {report_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LLM审核和修正数据标注")

    # 数据参数
    parser.add_argument("--input", type=str, required=True,
                       help="输入数据文件")
    parser.add_argument("--output", type=str, required=True,
                       help="输出数据文件")

    # API参数
    parser.add_argument("--api_key", type=str, required=True,
                       help="API密钥")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                       help="API基础URL")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                       help="模型名称")

    # 处理参数
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最多审核的样本数（用于测试）")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="API调用间隔（秒）")
    parser.add_argument("--focus_on_errors", action="store_true",
                       help="只审核模型预测错误的样本")
    parser.add_argument("--error_analysis_file", type=str,
                       default="outputs/evaluation_results_v2.json",
                       help="评估结果文件，用于识别错误样本")

    args = parser.parse_args()

    review_dataset(
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_samples=args.max_samples,
        delay=args.delay,
        focus_on_errors=args.focus_on_errors,
        error_analysis_file=args.error_analysis_file
    )

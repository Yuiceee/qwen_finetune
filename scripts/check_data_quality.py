"""
数据质量检查脚本
分析训练数据的标注质量，找出潜在问题
"""

import json
from collections import Counter, defaultdict
import re

def check_data_quality(file_path):
    """检查JSONL数据质量"""

    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"\n{'=' * 80}")
    print(f"数据质量检查: {file_path}")
    print(f"{'=' * 80}\n")
    print(f"总样本数: {len(data)}\n")

    # 1. 检查输出字段的一致性
    print("=" * 80)
    print("1. 输出字段分析")
    print("=" * 80)

    field_counts = Counter()
    field_examples = defaultdict(list)

    for i, item in enumerate(data):
        if 'messages' in item:
            output = item['messages'][2]['content']
        elif 'output' in item:
            output = item['output']
        else:
            continue

        try:
            parsed = json.loads(output)
            for key in parsed.keys():
                field_counts[key] += 1
                if len(field_examples[key]) < 3:
                    field_examples[key].append((i, parsed[key]))
        except json.JSONDecodeError:
            pass

    print(f"\n字段出现频率 (总样本数: {len(data)}):")
    for field, count in field_counts.most_common():
        percentage = count / len(data) * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")

    print(f"\n字段示例:")
    for field, examples in list(field_examples.items())[:3]:
        print(f"\n  {field}:")
        for idx, value in examples[:2]:
            value_str = str(value)[:100]
            print(f"    样本{idx}: {value_str}")

    # 2. 检查time字段的格式
    print(f"\n{'=' * 80}")
    print("2. time字段格式分析 (最差字段)")
    print("=" * 80)

    time_formats = Counter()
    time_examples = []

    for i, item in enumerate(data[:50]):  # 只检查前50个
        if 'messages' in item:
            output = item['messages'][2]['content']
        elif 'output' in item:
            output = item['output']
        else:
            continue

        try:
            parsed = json.loads(output)
            if 'time' in parsed:
                time_val = str(parsed['time'])
                time_examples.append((i, time_val))

                # 分类time格式
                if re.search(r'\d{4}', time_val):
                    time_formats['包含年份'] += 1
                if re.search(r'(周|星期)', time_val):
                    time_formats['包含星期'] += 1
                if re.search(r'(\d{1,2}:\d{2}|\d{1,2}点)', time_val):
                    time_formats['包含具体时间'] += 1
                if '月' in time_val or re.search(r'\d{1,2}/\d{1,2}', time_val):
                    time_formats['包含日期'] += 1
        except json.JSONDecodeError:
            pass

    print(f"\ntime字段格式分布:")
    for format_type, count in time_formats.most_common():
        print(f"  {format_type}: {count}")

    print(f"\ntime字段示例 (前10个):")
    for idx, time_val in time_examples[:10]:
        print(f"  样本{idx}: {time_val}")

    # 3. 检查JSON格式错误
    print(f"\n{'=' * 80}")
    print("3. JSON格式问题检查")
    print("=" * 80)

    json_errors = []
    for i, item in enumerate(data):
        if 'messages' in item:
            output = item['messages'][2]['content']
        elif 'output' in item:
            output = item['output']
        else:
            continue

        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            if len(json_errors) < 5:
                json_errors.append((i, str(e), output[:200]))

    print(f"\nJSON格式错误数: {len(json_errors)}")
    if json_errors:
        print(f"\n错误示例:")
        for idx, error, content in json_errors[:3]:
            print(f"\n  样本{idx}:")
            print(f"    错误: {error}")
            print(f"    内容: {content}...")

    # 4. 检查输入长度分布
    print(f"\n{'=' * 80}")
    print("4. 输入邮件长度分布")
    print("=" * 80)

    input_lengths = []
    for item in data:
        if 'messages' in item:
            input_text = item['messages'][1]['content']
        elif 'input' in item:
            input_text = item['input']
        else:
            continue
        input_lengths.append(len(input_text))

    input_lengths.sort()
    print(f"\n邮件长度统计:")
    print(f"  最小: {min(input_lengths)} 字符")
    print(f"  最大: {max(input_lengths)} 字符")
    print(f"  平均: {sum(input_lengths)/len(input_lengths):.0f} 字符")
    print(f"  中位数: {input_lengths[len(input_lengths)//2]} 字符")

    # 5. 检查是否有事件信息的邮件
    print(f"\n{'=' * 80}")
    print("5. 非事件邮件检查")
    print("=" * 80)

    no_event_count = 0
    no_event_examples = []

    for i, item in enumerate(data[:100]):
        if 'messages' in item:
            input_text = item['messages'][1]['content']
        elif 'input' in item:
            input_text = item['input']
        else:
            continue

        # 检查是否包含常见的事件关键词
        event_keywords = ['会议', '活动', '培训', '讨论', '评审', 'meeting', 'conference',
                         'event', 'session', 'call', 'discussion']

        has_event_keyword = any(kw in input_text.lower() for kw in event_keywords)

        if not has_event_keyword:
            no_event_count += 1
            if len(no_event_examples) < 3:
                no_event_examples.append((i, input_text[:200]))

    print(f"\n可能不包含事件信息的邮件: {no_event_count}/100")
    if no_event_examples:
        print(f"\n示例:")
        for idx, content in no_event_examples:
            print(f"\n  样本{idx}: {content}...")

    print(f"\n{'=' * 80}")
    print("检查完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys

    files_to_check = [
        "data/processed/train.jsonl",
        "data/processed/valid.jsonl",
        "data/processed/test.jsonl"
    ]

    for file_path in files_to_check:
        try:
            check_data_quality(file_path)
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
        except Exception as e:
            print(f"检查 {file_path} 时出错: {e}")

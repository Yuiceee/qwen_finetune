"""
数据清理和标准化脚本
1. 统一输出schema（只保留核心6个字段）
2. 标准化time字段格式
3. 修复JSON格式错误
4. 验证所有样本的质量
"""

import json
import re
from datetime import datetime
from typing import Dict, Optional

def standardize_time_field(time_str: str) -> str:
    """
    标准化时间字段
    目标格式：YYYY-MM-DD HH:MM 或者 YYYY-MM-DD
    保留原始表达如果无法解析
    """
    if not time_str or time_str.lower() in ['null', 'none', 'n/a']:
        return ""

    # 如果已经是标准格式，直接返回
    if re.match(r'\d{4}-\d{2}-\d{2}', time_str):
        return time_str

    # 保留原始字符串，因为邮件中的时间表达方式很多样
    # 这里只做基本清理，不强制转换格式
    cleaned = time_str.strip()

    # 移除多余的空格
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned


def standardize_output(output_str: str) -> Optional[Dict]:
    """
    标准化输出JSON
    1. 只保留核心6个字段：event_type, title, time, location, participants, organizer
    2. 确保所有字段都存在（没有的设为null）
    3. 标准化时间格式
    """
    try:
        # 尝试解析JSON
        data = json.loads(output_str)

        # 处理列表情况（有些标注返回多个事件）
        if isinstance(data, list):
            if len(data) == 0:
                return None
            data = data[0]  # 只取第一个事件

        if not isinstance(data, dict):
            return None

        # 提取并标准化核心字段
        standardized = {
            "event_type": data.get("event_type", ""),
            "title": data.get("title", ""),
            "time": standardize_time_field(str(data.get("time", ""))),
            "location": data.get("location", ""),
            "participants": data.get("participants", []),
            "organizer": data.get("organizer", "")
        }

        # 确保participants是列表
        if not isinstance(standardized["participants"], list):
            if standardized["participants"]:
                standardized["participants"] = [str(standardized["participants"])]
            else:
                standardized["participants"] = []

        # 移除空字段（除了participants，它应该保持为空列表）
        for key in ["event_type", "title", "time", "location", "organizer"]:
            if not standardized[key] or str(standardized[key]).lower() in ['null', 'none', 'n/a']:
                standardized[key] = None

        return standardized

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"内容: {output_str[:200]}...")
        return None
    except Exception as e:
        print(f"处理错误: {e}")
        return None


def clean_dataset(input_file: str, output_file: str, min_fields: int = 3):
    """
    清理数据集

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_fields: 最少必需的非空字段数（event_type, title, time基本必须有）
    """
    print(f"\n{'=' * 80}")
    print(f"清理数据集: {input_file} -> {output_file}")
    print(f"{'=' * 80}\n")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"原始样本数: {len(data)}")

    cleaned_data = []
    skipped = []

    for i, item in enumerate(data):
        # 提取输入和输出
        if 'messages' in item:
            instruction = item['messages'][1]['content']
            output_str = item['messages'][2]['content']
        elif 'input' in item and 'output' in item:
            instruction = item['instruction'] + '\n\n' + item['input']
            output_str = item['output']
        else:
            skipped.append(('格式不支持', i))
            continue

        # 标准化输出
        standardized_output = standardize_output(output_str)

        if not standardized_output:
            skipped.append(('无法解析JSON', i))
            continue

        # 检查必需字段数量
        non_null_fields = sum(1 for k, v in standardized_output.items()
                             if k != 'participants' and v is not None and v != "")

        if non_null_fields < min_fields:
            skipped.append((f'有效字段不足{min_fields}个，只有{non_null_fields}个', i))
            continue

        # 构建新格式
        cleaned_item = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的邮件事件信息提取助手。"
                },
                {
                    "role": "user",
                    "content": instruction
                },
                {
                    "role": "assistant",
                    "content": json.dumps(standardized_output, ensure_ascii=False, indent=2)
                }
            ]
        }

        cleaned_data.append(cleaned_item)

    # 保存清理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n清理完成:")
    print(f"  保留样本: {len(cleaned_data)}")
    print(f"  跳过样本: {len(skipped)}")
    print(f"  保留比例: {len(cleaned_data)/len(data)*100:.1f}%")

    if skipped:
        print(f"\n跳过原因统计:")
        skip_reasons = {}
        for reason, _ in skipped:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count}")

    # 显示示例
    if cleaned_data:
        print(f"\n清理后的样本示例:")
        print(json.dumps(cleaned_data[0], ensure_ascii=False, indent=2)[:500] + "...")

    print(f"\n{'=' * 80}\n")
    return len(cleaned_data), len(skipped)


if __name__ == "__main__":
    import os

    # 创建清理后的数据目录
    os.makedirs("data/cleaned", exist_ok=True)

    # 清理所有数据集
    datasets = [
        ("data/processed/train.jsonl", "data/cleaned/train.jsonl"),
        ("data/processed/valid.jsonl", "data/cleaned/valid.jsonl"),
        ("data/processed/test.jsonl", "data/cleaned/test.jsonl")
    ]

    total_kept = 0
    total_skipped = 0

    for input_file, output_file in datasets:
        if os.path.exists(input_file):
            kept, skipped = clean_dataset(input_file, output_file, min_fields=3)
            total_kept += kept
            total_skipped += skipped
        else:
            print(f"文件不存在: {input_file}\n")

    print(f"\n{'=' * 80}")
    print(f"总体统计")
    print(f"{'=' * 80}")
    print(f"总保留样本: {total_kept}")
    print(f"总跳过样本: {total_skipped}")
    print(f"总保留比例: {total_kept/(total_kept+total_skipped)*100:.1f}%")
    print(f"{'=' * 80}\n")

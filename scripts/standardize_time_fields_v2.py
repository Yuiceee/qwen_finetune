"""
Time字段标准化脚本 V2 - 修复版
核心改进：从邮件日期字段提取年份，用于补全缺少年份的时间表达
"""

import json
import re
from datetime import datetime
from dateutil import parser
from typing import Optional, Tuple

def extract_year_from_email(email_content: str) -> Optional[int]:
    """
    从邮件内容中提取年份

    查找顺序：
    1. 邮件头部的"日期"字段
    2. 邮件内容中的明确年份
    """
    # 查找邮件日期字段: 日期：Mon, 30 Jul 2001 08:33:00 -0700 (PDT)
    date_match = re.search(r'日期[：:]\s*\w+,\s*\d+\s+\w+\s+(\d{4})', email_content)
    if date_match:
        return int(date_match.group(1))

    # 英文格式: Date: Mon, 30 Jul 2001
    date_match = re.search(r'Date[：:]\s*\w+,\s*\d+\s+\w+\s+(\d{4})', email_content, re.IGNORECASE)
    if date_match:
        return int(date_match.group(1))

    # 提取邮件中第一个出现的4位数年份（2000-2025范围内）
    year_match = re.search(r'\b(20[0-2][0-9])\b', email_content)
    if year_match:
        return int(year_match.group(1))

    return None


def parse_time_field(time_str: str, default_year: Optional[int] = None) -> Tuple[Optional[str], str]:
    """
    尝试解析时间字段

    Args:
        time_str: 时间字符串
        default_year: 默认年份（从邮件日期提取）

    Returns:
        (standardized_time, confidence)
        confidence: 'high' (完全解析), 'medium' (部分解析), 'low' (无法解析，保留原样)
    """
    if not time_str or str(time_str).strip() == '':
        return None, 'low'

    time_str = str(time_str).strip()

    # 跳过明确表示无时间的
    no_time_indicators = ['未明确指定', 'null', 'None', 'N/A', 'TBD', 'not specified']
    if any(indicator in time_str for indicator in no_time_indicators):
        return None, 'low'

    # 跳过过于模糊的描述
    vague_indicators = ['下周', '本周', '明天', '今天', '今晚', '星期', '周一', '周二', '周三', '周四', '周五', '周六', '周日',
                        'this week', 'next week', 'tomorrow', 'today', 'tonight']
    if time_str in vague_indicators:
        return None, 'low'

    # 1. 尝试直接解析标准格式
    # YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD
    iso_match = re.match(r'(\d{4}-\d{2}-\d{2})(\s+\d{2}:\d{2}(:\d{2})?)?', time_str)
    if iso_match:
        return iso_match.group(0).rstrip(':00'), 'high'

    # 2. 尝试解析中文日期格式: 2001年7月28日
    chinese_date = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', time_str)
    if chinese_date:
        year, month, day = chinese_date.groups()
        # 检查是否有时间
        chinese_time = re.search(r'(\d{1,2}):(\d{2})', time_str)
        if chinese_time:
            hour, minute = chinese_time.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:{minute}", 'high'
        else:
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}", 'high'

    # 3. 尝试解析常见的非标准格式
    # MM/DD/YYYY HH:MM
    us_date = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', time_str)
    if us_date:
        month, day, year = us_date.groups()
        # 检查时间
        time_part = re.search(r'(\d{1,2}):(\d{2})', time_str)
        if time_part:
            hour, minute = time_part.groups()
            # 检查AM/PM
            if 'PM' in time_str.upper() or 'pm' in time_str:
                if int(hour) < 12:
                    hour = str(int(hour) + 12)
            elif 'AM' in time_str.upper() or 'am' in time_str:
                if int(hour) == 12:
                    hour = '00'
            return f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:{minute}", 'high'
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}", 'high'

    # 4. 尝试使用dateutil解析英文日期
    try:
        # 移除一些干扰信息
        clean_str = re.sub(r'\(.*?\)', '', time_str)  # 移除括号内容
        clean_str = re.sub(r'(Houston|Dubai|PST|EST|CST|MST|PDT|EDT|CDT|MDT|Central|Eastern|Pacific|Mountain)\s*(time)?',
                          '', clean_str, flags=re.IGNORECASE)

        # 检查原始字符串是否包含年份
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', time_str))

        if has_year:
            # 包含年份，直接解析
            parsed = parser.parse(clean_str, fuzzy=True)
            confidence = 'high'
        elif default_year:
            # 没有年份但有默认年份，使用默认年份
            parsed = parser.parse(clean_str, fuzzy=True, default=datetime(default_year, 1, 1))
            confidence = 'medium'
        else:
            # 既没有年份也没有默认年份，标记为低置信度
            return time_str, 'low'

        # 如果原文有具体时间，保留时分
        if ':' in time_str or 'am' in time_str.lower() or 'pm' in time_str.lower():
            result = parsed.strftime('%Y-%m-%d %H:%M')
        else:
            result = parsed.strftime('%Y-%m-%d')

        return result, confidence
    except:
        pass

    # 5. 处理日期范围 (如果只有一个明确日期，提取第一个)
    # 例如: "May 31-June 1, 2001" 或 "2001年5月31日-6月1日"
    range_with_year = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日?[-~至]', time_str)
    if range_with_year:
        year, month, day = range_with_year.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}", 'medium'

    # 6. 无法解析，保留原样
    return time_str, 'low'


def analyze_time_fields(input_file: str, output_report: str):
    """
    分析数据集中time字段的可解析性
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    stats = {
        'total': 0,
        'empty': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0,
        'has_email_year': 0,
        'no_email_year': 0,
        'examples': {
            'high': [],
            'medium': [],
            'low': []
        }
    }

    for item in data:
        # 提取邮件内容和年份
        email_content = item['messages'][1]['content']
        email_year = extract_year_from_email(email_content)

        if email_year:
            stats['has_email_year'] += 1
        else:
            stats['no_email_year'] += 1

        output = json.loads(item['messages'][2]['content'])
        time_val = output.get('time')

        stats['total'] += 1

        if not time_val or time_val == 'None':
            stats['empty'] += 1
            continue

        standardized, confidence = parse_time_field(time_val, email_year)

        if confidence == 'high':
            stats['high_confidence'] += 1
            if len(stats['examples']['high']) < 10:
                stats['examples']['high'].append((time_val, standardized, email_year))
        elif confidence == 'medium':
            stats['medium_confidence'] += 1
            if len(stats['examples']['medium']) < 10:
                stats['examples']['medium'].append((time_val, standardized, email_year))
        else:
            stats['low_confidence'] += 1
            if len(stats['examples']['low']) < 10:
                stats['examples']['low'].append((time_val, standardized, email_year))

    # 打印报告
    print(f"\n{'=' * 80}")
    print(f"Time字段解析分析 (V2 - 修复版): {input_file}")
    print(f"{'=' * 80}\n")

    print(f"总样本数: {stats['total']}")
    print(f"空time字段: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)")
    print(f"\n邮件年份提取:")
    print(f"  成功提取年份: {stats['has_email_year']} ({stats['has_email_year']/stats['total']*100:.1f}%)")
    print(f"  无法提取年份: {stats['no_email_year']} ({stats['no_email_year']/stats['total']*100:.1f}%)")

    print(f"\n解析置信度分布:")
    print(f"  高置信度 (完全解析): {stats['high_confidence']} ({stats['high_confidence']/stats['total']*100:.1f}%)")
    print(f"  中置信度 (部分解析): {stats['medium_confidence']} ({stats['medium_confidence']/stats['total']*100:.1f}%)")
    print(f"  低置信度 (保留原样): {stats['low_confidence']} ({stats['low_confidence']/stats['total']*100:.1f}%)")

    parseable = stats['high_confidence'] + stats['medium_confidence']
    print(f"\n✅ 可标准化率: {parseable} / {stats['total']} = {parseable/stats['total']*100:.1f}%")

    for conf_level in ['high', 'medium', 'low']:
        if stats['examples'][conf_level]:
            print(f"\n{conf_level.upper()}置信度示例:")
            for item in stats['examples'][conf_level]:
                if len(item) == 3:
                    orig, std, year = item
                    print(f"  原始: {orig}")
                    print(f"  标准: {std}")
                    print(f"  邮件年份: {year if year else 'N/A'}")
                    print()
                else:
                    orig, std = item
                    print(f"  原始: {orig}")
                    print(f"  标准: {std}")
                    print()

    # 保存报告
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"详细报告已保存到: {output_report}")
    print(f"{'=' * 80}\n")

    return stats


def standardize_dataset_times(input_file: str, output_file: str, min_confidence: str = 'medium'):
    """
    标准化数据集中的time字段

    Args:
        min_confidence: 'high', 'medium', 'low' - 最低接受的置信度
    """
    confidence_levels = {'high': 2, 'medium': 1, 'low': 0}
    min_level = confidence_levels[min_confidence]

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    standardized_data = []
    skipped = 0

    for item in data:
        # 提取邮件年份
        email_content = item['messages'][1]['content']
        email_year = extract_year_from_email(email_content)

        output = json.loads(item['messages'][2]['content'])
        time_val = output.get('time')

        if time_val and time_val != 'None':
            standardized, confidence = parse_time_field(time_val, email_year)

            # 如果置信度不够，跳过该样本
            if confidence_levels[confidence] < min_level:
                skipped += 1
                continue

            output['time'] = standardized if standardized else None

        # 更新assistant消息
        item['messages'][2]['content'] = json.dumps(output, ensure_ascii=False, indent=2)
        standardized_data.append(item)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in standardized_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n标准化完成:")
    print(f"  保留样本: {len(standardized_data)}")
    print(f"  跳过样本: {skipped}")
    print(f"  保留比例: {len(standardized_data)/(len(standardized_data)+skipped)*100:.1f}%\n")

    return len(standardized_data), skipped


if __name__ == "__main__":
    import os
    import sys

    # 先分析当前数据
    print("\n" + "=" * 80)
    print("第一步：分析time字段可解析性（修复版）")
    print("=" * 80)

    datasets = [
        ('data/cleaned/train.jsonl', 'data/cleaned/train_time_analysis_v2.json'),
        ('data/cleaned/valid.jsonl', 'data/cleaned/valid_time_analysis_v2.json'),
        ('data/cleaned/test.jsonl', 'data/cleaned/test_time_analysis_v2.json')
    ]

    all_stats = {}
    for dataset, report in datasets:
        if os.path.exists(dataset):
            stats = analyze_time_fields(dataset, report)
            all_stats[dataset] = stats

    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)

    total_samples = sum(s['total'] for s in all_stats.values())
    total_parseable = sum(s['high_confidence'] + s['medium_confidence'] for s in all_stats.values())

    print(f"\n总样本数: {total_samples}")
    print(f"可标准化样本: {total_parseable} ({total_parseable/total_samples*100:.1f}%)")

    # 询问是否执行标准化
    print("\n" + "=" * 80)
    print("第二步：执行标准化")
    print("=" * 80)

    if '--execute' in sys.argv or '--standardize' in sys.argv:
        print("\n开始执行标准化...")

        # 确定置信度级别
        min_conf = 'medium'  # 默认
        for arg in sys.argv:
            if arg.startswith('--min_confidence='):
                min_conf = arg.split('=')[1]

        print(f"使用最低置信度: {min_conf}\n")

        # 创建输出目录
        os.makedirs('data/standardized', exist_ok=True)

        # 标准化所有数据集
        standardize_tasks = [
            ('data/cleaned/train.jsonl', 'data/standardized/train.jsonl'),
            ('data/cleaned/valid.jsonl', 'data/standardized/valid.jsonl'),
            ('data/cleaned/test.jsonl', 'data/standardized/test.jsonl')
        ]

        total_kept = 0
        total_skipped = 0

        for input_file, output_file in standardize_tasks:
            if os.path.exists(input_file):
                print(f"\n处理: {input_file} -> {output_file}")
                kept, skipped = standardize_dataset_times(input_file, output_file, min_conf)
                total_kept += kept
                total_skipped += skipped

        print(f"\n{'=' * 80}")
        print("标准化总结")
        print(f"{'=' * 80}")
        print(f"总保留样本: {total_kept}")
        print(f"总跳过样本: {total_skipped}")
        print(f"总保留比例: {total_kept/(total_kept+total_skipped)*100:.1f}%")
        print(f"\n✓ 标准化数据已保存到: data/standardized/")
        print(f"{'=' * 80}\n")
    else:
        print("""
要执行标准化，请运行：

  推荐（保留中高置信度）:
    uv run python scripts/standardize_time_fields_v2.py --execute --min_confidence=medium

  保守（只保留高置信度）:
    uv run python scripts/standardize_time_fields_v2.py --execute --min_confidence=high

  宽松（保留所有）:
    uv run python scripts/standardize_time_fields_v2.py --execute --min_confidence=low
""")

"""
处理Enron emails.csv文件
从大文件中采样邮件数据，准备用于标注和训练
"""
import csv
import email
import json
import random
import sys
import hashlib
from pathlib import Path
from tqdm import tqdm
import argparse

# 增加CSV字段大小限制，处理大邮件
csv.field_size_limit(sys.maxsize)


def parse_email_message(message_text):
    """解析邮件文本"""
    try:
        msg = email.message_from_string(message_text)

        # 提取邮件头信息
        subject = msg.get('Subject', '')
        from_addr = msg.get('From', '')
        to_addr = msg.get('To', '')
        date = msg.get('Date', '')

        # 提取邮件正文
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload()

        # 清理和验证
        if not body or len(body.strip()) < 50:  # 过滤太短的邮件
            return None

        email_data = {
            'subject': subject.strip(),
            'from': from_addr.strip(),
            'to': to_addr.strip(),
            'date': date.strip(),
            'body': body.strip()
        }
        
        # 生成唯一标识符
        email_data['hash'] = generate_email_hash(email_data)
        
        return email_data
    except Exception as e:
        return None


def generate_email_hash(email_data):
    """为邮件生成唯一哈希值"""
    content = f"{email_data['subject']}|{email_data['from']}|{email_data['to']}|{email_data['date']}|{email_data['body'][:200]}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def load_existing_hashes(existing_files):
    """加载已有邮件的哈希值"""
    existing_hashes = set()
    
    for file_path in existing_files:
        if Path(file_path).exists():
            print(f"加载已有数据: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        email_data = json.loads(line.strip())
                        if 'hash' in email_data:
                            existing_hashes.add(email_data['hash'])
                        else:
                            # 如果没有哈希值，生成一个
                            email_hash = generate_email_hash(email_data)
                            existing_hashes.add(email_hash)
                    except json.JSONDecodeError:
                        continue
            print(f"已加载 {len(existing_hashes)} 个唯一邮件哈希")
    
    return existing_hashes


def count_valid_emails(csv_file, max_check=10000):
    """快速检查前N行，估算有效邮件数量"""
    print("检查数据质量...")
    valid_count = 0
    total_count = 0

    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_check:
                break
            total_count += 1
            if 'message' in row:
                parsed = parse_email_message(row['message'])
                if parsed:
                    valid_count += 1

    valid_ratio = valid_count / total_count if total_count > 0 else 0
    print(f"前{max_check}行中有效邮件: {valid_count}/{total_count} ({valid_ratio*100:.1f}%)")
    return valid_ratio


def sample_emails(csv_file, output_file, sample_size=500, seed=42, existing_hashes=None, avoid_duplicates=True):
    """
    从大CSV文件中采样邮件
    支持去重和增量采样

    Args:
        csv_file: 输入CSV文件
        output_file: 输出JSONL文件
        sample_size: 采样数量
        seed: 随机种子
        existing_hashes: 已有邮件的哈希值集合
        avoid_duplicates: 是否避免重复
    """
    random.seed(seed)
    
    if existing_hashes is None:
        existing_hashes = set()

    print(f"开始处理文件: {csv_file}")
    print(f"目标采样数量: {sample_size}")
    if avoid_duplicates and existing_hashes:
        print(f"已存在 {len(existing_hashes)} 个邮件，将避免重复")

    # 第一遍：统计总行数和有效邮件数
    print("\n第1步: 统计总行数...")
    total_lines = 0
    valid_emails = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(tqdm(reader, desc="解析邮件")):
            total_lines += 1
            if 'message' in row:
                parsed = parse_email_message(row['message'])
                if parsed:
                    # 检查是否重复
                    if avoid_duplicates and parsed['hash'] in existing_hashes:
                        continue
                    valid_emails.append((idx, parsed))

    print(f"总邮件数: {total_lines:,}")
    print(f"有效且非重复邮件数: {len(valid_emails):,}")

    # 第二步：采样
    print(f"\n第2步: 采样邮件...")
    
    if len(valid_emails) <= sample_size:
        print(f"有效邮件数量小于等于采样数量，将使用所有有效邮件")
        sampled_indices = [i for i, _ in valid_emails]
        sampled_emails = [email for _, email in valid_emails]
    else:
        # 随机采样
        sampled_pairs = random.sample(valid_emails, sample_size)
        sampled_indices = [idx for idx, _ in sampled_pairs]
        sampled_emails = [email for _, email in sampled_pairs]

    print(f"\n成功采样 {len(sampled_emails)} 封邮件")

    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for email_data in sampled_emails:
            f.write(json.dumps(email_data, ensure_ascii=False) + '\n')

    print(f"✓ 数据已保存到: {output_file}")

    # 显示示例
    if sampled_emails:
        print("\n" + "="*60)
        print("示例邮件（第1封）:")
        print("="*60)
        first = sampled_emails[0]
        print(f"主题: {first['subject']}")
        print(f"发件人: {first['from']}")
        print(f"收件人: {first['to']}")
        print(f"日期: {first['date']}")
        print(f"正文预览: {first['body'][:200]}...")
        print(f"哈希: {first['hash']}")
        print("="*60)

    return sampled_emails


def create_annotation_template(sampled_emails, output_file):
    """
    创建标注模板，便于后续标注
    Args:
        sampled_emails: 采样的邮件列表
        output_file: 输出文件
    """
    print(f"\n创建标注模板...")

    annotation_data = []
    for i, email_data in enumerate(sampled_emails):
        # 构建邮件内容文本
        email_text = f"""主题：{email_data['subject']}
发件人：{email_data['from']}
收件人：{email_data['to']}
日期：{email_data['date']}

{email_data['body']}"""

        annotation_data.append({
            'id': i,
            'email': email_text,
            'instruction': '请从以下邮件中提取事件信息，包括标题、时间、地点、参与者等关键信息，以JSON格式输出。',
            'annotation': None  # 待标注
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 标注模板已保存到: {output_file}")
    print(f"\n下一步:")
    print(f"1. 使用GPT-4批量生成标注:")
    print(f"   uv run python scripts/generate_annotations.py")
    print(f"2. 或手动标注这些邮件")


def main():
    parser = argparse.ArgumentParser(description="处理Enron emails.csv文件")
    parser.add_argument("--input", type=str, default="data/emails.csv",
                        help="输入CSV文件路径")
    parser.add_argument("--output", type=str, default="data/raw/enron_sampled.jsonl",
                        help="输出JSONL文件路径")
    parser.add_argument("--template", type=str, default="data/raw/annotation_template.json",
                        help="标注模板输出路径")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="采样数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--check_only", action="store_true",
                        help="仅检查数据质量，不进行采样")
    parser.add_argument("--avoid_duplicates", action="store_true",
                        help="避免与已有数据重复")
    parser.add_argument("--existing_files", type=str, nargs='+', default=[],
                        help="已有数据文件路径，用于去重")

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"✗ 文件不存在: {args.input}")
        return

    # 检查数据质量
    if args.check_only:
        count_valid_emails(args.input)
        return

    # 加载已有数据的哈希值
    existing_hashes = set()
    if args.avoid_duplicates and args.existing_files:
        existing_hashes = load_existing_hashes(args.existing_files)

    # 采样邮件
    sampled_emails = sample_emails(
        args.input,
        args.output,
        args.sample_size,
        args.seed,
        existing_hashes=existing_hashes,
        avoid_duplicates=args.avoid_duplicates
    )

    # 创建标注模板
    if sampled_emails:
        create_annotation_template(sampled_emails, args.template)

    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()

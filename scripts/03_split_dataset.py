"""
拆分数据集为 train/valid/test
"""
import json
import random
from pathlib import Path
import argparse


def split_dataset(input_file, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """拆分数据集"""
    # 验证比例
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"总数据: {len(data)} 条")

    # 过滤空标注（只保留有事件的数据）
    valid_data = [d for d in data if d['output'].strip() != '{}']
    print(f"有效数据: {len(valid_data)} 条 (过滤掉 {len(data) - len(valid_data)} 条空标注)")

    # 随机打乱
    random.seed(seed)
    random.shuffle(valid_data)

    # 计算分割点
    total = len(valid_data)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)

    train_data = valid_data[:train_size]
    valid_data_split = valid_data[train_size:train_size + valid_size]
    test_data = valid_data[train_size + valid_size:]

    print(f"\n拆分结果:")
    print(f"  Train: {len(train_data)} 条")
    print(f"  Valid: {len(valid_data_split)} 条")
    print(f"  Test:  {len(test_data)} 条")

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, dataset in [('train', train_data), ('valid', valid_data_split), ('test', test_data)]:
        file_path = output_path / f"{name}.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拆分数据集")
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    split_dataset(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed
    )

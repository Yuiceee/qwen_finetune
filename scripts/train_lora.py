"""
LoRA微调训练脚本
使用PEFT库对Qwen2.5-7B-Instruct进行低秩适应微调
集成Trackio进行实验跟踪
"""
import json
import torch
import time
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import argparse
import trackio

# 设置HuggingFace缓存目录
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/macroverse/public/database/huggingface'
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = '/macroverse/public/database/huggingface'


def load_jsonl_dataset(file_path):
    """加载JSONL格式的数据集，并转换为messages格式"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 转换 instruction-input-output 格式到 messages 格式
            if 'messages' not in item:
                messages = [
                    {"role": "system", "content": "你是一个专业的邮件事件信息提取助手。"},
                    {"role": "user", "content": f"{item['instruction']}\n\n邮件内容：\n{item['input']}"},
                    {"role": "assistant", "content": item['output']}
                ]
                item = {"messages": messages}
            data.append(item)
    return Dataset.from_list(data)


def format_chat_template(example, tokenizer):
    """将对话格式转换为模型输入格式"""
    # 使用Qwen的chat template
    text = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    model_inputs = tokenizer(
        text,
        max_length=2048,
        truncation=True,
        padding=False,
        return_tensors=None
    )

    # 设置labels（用于计算loss）
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def main(args):
    print("=" * 50)
    print("LoRA微调训练开始")
    print("=" * 50)

    # 初始化Trackio实验跟踪（只在本地使用，不上传到Space）
    trackio.init(
        project="email-lora-finetuning",
        space_id="Yurice/email-lora-finetuning",
        # 不设置 space_id，只在本地使用，不需要登录
        config={
            "model_name": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": 2048,
            "optimizer": "adamw",
            "warmup_steps": 100
        }
    )

    start_time = time.time()

    # 1. 加载模型和tokenizer
    print(f"\n1. 加载基础模型: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"  # 设置padding方向
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False  # 训练时关闭cache

    print(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")

    # 2. 配置LoRA
    print("\n2. 配置LoRA参数")
    lora_config = LoraConfig(
        r=args.lora_r,  # LoRA秩
        lora_alpha=args.lora_alpha,  # LoRA缩放因子
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"可训练参数: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.4f}%)")

    # 记录模型参数信息到Trackio
    trackio.log({
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "model_trainable_percentage": 100 * trainable_params / total_params
    })

    # 3. 加载数据集
    print(f"\n3. 加载训练数据: {args.train_data}")
    train_dataset = load_jsonl_dataset(args.train_data)
    print(f"训练样本数: {len(train_dataset)}")

    # 处理数据集
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="格式化训练数据"
    )

    # 如果有测试集，也加载
    eval_dataset = None
    if args.eval_data and Path(args.eval_data).exists():
        print(f"加载评估数据: {args.eval_data}")
        eval_dataset = load_jsonl_dataset(args.eval_data)
        print(f"评估样本数: {len(eval_dataset)}")
        eval_dataset = eval_dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            remove_columns=eval_dataset.column_names,
            desc="格式化评估数据"
        )

    # 4. 配置训练参数
    print("\n4. 配置训练参数")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        report_to="trackio",  # 使用Trackio
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        save_safetensors=True,
        remove_unused_columns=False
    )

    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")

    # 5. 创建Trainer
    print("\n5. 初始化Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # 6. 开始训练
    print("\n6. 开始训练...")
    print("=" * 50)
    trainer.train()

    # 7. 保存模型
    print("\n7. 保存模型")
    final_model_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"\n模型已保存到: {final_model_path}")

    # 记录训练总时间
    total_time = time.time() - start_time
    trackio.log({
        "total_training_time_seconds": total_time,
        "total_training_time_hours": total_time / 3600
    })

    print("=" * 50)
    print("训练完成！")
    print(f"总训练时间: {total_time / 3600:.2f} 小时")
    print("=" * 50)

    # 完成Trackio记录
    trackio.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA微调训练脚本")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基础模型名称")

    # 数据参数
    parser.add_argument("--train_data", type=str, default="data/processed/train.jsonl",
                        help="训练数据路径")
    parser.add_argument("--eval_data", type=str, default="data/processed/test.jsonl",
                        help="评估数据路径")

    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha参数")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/lora_model",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")

    args = parser.parse_args()

    main(args)

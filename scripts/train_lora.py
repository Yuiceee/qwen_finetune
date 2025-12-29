"""
LoRA微调训练脚本
使用PEFT库对Qwen2.5-7B-Instruct进行低秩适应微调
集成SwanLab进行实验跟踪
"""
import json
import torch
import time
import os
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import argparse
import swanlab
import numpy as np

# 设置HuggingFace缓存目录
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/macroverse/public/database/huggingface/hub/'
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = '/macroverse/public/database/huggingface/hub'


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


def compute_metrics(eval_preds):
    """
    计算评估指标
    - Perplexity（困惑度）：衡量模型对文本的预测能力，越低越好
    注意：为避免OOM，我们只使用loss来计算perplexity，不计算token accuracy
    """
    # 当 prediction_loss_only=True 时，这个函数可能不会被调用
    # 或者 predictions 可能为 None
    # 我们主要依赖 SwanLabCallback 中的 perplexity 计算
    return {}


class SwanLabCallback(TrainerCallback):
    """自定义SwanLab回调，用于记录额外的训练指标"""

    def __init__(self):
        self.start_time = None
        self.best_eval_loss = float('inf')
        self.best_eval_perplexity = float('inf')
        self.last_train_loss = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """训练开始时记录模型参数信息"""
        self.start_time = time.time()
        if model is not None:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            swanlab.log({
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "model_trainable_percentage": 100 * trainable_params / total_params
            })

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时保存训练loss"""
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时记录总时间"""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            swanlab.log({
                "total_training_time_seconds": total_time,
                "total_training_time_hours": total_time / 3600,
                "final_best_eval_loss": self.best_eval_loss,
                "final_best_eval_perplexity": self.best_eval_perplexity
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估后计算并记录额外的指标"""
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]

            # 1. Perplexity = e^(loss)
            perplexity = np.exp(eval_loss)
            metrics["eval_perplexity"] = float(perplexity)

            # 2. 跟踪最佳指标
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_eval_perplexity = perplexity
                metrics["is_best_eval"] = True
            else:
                metrics["is_best_eval"] = False

            # 3. 计算过拟合指标（如果有最近的训练loss）
            log_metrics = {
                "eval_perplexity": float(perplexity),
                "best_eval_loss": self.best_eval_loss,
                "best_eval_perplexity": self.best_eval_perplexity
            }

            if self.last_train_loss is not None:
                overfitting_gap = eval_loss - self.last_train_loss
                overfitting_ratio = eval_loss / self.last_train_loss if self.last_train_loss > 0 else 1.0

                metrics["overfitting_gap"] = float(overfitting_gap)
                metrics["overfitting_ratio"] = float(overfitting_ratio)

                log_metrics["overfitting_gap"] = float(overfitting_gap)
                log_metrics["overfitting_ratio"] = float(overfitting_ratio)

            # 记录所有指标到swanlab
            swanlab.log(log_metrics)


def main(args):
    print("=" * 50)
    print("LoRA微调训练开始")
    print("=" * 50)

    # 生成唯一的运行名称（包含时间戳和关键超参数）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lora_r{args.lora_r}_alpha{args.lora_alpha}_lr{args.learning_rate}_{timestamp}"

    # 初始化 SwanLab
    swanlab.init(
        project="email-lora-finetuning",
        experiment_name=run_name,
        config={
            "model_name": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )

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
        report_to="none",  # SwanLab通过callback记录
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model=args.metric_for_best_model if eval_dataset else None,
        greater_is_better=args.greater_is_better if eval_dataset else False,
        save_safetensors=True,
        remove_unused_columns=False,
        prediction_loss_only=True  # 只计算loss，不收集logits，避免OOM
    )

    print(f"输出目录: {args.output_dir}")
    print(f"运行名称: {run_name}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")

    # 5. 创建Trainer
    print("\n5. 初始化Trainer")

    # 准备回调函数
    callbacks = [SwanLabCallback()]

    # 添加早停回调（如果启用且有评估集）
    if args.early_stopping and eval_dataset:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping)
        print(f"早停已启用: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_metrics if eval_dataset else None,  # 添加评估指标
        callbacks=callbacks
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
    print("=" * 50)
    print("训练完成！")
    print("=" * 50)

    # 8. 完成 SwanLab 记录
    swanlab.finish()


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

    # 早停参数
    parser.add_argument("--early_stopping", action="store_true",
                        help="启用早停机制")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="早停耐心值：在多少个评估步骤后如果指标没有改善则停止训练")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help="早停阈值：指标改善的最小变化量")

    # 评估指标参数
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                        choices=["eval_loss", "eval_perplexity", "eval_accuracy"],
                        help="用于选择最佳模型的指标")
    parser.add_argument("--greater_is_better", action="store_true",
                        help="指标是否越大越好（如accuracy）。默认False（如loss）")

    args = parser.parse_args()

    main(args)

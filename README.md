推理加速项目

优化邮件归类事件提取的模型性能。
背景是用户邮件快速创建过程中云端API延迟2-3秒且持续调用计费不适合高频场景。方案是基于本地部署微调的开源模型Qwen2.5-7B-Instruct，通过三步优化实现性能和成本的平衡。

第一步：LoRA微调
使用PEFT库的LoRA实现低秩适应
只训练0.18%参数（1440万 vs 76亿）
将任务准确率从基础模型的65%提升到92%
选择LoRA而非全量微调是基于成本和效率考虑
训练数据来源：真实用户邮件标注样本、GPT-4生成的合成数据、Enron Email Dataset公开数据集
共500条样本，在Azure AI Foundry上训练完成
第二步：DPO对齐
使用TRL库的Direct Preference Optimization进行输出质量优化
构建了200对'chosen-rejected'偏好数据
rejected样本通过规则扰动生成（添加冗余字段、格式错误、信息缺失）
结合小量GPT-4生成的负样本
在LoRA基础上训练1小时
JSON格式规范性从85%提升到98%
显著减少了格式错误和冗余输出
第三步：量化部署
技术选择：对比了ONNX Runtime、PyTorch Mobile和llama.cpp三方案
最终选择llama.cpp用于CPU推理的深度优化
包括SIMD并行计算和内存管理优化
比ONNX快2.5倍
使用torch.ao.quantization库进行Q4_K_M混合精度Post-Training Quantization
模型从14GB压缩到4.3GB
推理速度从154步降到43秒，提升3.5倍
准确率仅损失2.5%（92%→89.5%）
方案在可接受范围内





## 预处理数据

1. 从emails.csv 中使用01_process_enron_csv.py 的脚本生成初步预处理的（随机采样的）483邮件
2. 使用llm api 使用 02_generate_annotations.py 脚本生成标注训练文件
3. 拆分train valid test


```shell
# 1 
uv run python scripts/01_process_enron_csv.py \
    --input "data/emails.csv" \
    --output "data/raw/enron_sampled_2000.jsonl" \
    --template "data/raw/annotation_template_2000.json" \
    --sample_size 2000 \
    --seed 42
# 2
uv run python scripts/02_generate_annotations.py \
    --api_key "432b1628-2b74-4fba-800a-3be77a46734f" \
    --base_url "https://ark.cn-beijing.volces.com/api/v3" \
    --model "doubao-seed-1-6-flash-250828" \
    --template "data/raw/annotation_template_2000.json" \
    --output "data/raw/train_data_all.jsonl" \
    --delay 0.5


uv run python scripts/02_generate_annotations.py \
    --api_key "sk-2d33bbad33544bb08f82b23163a86871" \
    --base_url "https://api.deepseek.com" \
    --model "deepseek-chat" \
    --template "data/raw/annotation_template_2000.json" \
    --output "data/raw/train_data_all.jsonl" \
    --delay 0.5

# 3
uv run python scripts/03_split_dataset.py \
    --input "data/raw/02_train_data_more.jsonl" \
    --output_dir "data/processed"

```


第二次执行
```shell
uv run python scripts/01_process_enron_csv.py \
    --input data/emails.csv \
    --output data/raw/more_samples.jsonl \
    --sample_size 10000 \
    --avoid_duplicates \
    --existing_files data/raw/enron_sampled_2000.jsonl


uv run python scripts/02_generate_annotations.py \
    --api_key "sk-2d33bbad33544bb08f82b23163a86871" \
    --base_url "https://api.deepseek.com" \
    --model "deepseek-chat" \
    --template "data/raw/annotation_template.json" \
    --output "data/raw/02_train_data_more.jsonl" \
    --delay 0.5

uv run python scripts/03_split_dataset.py \
    --input "data/raw/02_train_data_more.jsonl" \
    --output_dir "data/processed" 
# 总数据: 11928 条
# 有效数据: 1860 条 (过滤掉 10068 条空标注)

# 拆分结果:
#   Train: 1488 条
#   Valid: 186 条
#   Test:  186 条
# ✓ data/processed/train.jsonl
# ✓ data/processed/valid.jsonl
# ✓ data/processed/test.jsonl
```

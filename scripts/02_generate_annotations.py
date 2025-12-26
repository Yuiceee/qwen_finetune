"""
使用GPT-4为采样的邮件生成事件提取标注
支持中断继续和及时保存功能
"""
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import signal
import sys
from openai import OpenAI


def generate_annotation_with_gpt4(email_text, api_key, model="gpt-4", base_url=None, max_retries=3):
    """使用GPT-4或兼容API生成事件提取标注，支持重试机制"""
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    system_prompt = """你是一个专业的邮件事件信息提取助手。请从邮件中提取事件信息，包括：
- event_type: 事件类型（如会议、截止日期、客户拜访等）
- title: 事件标题
- time: 时间信息
- location: 地点（如果有）
- participants: 参与者列表
- organizer: 组织者
- 其他相关字段

请以JSON格式输出，只输出JSON，不要其他说明。如果邮件不包含明确的事件信息，返回空对象{}。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请从以下邮件中提取事件信息：\n\n{email_text}"}
                ],
                temperature=0.3,
                max_tokens=500
            )

            annotation = response.choices[0].message.content.strip()

            # 清理JSON格式
            if "```json" in annotation:
                annotation = annotation.split("```json")[1].split("```")[0].strip()
            elif "```" in annotation:
                annotation = annotation.split("```")[1].split("```")[0].strip()
            
            return annotation
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"达到最大重试次数，跳过此条数据")
                return "{}"  # 返回空JSON而不是抛出异常


def generate_simple_annotation(email_data):
    """生成简单的占位标注（用于快速测试）"""
    subject = email_data.get('subject', '')
    body = email_data.get('body', '')
    from_addr = email_data.get('from', '')

    annotation = {}

    if any(keyword in subject.lower() or keyword in body.lower()
           for keyword in ['meeting', 'conference', '会议', 'call']):
        annotation['event_type'] = 'meeting'
    elif any(keyword in subject.lower() or keyword in body.lower()
             for keyword in ['deadline', 'due', '截止', 'submit']):
        annotation['event_type'] = 'deadline'
    elif any(keyword in subject.lower() or keyword in body.lower()
             for keyword in ['visit', '拜访', 'appointment']):
        annotation['event_type'] = 'visit'

    if annotation:
        annotation['title'] = subject[:100] if subject else 'Untitled'
        annotation['organizer'] = from_addr

    return json.dumps(annotation, ensure_ascii=False) if annotation else "{}"


# 全局变量用于处理中断信号
interrupted = False
progress_file = None

def signal_handler(signum, frame):
    """处理中断信号"""
    global interrupted
    print("\n收到中断信号，正在保存当前进度...")
    interrupted = True

def save_progress(output_path, processed_count, total_count):
    """保存处理进度"""
    progress_data = {
        'processed_count': processed_count,
        'total_count': total_count,
        'timestamp': time.time()
    }
    progress_path = output_path.with_suffix('.progress.json')
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def load_progress(output_path):
    """加载处理进度"""
    progress_path = output_path.with_suffix('.progress.json')
    if progress_path.exists():
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def append_to_output(output_path, item):
    """追加单条数据到输出文件"""
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    global interrupted, progress_file
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="生成邮件事件提取标注")
    parser.add_argument("--template", type=str, default="data/raw/annotation_template.json")
    parser.add_argument("--output", type=str, default="data/raw/train_data.jsonl")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--use_simple", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--force_restart", action="store_true", help="强制重新开始，忽略已有进度")

    args = parser.parse_args()
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')

    if not args.use_simple and not api_key:
        raise ValueError("需要设置OPENAI_API_KEY环境变量或使用--api_key参数")

    # 设置输出路径
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载模板数据
    with open(args.template, 'r', encoding='utf-8') as f:
        template_data = json.load(f)

    if args.limit:
        template_data = template_data[:args.limit]

    total_count = len(template_data)
    start_index = 0
    processed_count = 0

    # 检查是否继续之前的任务
    if args.resume and not args.force_restart:
        progress = load_progress(output_path)
        if progress:
            start_index = progress['processed_count']
            processed_count = start_index
            print(f"从进度 {start_index}/{total_count} 处继续处理")
            
            # 检查输出文件是否存在，如果存在则备份
            if output_path.exists():
                backup_path = output_path.with_suffix('.backup.jsonl')
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(output_path, backup_path)
                    print(f"已备份现有输出文件到: {backup_path}")
        else:
            print("未找到进度文件，从头开始处理")

    # 如果是重新开始，清空输出文件
    if start_index == 0:
        with open(output_path, 'w', encoding='utf-8') as f:
            pass  # 清空文件

    print(f"开始处理 {total_count - start_index} 条数据 (总计: {total_count})")

    try:
        if args.use_simple:
            for i, item in enumerate(tqdm(template_data[start_index:], desc="生成标注", initial=start_index, total=total_count)):
                if interrupted:
                    break
                    
                email_lines = item['email'].split('\n')
                email_dict = {
                    'subject': email_lines[0].replace('主题：', '').strip() if email_lines else '',
                    'from': email_lines[1].replace('发件人：', '').strip() if len(email_lines) > 1 else '',
                    'body': '\n'.join(email_lines[4:]) if len(email_lines) > 4 else ''
                }
                annotation = generate_simple_annotation(email_dict)
                
                train_item = {
                    'instruction': item['instruction'],
                    'input': item['email'],
                    'output': annotation
                }
                
                # 立即保存到文件
                append_to_output(output_path, train_item)
                processed_count = start_index + i + 1
                
                # 每处理10条保存一次进度
                if processed_count % 10 == 0:
                    save_progress(output_path, processed_count, total_count)
        else:
            for i, item in enumerate(tqdm(template_data[start_index:], desc="生成标注", initial=start_index, total=total_count)):
                if interrupted:
                    break
                    
                annotation = generate_annotation_with_gpt4(
                    item['email'], api_key, args.model, args.base_url
                )
                
                if annotation:
                    train_item = {
                        'instruction': item['instruction'],
                        'input': item['email'],
                        'output': annotation
                    }
                    
                    # 立即保存到文件
                    append_to_output(output_path, train_item)
                    processed_count = start_index + i + 1
                    
                    # 每处理10条保存一次进度
                    if processed_count % 10 == 0:
                        save_progress(output_path, processed_count, total_count)
                
                if not interrupted:
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n用户中断处理")
        interrupted = True
    
    finally:
        # 保存最终进度
        save_progress(output_path, processed_count, total_count)
        
        if interrupted:
            print(f"处理已中断，已处理 {processed_count}/{total_count} 条数据")
            print(f"使用 --resume 参数可从断点继续处理")
        else:
            print(f"处理完成！共生成 {processed_count} 条训练数据，保存到: {args.output}")
            # 删除进度文件
            progress_path = output_path.with_suffix('.progress.json')
            if progress_path.exists():
                progress_path.unlink()


if __name__ == "__main__":
    main()

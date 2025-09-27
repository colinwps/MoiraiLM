# path: scripts/clean_wiki_text.py
import os
import json
import logging
import argparse
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_extracted_wiki(extracted_dir: str, output_file: str, min_line_length: int = 10):
    """
    读取WikiExtractor输出的JSON文件，提取、清洗文本并保存到单个文件中。
    """
    if not os.path.isdir(extracted_dir):
        logging.error(f"输入的目录不存在: {extracted_dir}")
        return

    total_articles = 0
    total_files = 0

    # 第一次遍历：获取所有需要处理的文件列表
    file_list = []
    for root, dirs, files in os.walk(extracted_dir):
        for file_name in files:
            # 仅处理 WikiExtractor 生成的以 'wiki_' 开头的文件
            if file_name.startswith('wiki_'):
                file_list.append(os.path.join(root, file_name))

    total_files = len(file_list)
    logging.info(f"找到 {total_files} 个文件等待处理。")
    if total_files == 0:
        logging.warning(f"目录 {extracted_dir} 中未找到任何 'wiki_' 文件。请检查路径。")
        return

    # 第二次遍历：处理文件并写入输出
    with open(output_file, 'w', encoding='utf-8') as f_out:

        # 使用 tqdm 包装文件列表，显示处理进度
        for file_path in tqdm(file_list, desc="📝 正在提取维基文本"):

            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        article = json.loads(line)
                        text_content = article.get('text', '').strip()

                        # --- 文本清洗和过滤 ---

                        # 1. 过滤掉过短的文章，它们通常是噪音或重定向页
                        if len(text_content) < 200:
                            continue

                        # 2. 移除文章标题（可选，通常在JSON输出中'text'字段包含标题）
                        title = article.get('title', '')
                        if title and text_content.startswith(title):
                            text_content = text_content[len(title):].lstrip('\n')

                        # 3. 按行处理文本，过滤短行和额外的空白
                        cleaned_lines = []
                        for text_line in text_content.split('\n'):
                            text_line = text_line.strip()
                            if len(text_line) >= min_line_length:
                                cleaned_lines.append(text_line)

                        final_text = ' '.join(cleaned_lines)

                        if final_text:
                            # 写入输出文件，用两个换行符分隔文章
                            f_out.write(final_text + '\n\n')
                            total_articles += 1

                    except json.JSONDecodeError:
                        logging.warning(f"无法解析 JSON 行，跳过: {file_path}")
                    except Exception as e:
                        logging.error(f"处理文件 {file_path} 时发生未知错误: {e}")

    logging.info(f"✅ 所有维基百科文本已成功提取。总文章数: {total_articles}。文件已保存到 {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="从 WikiExtractor 输出的 JSON 文件中提取并清洗纯文本。",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 位置参数 1: 输入目录
    parser.add_argument(
        "extracted_directory",
        type=str,
        help="WikiExtractor 输出的目录路径 (e.g., extracted_wiki_zh)"
    )

    # 位置参数 2: 输出文件
    parser.add_argument(
        "output_filename",
        type=str,
        help="最终合并的纯文本文件路径 (e.g., cleaned_wiki.txt)"
    )

    # 可选参数: 最小行长
    parser.add_argument(
        "--min_line_length",
        type=int,
        default=10,
        help="文章中单行文本必须达到的最小长度，用于过滤噪音。默认值: 10"
    )

    args = parser.parse_args()

    process_extracted_wiki(
        args.extracted_directory,
        args.output_filename,
        args.min_line_length
    )


if __name__ == "__main__":
    main()
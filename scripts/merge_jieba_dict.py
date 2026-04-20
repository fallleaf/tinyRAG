#!/usr/bin/env python3
"""
整合 jieba 用户自定义词典
合并三个词典文件，去重并统一频率
"""

from loguru import logger
from collections import defaultdict
from pathlib import Path

# 词典文件路径
data_dir = Path("/home/fallleaf/tinyRAG/data")
dict_files = [
    "jieba_user_dict.txt",
    "jieba_user_dict_16wen_jijian_2026.txt",
    "jieba_user_dict_carrier_2026.txt",
]

# 存储所有词条
entries = defaultdict(dict)  # {词: {freq: 频率, source: 来源文件}}

# 解析每个词典文件
for dict_file in dict_files:
    file_path = data_dir / dict_file
    if not file_path.exists():
        logger.info(f"警告: 文件不存在 {dict_file}")
        continue

    logger.info(f"解析文件: {dict_file}")
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 解析词条格式: 词 频率 词性
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    freq = int(parts[1])
                except ValueError:
                    # 如果频率不是数字，默认为 100
                    freq = 100

                # 保留最高频率
                if word not in entries or freq > entries[word]["freq"]:
                    entries[word] = {"freq": freq, "source": dict_file}

logger.info(f"\n总共找到 {len(entries)} 个唯一词条")

# 按频率排序
sorted_entries = sorted(entries.items(), key=lambda x: (-x[1]["freq"], x[0]))

# 输出整合后的词典
output_file = data_dir / "jieba_user_dict_merged.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("# jieba 用户自定义词典（整合版）\n")
    f.write(f"# 来源: {', '.join(dict_files)}\n")
    f.write(f"# 词条总数: {len(sorted_entries)}\n")
    f.write("# 生成时间: 2026-04-13\n")
    f.write("# 格式: 词 频率 词性\n")
    f.write("# 频率越高，分词时越优先匹配\n\n")

    for word, info in sorted_entries:
        f.write(f"{word} {info['freq']} n\n")

logger.info(f"\n整合完成，输出文件: {output_file}")
logger.info("词条统计:")
logger.info(f"  - 总词条数: {len(sorted_entries)}")
logger.info(f"  - 频率 >= 1000: {sum(1 for _, info in sorted_entries if info['freq'] >= 1000)}")
logger.info(f"  - 频率 >= 500: {sum(1 for _, info in sorted_entries if info['freq'] >= 500)}")
logger.info(f"  - 频率 >= 100: {sum(1 for _, info in sorted_entries if info['freq'] >= 100)}")

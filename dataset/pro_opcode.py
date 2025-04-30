import os
import re
import csv
from typing import List

"""
从 PHP 扩展的 stderr.txt 文件中提取操作码表，并将其转换为 CSV 格式
"""
def parse_opcode_table(lines: List[str]) -> List[dict]:
    entries = []

    pattern = re.compile(
        r'^\s*(\d+)?\s+(\d+)\s+([ >]*)\s*(\w+)'              # line, op_index, flags, opcode
        r'(?:\s+(\w+))?'                                      # fetch type
        r'(?:\s+(\~?\d+|\w+))?'                               # ext or return
        r'(?:\s+(\~?\d+|\w+))?'                               # return or extra
        r'\s*(.*)?$'                                          # operands
    )

    for line in lines:
        if not line.strip() or re.match(r'-{5,}', line):  # skip separators
            continue

        match = pattern.match(line)
        if not match:
            continue

        line_num, op_index, flags, opcode, fetch, ext, ret, operands = match.groups()
        entries.append({
            "line": line_num,
            "op_index": op_index,
            "flag": flags.strip(),
            "op": opcode,
            "fetch": fetch,
            "ext": ext,
            "ret": ret,
            "operands": operands.strip() if operands else ""
        })

    return entries

"""
将操作码表按表格分割
"""
def split_by_opcode_tables(content: str) -> List[List[str]]:
    tables = []
    current = []
    inside = False
    for line in content.splitlines():
        if "compiled vars" in line:
            inside = True
            current = []
        elif inside and re.match(r'\s*\d+\s+\d+\s+[ >]*\w+', line):
            current.append(line)
        elif inside and current and not line.strip():  # empty line indicates table end
            tables.append(current)
            inside = False
    if current:
        tables.append(current)
    return tables


"""
从文件中提取操作码表并保存为 CSV 文件
"""
def extract_from_file(filepath: str, output_dir: str = "./csv_output"):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    tables = split_by_opcode_tables(content)
    filename_prefix = os.path.basename(filepath).split('.')[0]

    os.makedirs(output_dir, exist_ok=True)

    for idx, table_lines in enumerate(tables):
        rows = parse_opcode_table(table_lines)
        outpath = os.path.join(output_dir, f"{filename_prefix}_{idx}.csv")

        with open(outpath, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "line", "op_index", "flag", "op", "fetch", "ext", "ret", "operands"
            ])
            writer.writeheader()
            writer.writerows(rows)

        print(f"[+] Wrote {outpath} ({len(rows)} rows)")


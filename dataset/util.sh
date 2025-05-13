#!/bin/bash

# 提示用户输入文件后缀名
read -p "请输入文件后缀名（例如 txt、php 等）： " extension

# 初始化文件计数
counter=1

echo "开始输入文件内容，按回车结束输入并保存文件，输入 'exit' 退出程序。"

while true; do
    # 提示用户输入文件内容
    echo "请输入文件内容（文件名：$counter.$extension）："
    content=""
    while IFS= read -r line; do
        # 如果用户输入 'exit'，退出程序
        if [[ "$line" == "exit" ]]; then
            echo "程序已退出。"
            exit 0
        fi
        # 如果用户输入空行，结束当前文件输入
        if [[ -z "$line" ]]; then
            break
        fi
        # 拼接输入内容
        content+="$line"$'\n'
    done

    # 将内容保存到文件
    echo -n "$content" > "$counter.$extension"
    echo "文件 $counter.$extension 已保存。"

    # 使用 cat 展示文件内容
    echo "========="
    cat "$counter.$extension"
    echo "========="
   

    # 文件计数递增
    ((counter++))
done
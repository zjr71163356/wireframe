#!/bin/bash
#chmod +x install_progressbar.sh
# 检查是否已安装pip
if ! command -v pip &> /dev/null
then
    echo "pip 未安装，请先安装 pip。"
    exit 1
fi

# 安装 progressbar 库
echo "正在安装 progressbar 库..."
pip install progressbar
# 安装完成提示
if [ $? -eq 0 ]; then
    echo "progressbar 库安装成功！"
else
    echo "安装失败，请检查错误信息。"
fi
echo "正在安装 progress 库..."
pip install progress
# 安装完成提示
if [ $? -eq 0 ]; then
    echo "progress 库安装成功！"
else
    echo "安装失败，请检查错误信息。"
fi
mkdir -p /home/featurize/logs && touch /home/featurize/logs/1.log

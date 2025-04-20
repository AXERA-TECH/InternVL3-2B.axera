#!/bin/bash

# 创建 datasets 目录（如果不存在）
mkdir -p datasets

echo "如果下载失败, 请从 releases 资源中手动下载."

# 下载到 datasets 目录，并添加错误处理
if ! wget -P datasets https://github.com/AXERA-TECH/InternVL3-2B.axera/releases/download/calibration/imagenet-calib.tar; then
    echo "错误：文件下载失败，请手动从以下链接下载："
    echo "https://github.com/AXERA-TECH/InternVL3-2B.axera/releases/download/calibration/imagenet-calib.tar"
    exit 1
fi

echo "下载成功！文件保存在 datasets 目录"
#!/usr/bin/env python3
"""跨平台重新打包项目为适用于Google Colab的zip文件

问题背景:
使用 Windows PowerShell Compress-Archive 生成的 zip 在 Colab 中可能出现文件名含反斜杠 (例如 'sdlm\arguments.py'),
导致目录结构丢失。此脚本使用 Python zipfile 并强制使用 POSIX ('/') 分隔符写入归档条目, 确保在 Linux/Colab 正常展开。

使用方法(在项目根目录 tess-diffusion 下运行):
  python make_colab_zip.py --output ../tess-diffusion-colab.zip

可选参数:
  --exclude-large-data  跳过超大原始数据文件(如 tess_train1.txt 等),减小上传尺寸。
  --extra-exclude PATTERN  追加排除模式,可多次使用,支持简单前缀/后缀匹配。

默认会包含:
  - 代码文件 (*.py, *.sh, *.json, *.yaml)
  - 配置、脚本、KG相关资源
  - tokenizer & 训练所需的 oneline 数据集 (tess_*_oneline.txt)

不会强制排除 README/文档, 如需缩减请使用 --extra-exclude。

验证:
  生成后可用 python -m zipfile -l ../tess-diffusion-colab.zip 查看, 路径应全部为正斜杠。
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).parent.resolve()

DEFAULT_LARGE_FILES = {
    'tess_train1.txt', 'tess_valid1.txt', 'tess_test1.txt',
    'tess_train1_oneline.txt.backup', 'tess_valid1_oneline.txt.backup'
}

KEEP_ONELINE = {
    'tess_train1_oneline.txt', 'tess_valid1_oneline.txt', 'tess_test1_oneline.txt'
}

def parse_args():
    p = argparse.ArgumentParser(description="打包项目为 Colab 友好 zip")
    p.add_argument('--output', type=Path, default=PROJECT_ROOT.parent / 'tess-diffusion-colab.zip',
                   help='输出zip路径 (默认: ../tess-diffusion-colab.zip)')
    p.add_argument('--exclude-large-data', action='store_true', help='排除默认的大体积原始数据文件')
    p.add_argument('--extra-exclude', action='append', default=[], help='附加排除模式(前缀或后缀匹配)')
    return p.parse_args()

def should_exclude(path: Path, args) -> bool:
    name = path.name
    if args.exclude_large_data and name in DEFAULT_LARGE_FILES:
        return True
    # 永远保留需要的 oneline 数据
    if name in KEEP_ONELINE:
        return False
    # 排除临时/缓存
    if name.endswith('.zip') or name.endswith('.egg-info'):
        return True
    if name in {'.git', '__pycache__'}:
        return True
    # 用户附加模式: 简单前缀或后缀
    for pattern in args.extra_exclude:
        if name.startswith(pattern) or name.endswith(pattern):
            return True
    return False

def iter_files(root: Path):
    for p in root.rglob('*'):
        if p.is_file():
            yield p

def make_posix_arcname(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return '/'.join(rel.parts)

def main():
    args = parse_args()
    out_path: Path = args.output
    if out_path.exists():
        print(f'[INFO] 删除已存在: {out_path}')
        out_path.unlink()

    to_add = []
    for f in iter_files(PROJECT_ROOT):
        if should_exclude(f, args):
            continue
        to_add.append(f)

    if not to_add:
        print('[ERROR] 没有文件可打包,请检查排除规则', file=sys.stderr)
        return 1

    print(f'[INFO] 开始打包 {len(to_add)} 个文件 → {out_path}')
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in to_add:
            arcname = make_posix_arcname(f, PROJECT_ROOT)
            zf.write(f, arcname)
    print('[INFO] 打包完成')
    print('[INFO] 验证路径示例:')
    with zipfile.ZipFile(out_path, 'r') as zf:
        for i, info in enumerate(zf.infolist()[:10]):
            print('  ', info.filename)
    print('[INFO] 所有条目应使用正斜杠 / 分隔')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

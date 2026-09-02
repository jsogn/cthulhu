#!/usr/bin/env bash
# 编译 DCT 重量化原生加速库（macOS，clang）。
#
# 用法：
#   scripts/build_native_dct.sh            # 当前机器架构
#   scripts/build_native_dct.sh arm64      # Apple Silicon
#   scripts/build_native_dct.sh x86_64     # Intel
#
# 双架构发布时分别编译两次并重命名归档；运行时按 CTHULHU_NATIVE_DCT_LIB
# 指定，或使用 native/ 目录下与架构匹配的 libdct_requant.dylib。
set -euo pipefail

cd "$(dirname "$0")/../src/cthulhu_backend/native"
arch="${1:-$(uname -m)}"
case "$arch" in
    arm64|aarch64) flag="-arch arm64" ;;
    x86_64) flag="-arch x86_64" ;;
    *) echo "unsupported arch: ${arch}" >&2; exit 1 ;;
esac

clang -O3 -funroll-loops $flag -dynamiclib -o libdct_requant.dylib dct_requant.c
echo "built native/libdct_requant.dylib (${arch})"

#!/usr/bin/env bash
# 下载静态 ffmpeg/ffprobe 到 packaging/ffmpeg（可选执行，需联网）。
# 多源重试：单个源超时或失败时自动切换下一个，避免海外源波动导致下载中断。
# 可用 FETCH_PLATFORM 指定目标平台：darwin-arm64 / darwin-x64 / win64；
# 缺省按当前机器平台自动判断。不同平台的二进制放入独立目录，互不覆盖。
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${FETCH_PLATFORM:-$(uname -s)/$(uname -m)}"

case "$TARGET" in
  Darwin/arm64)
    OUT_DIR="ffmpeg"
    KIND="mac-arm64"
    ;;
  Darwin/x86_64)
    OUT_DIR="ffmpeg-x64"
    KIND="mac-x64"
    ;;
  win64 | Windows/amd64 | Windows/x86_64)
    OUT_DIR="ffmpeg-win64"
    KIND="win64"
    ;;
  *)
    echo "不支持的平台标识：${TARGET}（可选 darwin-arm64 / darwin-x64 / win64）"
    exit 1
    ;;
esac

DIR="$ROOT/$OUT_DIR"
mkdir -p "$DIR"
echo "[fetch] 目标平台：${KIND}，输出目录：${DIR}"

FETCH() { # 用法：FETCH <名称> <URL>
  local name="$1" url="$2"
  local attempt
  for attempt in 1 2 3; do
    echo "[fetch] ${name} 尝试 ${attempt}/3：${url}"
    if curl -L --fail --connect-timeout 15 --max-time 300 -o "${DIR}/${name}" "$url"; then
      if [[ -s "${DIR}/${name}" ]]; then
        return 0
      fi
    fi
    echo "[fetch] ${name} 失败，等待 5 秒后重试"
    sleep 5
  done
  return 1
}

EXTRACT() { # 用法：EXTRACT <zip 路径> <目标目录>；兼容无 unzip 的 Windows 环境
  if command -v unzip >/dev/null 2>&1; then
    unzip -o "$1" -d "$2" >/dev/null 2>&1
  elif command -v powershell >/dev/null 2>&1 && command -v cygpath >/dev/null 2>&1; then
    powershell -NoProfile -Command \
      "Expand-Archive -Force -Path '$(cygpath -w "$1")' -DestinationPath '$(cygpath -w "$2")'"
  else
    return 1
  fi
}

UNZIP_ONE() { # 用法：UNZIP_ONE <zip 名> <期望输出名>
  local zip="$1" out="$2" tmp
  tmp="$(mktemp -d)"
  if EXTRACT "${DIR}/${zip}" "$tmp"; then
    local found
    found="$(find "$tmp" -type f -name "${out}" | head -1)"
    if [[ -n "$found" ]]; then
      mv "$found" "${DIR}/${out}"
      chmod +x "${DIR}/${out}"
      unlink "${DIR}/${zip}" 2>/dev/null || true
      return 0
    fi
  fi
  return 1
}

if [[ "$KIND" == "mac-arm64" ]]; then
  # Apple Silicon 首选原生 arm64 构建；evermeet 为 Intel 构建（依赖 Rosetta 2）作兜底。
  if FETCH ffmpeg.zip "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip" \
    && UNZIP_ONE ffmpeg.zip ffmpeg; then
    :
  elif FETCH ffmpeg.zip "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip" \
    && UNZIP_ONE ffmpeg.zip ffmpeg; then
    :
  fi
  if FETCH ffprobe.zip "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffprobe.zip" \
    && UNZIP_ONE ffprobe.zip ffprobe; then
    :
  elif FETCH ffprobe.zip "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip" \
    && UNZIP_ONE ffprobe.zip ffprobe; then
    :
  fi
elif [[ "$KIND" == "mac-x64" ]]; then
  if FETCH ffmpeg.zip "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip" \
    && UNZIP_ONE ffmpeg.zip ffmpeg; then
    :
  fi
  if FETCH ffprobe.zip "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip" \
    && UNZIP_ONE ffprobe.zip ffprobe; then
    :
  fi
elif [[ "$KIND" == "win64" ]]; then
  # Windows 静态构建源：BtbN/FFmpeg-Builds 的 win64-gpl 完整构建。
  if FETCH ffmpeg-win.zip "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"; then
    tmp="$(mktemp -d)"
    if EXTRACT "${DIR}/ffmpeg-win.zip" "$tmp"; then
      for exe in ffmpeg.exe ffprobe.exe; do
        found="$(find "$tmp" -type f -name "$exe" | head -1)"
        if [[ -n "$found" ]]; then
          mv "$found" "${DIR}/$exe"
        fi
      done
    fi
    unlink "${DIR}/ffmpeg-win.zip" 2>/dev/null || true
  fi
else
  echo "暂不支持的平台：$KIND"
  exit 1
fi

if [[ "$KIND" == "win64" ]]; then
  FFMPEG_BIN="$DIR/ffmpeg.exe"
  FFPROBE_BIN="$DIR/ffprobe.exe"
else
  FFMPEG_BIN="$DIR/ffmpeg"
  FFPROBE_BIN="$DIR/ffprobe"
fi

if [[ -x "$FFMPEG_BIN" && -x "$FFPROBE_BIN" ]]; then
  echo "静态 ffmpeg 已就绪：$DIR"
  "$FFMPEG_BIN" -version 2>/dev/null | head -1
else
  echo "下载未完成，请稍后重试或手动放置 ffmpeg / ffprobe 到 $DIR"
  exit 1
fi

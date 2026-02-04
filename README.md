# 雏鸡屁股视频抽帧（FFmpeg + OpenCV 清晰度筛选）

从 `mvs/male/`、`mvs/female/` 下的 `.avi` 视频抽取帧，并用 OpenCV 的清晰度指标（Laplacian 方差）筛选出更清晰的图片，分别保存到：

- `output/male/`
- `output/female/`

图片会直接平铺保存到对应性别目录下，通过文件名包含视频名来避免重名。

## 环境要求

- macOS / Linux / Windows 均可（本文示例为 macOS）
- Python 3
- 已安装且可用的 `ffmpeg`、`ffprobe`
- Python 依赖：`opencv-python`、`numpy`

检查 FFmpeg：

```bash
ffmpeg -version
ffprobe -version
```

## 安装依赖

如果你的 Python 环境不允许系统级 pip 安装（常见于 Homebrew 的 Python），推荐使用虚拟环境：

```bash
cd /Users/camel/Documents/GitHub/poc2026/jiji
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 用法

默认会处理 `mvs/male` 和 `mvs/female`：

```bash
. .venv/bin/activate
python extract_clear_frames.py
```

常用参数示例：

只跑 female，且只处理 1 个视频（调试用）：

```bash
python extract_clear_frames.py --categories female --limit-videos 1
```

提高抽帧频率（每秒 3 帧），每个视频最多保留 50 张：

```bash
python extract_clear_frames.py --fps 3 --keep-per-video 50
```

开启去重（减少相似画面）：

```bash
python extract_clear_frames.py --dedup
```

按清晰度只保留前 20% 的抽样帧（再结合每视频上限）：

```bash
python extract_clear_frames.py --top-percent 20 --keep-per-video 30
```

设置清晰度绝对阈值（Laplacian 方差），并覆盖已有输出：

```bash
python extract_clear_frames.py --min-score 15 --overwrite
```

## 输出结构

示例：

```
output/
  female/
    Video_20260204135254025__f000001__s3.87__r001.jpg
    ...
  male/
    Video_xxx__f000123__s10.12__r005.jpg
    ...
```

命名含义：

- `f000001`：抽帧序号（由 FFmpeg 导出的序号）
- `s3.87`：清晰度分数（Laplacian 方差，越大通常越清晰）
- `r001`：在该视频内保留序列中的顺序编号

## 测试步骤（建议）

1. 确认目录结构存在：

```bash
ls mvs/male
ls mvs/female
```

2. 确认 FFmpeg 可用：

```bash
ffmpeg -version
ffprobe -version
```

3. 创建虚拟环境并安装依赖（首次需要）：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

4. 先做小规模验证（只跑 1 个视频）：

```bash
python extract_clear_frames.py --categories female --limit-videos 1 --fps 1 --keep-per-video 5 --dedup
```

5. 检查导出结果：

```bash
find output/female -maxdepth 1 -type f | head
```

6. 全量跑完（male + female）：

```bash
python extract_clear_frames.py --fps 2 --keep-per-video 30 --dedup
```

## 排错与调参建议

- 输出太少：降低 `--min-score`，提高 `--keep-per-video`，或提高 `--fps`
- 输出重复多：开启 `--dedup`，并适当调大 `--dedup-delta`
- 需要定位抽帧情况：加 `--keep-temp` 查看临时抽帧目录（默认 `output/.tmp`）

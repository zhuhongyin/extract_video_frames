#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as e:
    missing = getattr(e, "name", None) or str(e)
    print(
        f"缺少Python依赖：{missing}。请先安装：pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2) from e


@dataclass(frozen=True)
class FrameScore:
    frame_path: Path
    score: float
    index: int


def _which_or_exit(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        print(f"未找到可执行文件：{name}。请先安装并确保在 PATH 中。", file=sys.stderr)
        raise SystemExit(2)
    return resolved


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("命令执行失败：", " ".join(cmd), file=sys.stderr)
        raise SystemExit(e.returncode) from e


def _parse_frame_index(p: Path) -> int:
    stem = p.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return -1
    try:
        return int(digits)
    except ValueError:
        return -1


def _laplacian_var_bgr(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _signature_gray(bgr: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _select_top_percent(frames: list[FrameScore], top_percent: float) -> list[FrameScore]:
    if not frames:
        return []
    p = max(0.0, min(100.0, top_percent))
    n = int(math.ceil(len(frames) * (p / 100.0)))
    n = max(1, n)
    return frames[:n]


def _pick_frames(
    scored: list[FrameScore],
    keep_per_video: int,
    min_score: Optional[float],
    top_percent: Optional[float],
    dedup: bool,
    dedup_delta: float,
) -> list[FrameScore]:
    if not scored:
        return []

    scored_sorted = sorted(scored, key=lambda x: x.score, reverse=True)
    candidates = scored_sorted

    if top_percent is not None:
        candidates = _select_top_percent(candidates, top_percent)

    if min_score is not None:
        filtered = [f for f in candidates if f.score >= min_score]
        candidates = filtered if filtered else candidates

    if keep_per_video <= 0:
        return []

    max_candidates = min(len(candidates), max(keep_per_video * 6, keep_per_video))
    candidates = candidates[:max_candidates]

    if not dedup:
        return candidates[:keep_per_video]

    kept: list[FrameScore] = []
    kept_sigs: list[np.ndarray] = []
    for frame in candidates:
        img = cv2.imread(str(frame.frame_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        sig = _signature_gray(img)
        is_dup = any(_mean_abs_diff(sig, prev) < dedup_delta for prev in kept_sigs)
        if is_dup:
            continue
        kept.append(frame)
        kept_sigs.append(sig)
        if len(kept) >= keep_per_video:
            break
    return kept


def _extract_frames_ffmpeg(
    ffmpeg_bin: str,
    video_path: Path,
    tmp_dir: Path,
    fps: float,
    jpeg_quality: int,
) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(tmp_dir / "%06d.jpg")
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(jpeg_quality),
        out_pattern,
    ]
    _run(cmd)
    return sorted(tmp_dir.glob("*.jpg"))


def _process_video(
    ffmpeg_bin: str,
    category: str,
    video_path: Path,
    output_category_dir: Path,
    tmp_root: Path,
    fps: float,
    keep_per_video: int,
    min_score: Optional[float],
    top_percent: Optional[float],
    dedup: bool,
    dedup_delta: float,
    jpeg_quality: int,
    overwrite: bool,
    keep_temp: bool,
    dry_run: bool,
) -> None:
    video_stem = video_path.stem
    out_dir = output_category_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = list(out_dir.glob(f"{video_stem}__f*.jpg")) + list(out_dir.glob(f"{video_stem}__f*.jpeg"))
        if existing:
            print(f"[跳过] {category}/{video_path.name} -> 已存在输出（{len(existing)}张）：{out_dir}")
            return

    tmp_dir = tmp_root / category / video_stem
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    if dry_run:
        print(f"[DRY-RUN] 将抽帧并筛选：{video_path} -> {out_dir}")
        return

    frame_paths = _extract_frames_ffmpeg(
        ffmpeg_bin=ffmpeg_bin,
        video_path=video_path,
        tmp_dir=tmp_dir,
        fps=fps,
        jpeg_quality=jpeg_quality,
    )
    if not frame_paths:
        print(f"[空] {category}/{video_path.name} 未抽取到帧")
        if not keep_temp and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return

    scored: list[FrameScore] = []
    for fp in frame_paths:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        score = _laplacian_var_bgr(img)
        scored.append(FrameScore(frame_path=fp, score=score, index=_parse_frame_index(fp)))

    kept = _pick_frames(
        scored=scored,
        keep_per_video=keep_per_video,
        min_score=min_score,
        top_percent=top_percent,
        dedup=dedup,
        dedup_delta=dedup_delta,
    )

    if overwrite:
        for old in out_dir.glob(f"{video_stem}__f*.jpg"):
            old.unlink()
        for old in out_dir.glob(f"{video_stem}__f*.jpeg"):
            old.unlink()

    kept_sorted_for_copy = sorted(kept, key=lambda x: x.index)
    for rank, frame in enumerate(kept_sorted_for_copy, start=1):
        dst_name = f"{video_stem}__f{frame.index:06d}__s{frame.score:.2f}__r{rank:03d}.jpg"
        shutil.copy2(frame.frame_path, out_dir / dst_name)

    scores = [f.score for f in scored]
    print(
        f"[完成] {category}/{video_path.name} 抽样={len(frame_paths)} 读取={len(scored)} 保留={len(kept_sorted_for_copy)} "
        f"score[min={min(scores):.2f}, max={max(scores):.2f}] -> {out_dir}"
    )

    if not keep_temp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def _iter_videos(dir_path: Path, suffixes: Iterable[str]) -> list[Path]:
    if not dir_path.exists():
        return []
    videos: list[Path] = []
    for suf in suffixes:
        videos.extend(dir_path.glob(f"*{suf}"))
    return sorted(set(videos))


def main() -> int:
    parser = argparse.ArgumentParser(description="从avi视频抽帧并筛选清晰图片（ffmpeg + OpenCV）")
    repo_root = Path(__file__).resolve().parent

    parser.add_argument("--input-root", type=Path, default=repo_root / "mvs", help="输入视频根目录，包含 male/ female/")
    parser.add_argument("--output-root", type=Path, default=repo_root / "output", help="输出图片根目录")
    parser.add_argument("--tmp-root", type=Path, default=repo_root / "output" / ".tmp", help="临时抽帧目录")

    parser.add_argument("--categories", nargs="*", default=["male", "female"], help="要处理的类别目录名")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧频率（每秒抽取多少帧）")
    parser.add_argument("--keep-per-video", type=int, default=30, help="每个视频最多保留多少张清晰图")
    parser.add_argument("--min-score", type=float, default=None, help="清晰度绝对阈值（Laplacian方差）")
    parser.add_argument("--top-percent", type=float, default=None, help="按清晰度保留前百分比（0-100）")

    parser.add_argument("--dedup", action="store_true", help="对候选帧做去重（减少相似画面）")
    parser.add_argument("--dedup-delta", type=float, default=4.0, help="去重差异阈值（越大越不容易判重复）")

    parser.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg导出jpg质量（2更高，31更低）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时抽帧文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要处理的内容，不实际抽帧")
    parser.add_argument("--limit-videos", type=int, default=None, help="最多处理多少个视频（调试用）")

    args = parser.parse_args()

    ffmpeg_bin = _which_or_exit("ffmpeg")
    _which_or_exit("ffprobe")

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    tmp_root: Path = args.tmp_root

    output_root.mkdir(parents=True, exist_ok=True)

    suffixes = [".avi", ".AVI"]

    total_videos = 0
    for category in args.categories:
        in_dir = input_root / category
        out_dir = output_root / category
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = _iter_videos(in_dir, suffixes=suffixes)
        if args.limit_videos is not None:
            videos = videos[: max(0, args.limit_videos)]

        if not videos:
            print(f"[无视频] {in_dir}")
            continue

        for vp in videos:
            total_videos += 1
            _process_video(
                ffmpeg_bin=ffmpeg_bin,
                category=category,
                video_path=vp,
                output_category_dir=out_dir,
                tmp_root=tmp_root,
                fps=float(args.fps),
                keep_per_video=int(args.keep_per_video),
                min_score=args.min_score,
                top_percent=args.top_percent,
                dedup=bool(args.dedup),
                dedup_delta=float(args.dedup_delta),
                jpeg_quality=int(args.jpeg_quality),
                overwrite=bool(args.overwrite),
                keep_temp=bool(args.keep_temp),
                dry_run=bool(args.dry_run),
            )

    if total_videos == 0:
        print("未找到任何可处理的视频。请检查 --input-root 和 --categories。", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

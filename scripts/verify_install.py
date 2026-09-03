"""Fast, non-rendering readiness check used by the launcher and installer."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]

def _ensure_ffmpeg_path():
    candidate_dirs = [
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"D:\ffmpeg\bin"),
        ROOT / "bin" / "ffmpeg" / "bin",
        ROOT / "bin",
    ]
    for d in candidate_dirs:
        if (d / "ffmpeg.exe").exists() and (d / "ffprobe.exe").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            break

_ensure_ffmpeg_path()

REQUIRED_ENGINES = (
    "vae_decoder_tile_256_21f.rtxplan",
    "vae_decoder_tile_512_5f.rtxplan",
    "vae_encoder_5f_tile512.rtxplan",
    "vae_encoder_21f_tile512.rtxplan",
)


def main() -> int:
    _ensure_ffmpeg_path()
    failures: list[str] = []
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            failures.append(f"{executable} is not on PATH")
    if not (ROOT / "vendor" / "seedvr2" / "inference_cli.py").exists():
        failures.append("vendored SeedVR2 source is missing")
    for module in ("torch", "fastapi", "uvicorn", "tensorrt_rtx", "onnx", "sageattention"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"cannot import {module}: {exc}")
    try:
        import torch
        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access an NVIDIA CUDA GPU")
    except Exception:
        pass
    artifacts = ROOT / "tensorrt_backend" / "artifacts"
    for name in REQUIRED_ENGINES:
        path = artifacts / name
        # If checking 5f decoder, allow either 256_5f or 512_5f
        if name == "vae_decoder_tile_512_5f.rtxplan":
            alt_path = artifacts / "vae_decoder_tile_256_5f.rtxplan"
            if (path.exists() and path.stat().st_size > 1_000_000) or (alt_path.exists() and alt_path.stat().st_size > 1_000_000):
                continue
        if not path.exists() or path.stat().st_size < 1_000_000:
            failures.append(f"TensorRT engine is missing: {name}")
    if failures:
        print("Installation is incomplete:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("SeedVR Studio installation is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

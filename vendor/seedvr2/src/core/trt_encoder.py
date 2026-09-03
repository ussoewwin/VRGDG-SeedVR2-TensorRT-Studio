# Modified by VRGDG SeedVR2 TensorRT Studio: adds TensorRT latent capture/encoder handoff integration.
"""Optional fixed-profile TensorRT VAE encoder used by SeedVR Studio."""

from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock

import torch
import tensorrt_rtx as trt


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT / "tensorrt_backend" / "artifacts"
_ENGINES: dict[int, tuple[object, object, object, str, str, torch.cuda.Stream]] = {}
_ENCODE_LOCK = Lock()


import subprocess
import sys

def _ensure_encoder_engine_file(frames: int) -> Path:
    if frames not in (5, 9, 13, 17, 21):
        raise ValueError(f"TensorRT VAE encoder supports temporal batches 5, 9, 13, 17, and 21, got {frames}")
    path = ARTIFACTS / f"vae_encoder_{frames}f_tile512.rtxplan"
    if not path.exists() or path.stat().st_size < 1_000_000:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        onnx_path = ARTIFACTS / f"vae_encoder_{frames}f_tile512.onnx"
        py = sys.executable
        if not onnx_path.exists():
            print(f"Exporting ONNX VAE encoder for {frames} frames...", flush=True)
            cmd_export = [
                str(py), "-u", str(ROOT / "tools" / "export_vae_encoder.py"),
                "--height", "512", "--width", "512", "--frames", str(frames), "--legacy-export",
                "--output", str(onnx_path)
            ]
            subprocess.run(cmd_export, cwd=ROOT, check=True)
        print(f"Building TensorRT VAE encoder engine for {frames} frames...", flush=True)
        cmd_build = [
            str(py), "-u", str(ROOT / "tools" / "build_tensorrt_engine.py"),
            str(onnx_path), "--output", str(path), "--workspace-gb", "8"
        ]
        subprocess.run(cmd_build, cwd=ROOT, check=True)
        if not path.exists():
            raise RuntimeError(f"Failed to build TensorRT encoder engine: {path}")
    return path


def _engine(frames: int):
    path = _ensure_encoder_engine_file(frames)
    cached = _ENGINES.get(frames)
    if cached is not None:
        return cached
    if not path.exists():
        raise FileNotFoundError(f"Missing TensorRT VAE encoder engine: {path}")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT encoder: {path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(
            f"TensorRT could not create an execution context for {path}. "
            "Rebuild the engine with scripts\\prepare_tensorrt.py."
        )
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    stream = torch.cuda.Stream()
    cached = (runtime, engine, context, input_name, output_name, stream)
    _ENGINES[frames] = cached
    return cached


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    """Cross-fade overlapping latent tiles instead of introducing hard averaging edges."""
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if left and overlap:
        weight[:overlap] = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
    if right and overlap:
        weight[-overlap:] = torch.minimum(
            weight[-overlap:], torch.linspace(1.0, 0.0, overlap + 1, device=device)[1:]
        )
    return weight


@torch.inference_mode()
def encode(sample: torch.Tensor) -> torch.Tensor:
    """Encode [B,3,T,H,W] to raw posterior mean [B,16,T/4,H/8,W/8]."""
    if sample.ndim != 5 or sample.shape[0] != 1 or sample.shape[1] != 3:
        raise ValueError(f"TensorRT encoder expects [1,3,T,H,W], got {tuple(sample.shape)}")
    _, _, frames, height, width = sample.shape
    if height % 8 or width % 8:
        raise ValueError("TensorRT encoder input dimensions must be divisible by 8")
    _, engine, _cached_context, input_name, output_name, stream = _engine(int(frames))
    # Keep the TensorRT engine cached for speed, but never reuse its execution
    # context across temporal batches. TensorRT-RTX can retain batch state.
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create a per-batch encoder context")
    source = sample.to(device="cuda", dtype=torch.float16).contiguous()
    tile, overlap = 512, 96
    ys, xs = _positions(height, tile, overlap), _positions(width, tile, overlap)
    padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
    source = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
    latent_frames = frames // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    raw_h, raw_w = padded_h // 8, padded_w // 8
    result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    overlap_latent = overlap // 8
    with _ENCODE_LOCK, torch.cuda.stream(stream):
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_output = torch.empty((1, 32, latent_frames, 64, 64), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE encoder failed at tile y={y}, x={x}")
                stream.synchronize()
                ly, lx = y // 8, x // 8
                wy = _feather(64, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(64, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, 64, 64)
                result[:, :, :, ly:ly + 64, lx:lx + 64] += tile_output.float() * window
                weights[:, :, :, ly:ly + 64, lx:lx + 64] += window
    encoded = (result / weights.clamp_min(1e-6))[:, :16, :, :latent_h, :latent_w].to(sample.dtype)
    del context, _cached_context
    return encoded

def release() -> None:
    """Release encoder contexts before the much larger DiT phase is loaded."""
    with _ENCODE_LOCK:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ENGINES.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

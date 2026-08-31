"""Portable, observable ONNX export helpers for TensorRT engine preparation."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch


def _register_custom_symbolics():
    try:
        from torch.onnx import register_custom_op_symbolic
        from torch.onnx.symbolic_helper import parse_args

        def symbolic_cudnn_convolution(g, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32):
            return g.op(
                "Conv",
                input,
                weight,
                pads_i=[p for p in padding for _ in (0, 1)] if isinstance(padding, (list, tuple)) else None,
                strides_i=stride if isinstance(stride, (list, tuple)) else None,
                dilations_i=dilation if isinstance(dilation, (list, tuple)) else None,
                group_i=groups if isinstance(groups, int) else 1,
            )

        register_custom_op_symbolic("::cudnn_convolution", symbolic_cudnn_convolution, 20)
        register_custom_op_symbolic("aten::cudnn_convolution", symbolic_cudnn_convolution, 20)
    except Exception:
        pass

_register_custom_symbolics()


def _math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


def _portable_export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, *, legacy: bool) -> None:
    is_encoder = args[0].ndim == 5 and args[0].shape[1] == 3
    with torch.inference_mode(), _math_attention_context():
        torch.onnx.export(
            module,
            args,
            str(output),
            input_names=["video"] if is_encoder else ["latent"],
            output_names=["latent_raw"] if is_encoder else ["sample"],
            opset_version=20,
            dynamo=not legacy,
            optimize=False,
            do_constant_folding=False,
        )


def export_portable_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: Path,
    *,
    legacy: bool,
) -> None:
    """Export on CUDA with portable operators, falling back to CPU only on OOM."""
    print(f"Exporting ONNX (legacy tracer, portable convs) -> {output}", flush=True)
    started = time.perf_counter()
    try:
        _portable_export(module, args, output, legacy=legacy)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("CUDA ONNX export ran out of memory; falling back to CPU export. This may take a while.", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _portable_export(module.cpu(), tuple(value.cpu() for value in args), output, legacy=legacy)
    print(f"ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)

"""Export a fixed-shape SeedVR2 VAE decoder graph for TensorRT experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure torch/lib is at the very front of PATH and DLL search to prevent version mismatch
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "seedvr2"
sys.path.insert(0, str(VENDOR))

from src.core.generation_utils import prepare_runner, setup_generation_context  # noqa: E402
from src.core.model_loader import materialize_model  # noqa: E402
from src.utils.debug import Debug  # noqa: E402
from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE  # noqa: E402
from src.models.video_vae_v3.modules.types import MemoryState  # noqa: E402
from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d  # noqa: E402
from onnx_export_utils import export_portable_onnx  # noqa: E402


class Decoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, memory_state=MemoryState.DISABLED)


def configure_fixed_profile(vae: torch.nn.Module) -> None:
    """Disable adaptive VAE slicing for a fixed-shape export profile."""
    if hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(float("inf"))
            module.set_memory_device(None)
        if hasattr(module, "slicing"):
            module.slicing = False


def main() -> None:
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--legacy-export", action="store_true",
                        help="Use the legacy fixed-shape ONNX tracer")
    parser.add_argument("--output", type=Path, default=ROOT / "tensorrt_backend" / "artifacts" / "vae_decoder.onnx")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VAE export")
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by 8")

    device = torch.device("cuda")
    debug = Debug(enabled=True)
    ctx = setup_generation_context(
        dit_device=device,
        vae_device=device,
        tensor_offload_device=None,
        debug=debug,
    )
    ctx["compute_dtype"] = torch.float16
    runner, _ = prepare_runner(
        dit_model=DEFAULT_DIT,
        vae_model=DEFAULT_VAE,
        model_dir=str(ROOT / "models" / "SEEDVR2"),
        debug=debug,
        ctx=ctx,
        block_swap_config={
            "blocks_to_swap": 0,
            "swap_io_components": False,
            "offload_device": None,
        },
        attention_mode="sdpa",
    )
    materialize_model(runner, "vae", device, runner.config, debug)
    vae = runner.vae.eval()
    configure_fixed_profile(vae)
    decoder = Decoder(vae.decoder).eval()

    latent = torch.randn(
        args.batch_size, 16, args.latent_frames, args.height // 8, args.width // 8,
        device=device, dtype=torch.float16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        export_portable_onnx(decoder, (latent,), args.output, legacy=args.legacy_export)
    print(f"Exported: {args.output}")
    print(f"Latent shape: {tuple(latent.shape)}")


if __name__ == "__main__":
    main()

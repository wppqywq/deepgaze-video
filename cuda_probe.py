"""Minimal CUDA BF16 autograd check, not an 8B model capacity measurement."""
import json
import os
from pathlib import Path


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA checks require a Slurm allocation")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one allocated CUDA device")
    if torch.version.cuda != "12.8" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected CUDA 12.8 and BF16 support")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = (x @ x.T).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("Non-finite or missing BF16 backward result")
    if not torch.count_nonzero(x.grad):
        raise RuntimeError("BF16 backward produced no nonzero gradients")
    result = {"job_id": os.environ["SLURM_JOB_ID"], "torch": torch.__version__,
              "wheel_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
              "reported_device_bytes": torch.cuda.get_device_properties(0).total_memory,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "mps_memory_limit": os.environ.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"),
              "loss": float(loss.detach()), "status": "CUDA BF16 forward/backward passed",
              "limitation": "Not a model test; physical GPU capacity is not an MPS allocation limit"}
    destination = Path(os.environ["ROUTEB_CODE"]) / "evidence" / ("cuda-" + result["job_id"] + ".json")
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

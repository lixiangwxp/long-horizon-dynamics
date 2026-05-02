import os
import torch


def mps_is_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device(accelerator="auto", gpu_id=0, num_devices=1):
    requested = accelerator.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("accelerator must be one of [auto, cuda, mps, cpu]")

    if requested in {"auto", "cuda"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if requested == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif mps_is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = requested

    if resolved == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        devices = num_devices if num_devices > 0 else 1
        return {
            "resolved": "cuda",
            "device": torch.device("cuda:0"),
            "lightning_accelerator": "gpu",
            "devices": devices,
            "pin_memory": True,
        }

    if resolved == "mps":
        if not mps_is_available():
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is False."
            )
        return {
            "resolved": "mps",
            "device": torch.device("mps"),
            "lightning_accelerator": "mps",
            "devices": 1,
            "pin_memory": False,
        }

    return {
        "resolved": "cpu",
        "device": torch.device("cpu"),
        "lightning_accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
    }

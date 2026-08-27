from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from ..calibration.windows import verify_sealed_corpus
from ..core.artifacts import bind_files, prepare_empty_destination, require_execute, sha256_file, write_json
from ..evaluation.kld_window import SCHEMA as KLD_WINDOW_SCHEMA, verify_kld_window


def _release_version(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"(\d+(?:\.\d+)+)(?:[+.-].*)?", value)
    if match is None:
        raise ValueError(f"unsupported release version {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def _resolve_model_class(transformers_module: Any, config: dict) -> tuple[type, dict]:
    """Resolve the exact checkpoint architecture without a stale auto mapping.

    GLM-5.3 was released with ``Glm5NextForConditionalGeneration`` and a
    checkpoint-declared Transformers version. The architecture is new enough
    that an older AutoModel mapping can silently be absent even when generic
    text-generation examples exist, so this path is deliberately exact.
    """

    architectures = config.get("architectures")
    model_type = config.get("model_type")
    if model_type == "glm5_next":
        if architectures != ["Glm5NextForConditionalGeneration"]:
            raise ValueError("released glm5_next checkpoint architecture is not exact")
        required_version = config.get("transformers_version")
        installed_version = str(getattr(transformers_module, "__version__", ""))
        if (
            not isinstance(required_version, str)
            or not installed_version
            or _release_version(installed_version) < _release_version(required_version)
        ):
            raise RuntimeError(
                f"glm5_next capture requires Transformers >= {required_version}; installed {installed_version or 'unknown'}"
            )
        model_class = getattr(transformers_module, architectures[0], None)
        if model_class is None:
            raise RuntimeError(
                "the checkpoint-declared Glm5NextForConditionalGeneration class is absent from this Transformers build"
            )
        return model_class, {
            "resolution": "checkpoint_exact_architecture",
            "model_type": model_type,
            "architecture": architectures[0],
            "required_transformers": required_version,
            "installed_transformers": installed_version,
        }

    model_class = getattr(transformers_module, "AutoModelForCausalLM", None)
    if model_class is None:
        raise RuntimeError("Transformers build has no AutoModelForCausalLM")
    return model_class, {
        "resolution": "auto_model_for_causal_lm",
        "model_type": model_type,
        "architecture": architectures[0] if isinstance(architectures, list) and architectures else None,
        "required_transformers": config.get("transformers_version"),
    }


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(values) or None


def capture_logits(
    model_path: str,
    sealed_corpus: str | Path,
    role: str,
    output_dir: str | Path,
    execute: bool,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    model_revision: str = "",
) -> dict:
    require_execute(execute, "load the model and capture logits")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("capture requires the expected 40-hex model revision")
    try:
        import torch
        from safetensors.torch import save_file
        import transformers
    except Exception as error:  # pragma: no cover
        raise RuntimeError("install quant-pipeline[hf] for capture") from error
    sealed = json.loads(Path(sealed_corpus).read_text())
    if sealed.get("schema") == KLD_WINDOW_SCHEMA:
        if role != "kld":
            raise ValueError("a KLD window must be captured with role='kld'")
        verify_kld_window(sealed, Path(sealed_corpus).resolve().parent)
        windows = [{"token_ids": sealed["token_ids"], "token_sha256": sealed["token_sha256"]}]
    else:
        if role == "kld":
            raise ValueError("role='kld' requires a sealed KLD-window artifact")
        verify_sealed_corpus(sealed)
        windows = sealed["windows"][role]
    destination = prepare_empty_destination(output_dir)
    torch_dtype = getattr(torch, dtype)
    local_model = Path(model_path)
    if not local_model.is_dir():
        raise ValueError("capture requires a local immutable checkpoint directory")
    config_path = local_model / "config.json"
    config = json.loads(config_path.read_text())
    index_path = local_model / "model.safetensors.index.json"
    identity_files = [config_path]
    if index_path.exists():
        index = json.loads(index_path.read_text())
        identity_files.append(index_path)
        identity_files.extend(local_model / name for name in sorted(set(index["weight_map"].values())))
    elif (local_model / "model.safetensors").exists():
        identity_files.append(local_model / "model.safetensors")
    else:
        raise FileNotFoundError("checkpoint has no safetensors index or model.safetensors")
    model_identity = {"expected_revision": model_revision, "files": bind_files(identity_files)}
    model_class, model_resolution = _resolve_model_class(transformers, config)
    model = model_class.from_pretrained(
        local_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "output_router_logits"):
        model.config.text_config.output_router_logits = True
    records = []
    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for index, window in enumerate(windows):
            input_ids = torch.tensor([window["token_ids"]], dtype=torch.long, device=input_device)
            call_options = {"input_ids": input_ids, "use_cache": False, "return_dict": True}
            if hasattr(model.config, "output_router_logits"):
                call_options["output_router_logits"] = True
            output = model(**call_options)
            tensors = {"logits": output.logits[0, :-1].to(torch.float32).cpu()}
            for layer, router in enumerate(getattr(output, "router_logits", None) or ()):
                tensors[f"router_logits.layer_{layer:03d}"] = router.detach().to(torch.float32).cpu()
            path = destination / f"window-{index:04d}.safetensors"
            save_file(tensors, path, metadata={"token_sha256": window["token_sha256"], "role": role})
            records.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "token_sha256": window["token_sha256"]})
    receipt = {
        "schema": "quant-pipeline.hf-logit-capture.v1",
        "model_path": str(Path(model_path).resolve()) if Path(model_path).exists() else model_path,
        "model_identity": model_identity,
        "model_resolution": model_resolution,
        "sealed_corpus_sha256": sha256_file(sealed_corpus),
        "role": role,
        "dtype": dtype,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "image_digest": os.environ.get("QUANT_PIPELINE_IMAGE_DIGEST"),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "transformers": __import__("transformers").__version__,
            "safetensors": __import__("safetensors").__version__,
            "device_map": device_map,
            "model_class": type(model).__name__,
            "cuda_devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "records": records,
    }
    write_json(destination / "capture-receipt.json", receipt)
    return receipt

"""Shared JevBench/R047 singleton forward and native option probabilities."""

import hashlib
import math
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_native_probs(probs, labels):
    if not isinstance(probs, dict) or list(probs) != labels:
        raise ValueError("probability keys/order must exactly match task labels")
    clean = {}
    for label, value in probs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probability for {label!r} is not numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"probability for {label!r} is not finite in [0,1]")
        clean[label] = value
    if not math.isclose(sum(clean.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("probabilities do not sum to one")
    return clean


def probabilities_from_option_scores(labels, scores):
    """Stable softmax with JevBench's final residual and Python float precision."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and aligned")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("option scores must be finite")
    peak = max(float(score) for score in scores)
    weights = [math.exp(float(score) - peak) for score in scores]
    total = sum(weights)
    values = [weight / total for weight in weights]
    if len(values) > 1:
        values[-1] = 1.0 - sum(values[:-1])
    return validate_native_probs(dict(zip(labels, values)), labels)


class ForwardObservations:
    """Observe canonical forwards, profiling the first CUDA call when requested."""

    def __init__(self, require_sdpa=False):
        self.batch_histogram = Counter()
        self.logits_dtypes = set()
        self.require_sdpa = require_sdpa
        self.sdpa_observed = False

    def report(self):
        return {str(key): count for key, count in sorted(self.batch_histogram.items())}


def singleton_option_probabilities(model, input_ids, groups, labels, observations=None):
    """The exact Jev-style forward: [1,L], no mask, last logit, float32 reduction."""
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("singleton scorer requires input_ids shape [1,L], B=1")
    if not labels or len(groups) < len(labels):
        raise ValueError("option token groups must cover every label")

    def forward():
        return model(input_ids=input_ids, use_cache=False, logits_to_keep=1)

    if observations is not None:
        observations.batch_histogram[int(input_ids.shape[0])] += 1
    if observations is not None and observations.require_sdpa and not observations.sdpa_observed:
        if input_ids.device.type != "cuda":
            raise ValueError("SDPA profiler canary requires actual CUDA inputs")
        # CPU dispatcher events include aten::scaled_dot_product_attention for all
        # CUDA SDP backends, without relying on CUPTI or backend-specific kernel names.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            output = forward()
        observations.sdpa_observed = any(
            "scaled_dot_product_attention" in event.key for event in profiler.key_averages())
        if not observations.sdpa_observed:
            raise ValueError("SDPA profiler canary did not observe scaled_dot_product_attention")
    else:
        output = forward()
    if observations is not None:
        observations.logits_dtypes.add(str(output.logits.dtype).removeprefix("torch."))
    logits = output.logits[0, -1].float()
    scores = [float(torch.logsumexp(logits[group.to(input_ids.device)], dim=0).item())
              for group in groups[:len(labels)]]
    return probabilities_from_option_scores(labels, scores)


RUNTIME_FIELDS = {
    "torch_version", "transformers_version", "peft_version", "cuda_runtime_version",
    "cudnn_version", "gpu_name", "gpu_capability", "driver_version",
    "deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic",
    "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
    "sdp_flash_enabled", "sdp_math_enabled", "sdp_mem_efficient_enabled",
    "sdp_cudnn_enabled", "base_parameter_dtype_counts", "adapter_parameter_dtype_counts",
    "observed_logits_dtype", "text_attention_implementations", "sdpa_profiler_observed",
}
RUNTIME_BOOLS = {
    "deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic",
    "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
    "sdp_flash_enabled", "sdp_math_enabled", "sdp_mem_efficient_enabled",
    "sdp_cudnn_enabled", "sdpa_profiler_observed",
}


def validate_runtime_fingerprint(value):
    """Validate a CUDA causal receipt without importing torch or probing hardware."""
    if not isinstance(value, dict) or set(value) != RUNTIME_FIELDS:
        raise ValueError("runtime fingerprint schema is invalid")
    strings = {"torch_version", "transformers_version", "peft_version",
               "cuda_runtime_version", "gpu_name", "observed_logits_dtype"}
    if any(not isinstance(value[key], str) or not value[key].strip() for key in strings):
        raise ValueError("runtime fingerprint strings are invalid")
    if any(type(value[key]) is not bool for key in RUNTIME_BOOLS):
        raise ValueError("runtime fingerprint flags must be booleans")
    if (type(value["cudnn_version"]) is not int or value["cudnn_version"] <= 0
            or (value["driver_version"] is not None
                and (not isinstance(value["driver_version"], str) or not value["driver_version"].strip()))
            or not isinstance(value["gpu_capability"], list)
            or len(value["gpu_capability"]) != 2
            or any(type(item) is not int or item < 0 for item in value["gpu_capability"])):
        raise ValueError("runtime fingerprint hardware types are invalid")
    for key in ("base_parameter_dtype_counts", "adapter_parameter_dtype_counts"):
        counts = value[key]
        if (not isinstance(counts, dict) or not counts
                or any(dtype not in {"bfloat16", "float16", "float32", "float64"}
                       or type(count) is not int or count <= 0 for dtype, count in counts.items())):
            raise ValueError("runtime fingerprint parameter dtype counts are invalid")
    if (set(value["base_parameter_dtype_counts"]) != {"bfloat16"}
            or value["observed_logits_dtype"] not in {"bfloat16", "float32"}
            or value["text_attention_implementations"] != ["sdpa"]
            or value["sdpa_profiler_observed"] is not True):
        raise ValueError("runtime fingerprint violates bfloat16/SDPA contract")
    return value


def collect_runtime_fingerprint(model, observations, device):
    import torch

    base, adapter = Counter(), Counter()
    for name, parameter in model.named_parameters():
        counts = adapter if "lora_" in name else base
        counts[str(parameter.dtype).removeprefix("torch.")] += parameter.numel()
    attention = set()
    text_config = model.config.get_text_config()
    attention.add(getattr(text_config, "_attn_implementation", None))
    for name, module in model.named_modules():
        if "self_attn" in name and not any(part in name for part in ("vision", "audio")):
            implementation = getattr(getattr(module, "config", None), "_attn_implementation", None)
            if implementation is not None:
                attention.add(implementation)
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5)
        versions = set(result.stdout.splitlines())
        if len(versions) == 1:
            driver = versions.pop().strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    if len(observations.logits_dtypes) != 1:
        raise ValueError("runtime fingerprint requires one observed logits dtype")
    value = {
        "torch_version": str(torch.__version__), "transformers_version": version("transformers"),
        "peft_version": version("peft"), "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)), "driver_version": driver,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "sdp_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdp_math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "sdp_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdp_cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "base_parameter_dtype_counts": dict(sorted(base.items())),
        "adapter_parameter_dtype_counts": dict(sorted(adapter.items())),
        "observed_logits_dtype": next(iter(observations.logits_dtypes)),
        "text_attention_implementations": sorted(attention, key=str),
        "sdpa_profiler_observed": observations.sdpa_observed,
    }
    return validate_runtime_fingerprint(value)

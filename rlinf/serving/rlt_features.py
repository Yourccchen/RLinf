"""Pack frozen Soma conditions and train the existing RLT token compressor.

This entrypoint needs PyTorch, not RLinf's distributed model registry. Artifacts
contain tensors and plain metadata and are loaded with weights_only=True.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

FORMAT = "soma-rlt-features-v1"


@lru_cache(maxsize=1)
def _transformer_class():
    # Load the existing standalone torch module, without importing models/__init__
    # (the distributed registry imports Ray). No duplicate implementation/weights.
    path = Path(__file__).parents[1] / "models/embodiment/modules/rlt_token_transformer.py"
    spec = importlib.util.spec_from_file_location("_rlinf_rlt_token_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RLTTokenTransformer


def pack_features(features: Mapping[str, Any], spec: Mapping[str, Any]) -> torch.Tensor:
    """Return [N,C] float32 tokens in view, time, row, column order."""
    source = spec.get("source")
    if source not in features:
        raise ValueError(f"Missing feature source {source!r}")
    value = features[source]
    if source == "video_condition":
        temporal = spec.get("temporal", "last")
        if temporal not in ("last", "all"):
            raise ValueError("temporal must be last or all")
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("video_condition must contain ordered view tensors")
        views = []
        for latent in value:
            if latent.ndim != 4 or min(latent.shape) < 1:
                raise ValueError("video_condition views must be non-empty [C,T,H,W]")
            latent = latent.detach().float()
            if temporal == "last":
                latent = latent[:, -1:]
            pool = spec.get("spatial_pool")
            if pool is not None:
                if len(pool) != 2 or any(int(n) != n or n < 1 for n in pool):
                    raise ValueError("spatial_pool must be two positive integers")
                if any(n > limit for n, limit in zip(pool, latent.shape[-2:])):
                    raise ValueError("spatial_pool must not enlarge spatial dimensions")
                latent = F.adaptive_avg_pool2d(latent, tuple(pool))
            views.append(latent.permute(1, 2, 3, 0).reshape(-1, latent.shape[0]))
        if len({view.shape[1] for view in views}) != 1:
            raise ValueError("All views must have the same channel dimension")
        tokens = torch.cat(views)
    elif source == "text_condition":
        # Soma's runtime returns one prompt embedding [L,D].
        tokens = value.detach().float()
    else:
        raise ValueError(f"Unsupported feature source {source!r}")
    _validate_tokens(tokens)
    return tokens.contiguous()


def _validate_tokens(tokens: torch.Tensor) -> None:
    if tokens.ndim != 2 or min(tokens.shape) < 1 or not torch.isfinite(tokens).all():
        raise ValueError("tokens must be finite, non-empty [N,C]")


def save_feature_sample(
    path: str | Path, tokens: torch.Tensor, feature_spec: Mapping[str, Any],
    feature_identity: str, *, metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write one immutable sample; identity names weights, config and view order."""
    _validate_tokens(tokens)
    if not feature_identity.strip():
        raise ValueError("feature_identity must name a frozen base-model version")
    artifact = dict(format=FORMAT, kind="sample", tokens=tokens.detach().float().cpu(),
                    feature_spec=dict(feature_spec), feature_identity=feature_identity,
                    metadata=dict(metadata or {}))
    with Path(path).open("xb") as stream:
        torch.save(artifact, stream)


def _load(path: str | Path, kind: str) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != FORMAT or artifact.get("kind") != kind:
        raise ValueError(f"Not a {FORMAT} {kind}: {path}")
    return artifact


def train_encoder(
    paths: Sequence[str | Path], output: str | Path, model_config: Mapping[str, Any],
    *, steps: int, batch_size: int = 8, lr: float = 1e-4,
    device: str = "cpu", seed: int = 0,
) -> list[float]:
    """Train reconstruction on detached samples and save a frozen encoder artifact.

    This small offline entrypoint loads the corpus into CPU memory. No backbone,
    robot, Ray cluster, actor or critic is created. Start with a bounded corpus.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    if not paths or steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Require samples, positive steps, batch_size and lr")
    samples = [_load(path, "sample") for path in paths]
    first = samples[0]
    for sample in samples:
        if sample["feature_identity"] != first["feature_identity"]:
            raise ValueError("Mixed feature identity in training samples")
        if sample["feature_spec"] != first["feature_spec"]:
            raise ValueError("Mixed feature_spec in training samples")
        _validate_tokens(sample["tokens"])
    config = dict(model_config)
    input_dim = int(config["input_dim"])
    max_length = int(config["prefix_seq_len"])
    for sample in samples:
        n, c = sample["tokens"].shape
        if c != input_dim or n > max_length:
            raise ValueError(f"Sample shape {(n, c)} exceeds model config {config}")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = _transformer_class()(**config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        indices = torch.randint(len(samples), (batch_size,), generator=generator)
        batch = [samples[int(i)]["tokens"] for i in indices]
        length = max(t.shape[0] for t in batch)
        tokens = torch.zeros(batch_size, length, input_dim, device=device)
        mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        for i, value in enumerate(batch):
            tokens[i, :len(value)] = value.to(device)
            mask[i, :len(value)] = True
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(tokens, mask)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite reconstruction loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    artifact = dict(format=FORMAT, kind="encoder", model_config=config,
                    feature_spec=first["feature_spec"], feature_identity=first["feature_identity"],
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    steps=steps, seed=seed, losses=losses)
    with Path(output).open("xb") as stream:
        torch.save(artifact, stream)
    return losses


def load_encoder(path: str | Path, *, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Restore the trained compressor and the exact input feature contract."""
    artifact = _load(path, "encoder")
    # Loading a frozen feature model must not perturb action sampling's RNG.
    with torch.random.fork_rng(devices=[]):
        model = _transformer_class()(**artifact["model_config"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Encoder contains non-finite parameters")
    model = model.to(device).eval().requires_grad_(False)
    return model, {k: v for k, v in artifact.items() if k != "state_dict"}

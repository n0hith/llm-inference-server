"""Load a HuggingFace Llama checkpoint (safetensors) into our `LlamaModel`.

We read the safetensors files directly instead of going through
`transformers.AutoModelForCausalLM`, so the engine never instantiates HF's model.
The steps:

1. Resolve the model: a local directory, or a hub id downloaded with huggingface_hub
   (only config/tokenizer/safetensors files; no PyTorch .bin duplicates).
2. Build our model on the **meta device**: modules get shapes but no memory and no random
   init. Otherwise we'd allocate and randomly initialize ~1B parameters only to overwrite them,
   and briefly hold two copies of the weights (which doesn't fit for fp32 on a 6 GB GPU).
3. Stream each checkpoint tensor straight to the target device, cast to the target dtype,
   rename it to our parameter name, and attach it with `load_state_dict(assign=True)`.
4. Strict checks: every parameter of our model must be loaded and every checkpoint tensor
   must be used. A silently missing weight would leave a meta tensor or garbage behind.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from engine.config import ModelConfig
from engine.model import LlamaModel

ALLOW_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model"]


def resolve_model_dir(name_or_path: str | Path) -> Path:
    """Local directory as-is; otherwise a hub repo id (downloaded once, then cached)."""
    path = Path(name_or_path)
    if path.is_dir():
        return path
    return Path(snapshot_download(str(name_or_path), allow_patterns=ALLOW_PATTERNS))


def hf_to_ours(name: str) -> str | None:
    """Map a HF checkpoint tensor name to our parameter name (None = intentionally skipped).

    Our modules mirror HF's naming, so this is just dropping the `model.` prefix:
        model.layers.3.self_attn.q_proj.weight -> layers.3.self_attn.q_proj.weight
        model.embed_tokens.weight              -> embed_tokens.weight
        lm_head.weight                         -> lm_head.weight
    """
    if name.endswith("rotary_emb.inv_freq"):  # stored by some old checkpoints; we recompute it
        return None
    return name.removeprefix("model.")


def load_state_dict(
    model_dir: Path, dtype: torch.dtype, device: torch.device | str
) -> dict[str, torch.Tensor]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        # device=... makes safetensors copy each tensor straight to the GPU; the cast to `dtype`
        # then happens there, so the transient extra memory is one tensor at a time.
        with safe_open(file, framework="pt", device=str(device)) as f:
            for key in f.keys():  # noqa: SIM118 (safe_open isn't a dict)
                ours = hf_to_ours(key)
                if ours is not None:
                    state[ours] = f.get_tensor(key).to(dtype)
    return state


def load_model(
    name_or_path: str | Path,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    attn_impl: str = "sdpa",
) -> LlamaModel:
    """Build our LlamaModel with weights from a HF Llama checkpoint."""
    model_dir = resolve_model_dir(name_or_path)
    cfg = ModelConfig.from_pretrained(model_dir)

    with torch.device("meta"):
        model = LlamaModel(cfg, attn_impl=attn_impl)

    state = load_state_dict(model_dir, dtype, device)
    if cfg.tie_word_embeddings:
        # Tied models (e.g. Llama-3.2) reuse the input embedding matrix as the output projection
        # and don't store lm_head in the checkpoint. Same tensor -> no extra memory.
        if "lm_head.weight" in state:
            raise ValueError("config says tie_word_embeddings but checkpoint has lm_head.weight")
        state["lm_head.weight"] = state["embed_tokens.weight"]

    expected = set(model.state_dict().keys())
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"weight mismatch: missing={sorted(missing)} extra={sorted(unexpected)}")
    for name, param in model.state_dict().items():
        got, want = tuple(state[name].shape), tuple(param.shape)
        if got != want:
            raise ValueError(f"{name}: checkpoint shape {got} != model shape {want}")

    model.load_state_dict(state, strict=True, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight  # make the tie explicit (one Parameter)

    # Parameters are already on `device`; this moves the non-persistent RoPE buffer (built on CPU).
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

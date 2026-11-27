"""Score pre-tokenization data with BlockScan's own model and scorer.

This file is **not** imported by txbench: it runs as a script under the pinned
dependency set the authors declare (`requirements.txt`: torch 2.1.0,
transformers 4.32.0), in a separate interpreter, because that set and this
project's own cannot share one environment (numpy 1 vs numpy 2). It imports
`src_eth/detection.py` from the pinned checkout, which loads the authors'
tokenizer and checkpoint at module scope, and then calls their
`compute_abnormal_score` unchanged.

Three shims stand between that code and a machine with no NVIDIA GPU. Each one
is narrow, and each is recorded in the output so no reader has to trust this
docstring:

  1. `replace_roberta_attn()` probes `torch.cuda.get_device_capability()`
     unconditionally - only to switch flash attention off on pre-A100 cards.
     The caller already passes `_use_flash_attn=False`, so reporting (0, 0)
     changes no behaviour and lets the call return at all.
  2. the scorer moves the model and every tensor with `.cuda()`. With no CUDA
     device those become identity, and the whole forward pass runs on the CPU.
     The arithmetic is the same; only the device differs.
  3. `foundation_model.py` imports `xformers` at module scope but calls it only
     when flash attention is on. The stub installed here *raises* on any
     attribute access, so if the inference path ever reached xformers the run
     would fail loudly instead of substituting something quietly.

Nothing in the authors' files is edited. The masking policy, the top-s rule and
the abnormal-token count are theirs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

SHIMS = [
    {"target": "torch.cuda.get_device_capability",
     "reason": "replace_roberta_attn() probes it unconditionally; flash attention is already "
               "off, so (0, 0) changes no behaviour and lets the call return without a GPU"},
    {"target": "torch.nn.Module.cuda / torch.Tensor.cuda",
     "reason": "the pinned scorer moves the model and tensors to CUDA; with no CUDA device "
               "these become identity and the same arithmetic runs on the CPU"},
    {"target": "sys.modules['xformers']",
     "reason": "imported at module scope by foundation_model.py but only called when flash "
               "attention is on; the stub raises on any attribute access, so the inference "
               "path cannot use it silently"},
]


def install_shims() -> dict[str, object]:
    import torch

    torch.cuda.get_device_capability = lambda *a, **k: (0, 0)
    torch.cuda.is_available = lambda: False
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self

    class _Forbidden:
        def __getattr__(self, name):
            raise RuntimeError(
                f"xformers.ops.{name} was called: the inference path must not use xformers")

    module = types.ModuleType("xformers")
    module.ops = _Forbidden()
    sys.modules["xformers"] = module
    sys.modules["xformers.ops"] = module.ops
    return {"torch_version": torch.__version__, "device": "cpu", "shims": SHIMS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="the pinned BlockScan checkout")
    parser.add_argument("--data-dir", required=True,
                        help="a folder holding preprocessed_tx.txt and tx_path.txt")
    parser.add_argument("--out", required=True, help="where to write the scores as JSON")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="the authors' own default: sequences above it are truncated")
    parser.add_argument("--mask-percent", type=int, default=15, help="the authors' g")
    parser.add_argument("--top-s", type=int, default=3, help="the authors' s")
    parser.add_argument("--numpy-seed", type=int, default=42,
                        help="detection.py seeds numpy at import; stated here for the record")
    parser.add_argument("--decode-out", default=None,
                        help="also write back the decoded text of each truncated input, which "
                             "is what the model actually saw (used by the smoke test)")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    provenance = install_shims()
    sys.path.insert(0, str(repo))
    # detection.py reads '../config/tokenizer' and '../model/rope_roberta', relative to the
    # directory it is run from, so it is run from the directory the authors run it from.
    os.chdir(repo / "src_eth")
    import numpy as np
    import transformers

    np.random.seed(args.numpy_seed)
    # Read the seed back out of NumPy's own state, inside the process that will
    # consume the stream. For MT19937 the first word of the state key is the
    # seed it was initialised with, so this is evidence the call took effect
    # rather than a copy of what was requested. detection.py also calls
    # np.random.seed(42) at import, so the value after the import is what the
    # scorer actually ran with - which is why it is captured twice.
    seed_before_import = int(np.random.get_state()[1][0])
    import src_eth.detection as detection      # loads the tokenizer and the checkpoint
    np.random.seed(args.numpy_seed)             # detection.py reseeds at import; restore ours
    seed_after_import = int(np.random.get_state()[1][0])

    attention = transformers.models.roberta.modeling_roberta.RobertaSelfAttention
    tokenizer, model = detection.tokenizer, detection.fm_model

    # Token lengths *before* the authors' truncation, measured with their tokenizer.
    lines = (data_dir / "preprocessed_tx.txt").read_text(encoding="utf-8").splitlines()
    paths = (data_dir / "tx_path.txt").read_text(encoding="utf-8").splitlines()
    if len(lines) != len(paths):
        raise SystemExit(f"{data_dir}: {len(lines)} trace line(s) but {len(paths)} path line(s)")
    lengths = []
    for line in lines:
        ids = tokenizer(line, return_tensors=None)["input_ids"]
        lengths.append(len(ids))

    # The authors' scorer, called as written.
    scores, abnormal = detection.compute_abnormal_score(
        str(data_dir), model=model, tokenizer=tokenizer, random_masking=True,
        g=args.mask_percent, s=args.top_s, seq_len=args.seq_len)

    by_path = {str(name).strip(): int(value) for name, value in scores}
    rows = []
    for path, line, length in zip(paths, lines, lengths):
        key = str(path).strip()
        kept = min(length, args.seq_len)
        # the scorer pads to a multiple of 8 and masks int(len * g / 100) positions
        padded = kept + ((8 - kept % 8) % 8)
        rows.append({
            "path": key,
            "score": by_path.get(key),
            "tokens_before_truncation": length,
            "tokens_scored": kept,
            "tokens_padded": padded,
            "masked_tokens": int(padded * args.mask_percent / 100),
            "truncated": length > args.seq_len,
            "trace_token_words": len(line.split()),
        })
    missing = [r["path"] for r in rows if r["score"] is None]

    if args.decode_out:
        # Exactly the ids the scorer fed the model, decoded back to text: the only
        # honest way to answer "did this frame reach the model input?".
        decoded = {}
        for path, line in zip(paths, lines):
            ids = tokenizer(line, truncation=True, max_length=args.seq_len,
                            return_tensors=None)["input_ids"]
            decoded[str(path).strip()] = tokenizer.decode(ids)
        Path(args.decode_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.decode_out).write_text(json.dumps(decoded, indent=2) + "\n", encoding="utf-8")

    payload = {
        "data_dir": str(data_dir),
        "repo": str(repo),
        "entry_points": ["src_eth/detection.py::compute_abnormal_score"],
        "model": {
            "class": type(model).__name__,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "attention_module": attention.__module__,
            "config": {k: v for k, v in model.config.to_dict().items()
                       if k in ("model_type", "num_hidden_layers", "hidden_size", "vocab_size",
                                "max_position_embeddings", "num_attention_heads",
                                "transformers_version")},
        },
        "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": int(tokenizer.vocab_size)},
        "scorer_settings": {"random_masking": True, "mask_percent": args.mask_percent,
                            "top_s": args.top_s, "seq_len": args.seq_len,
                            "numpy_seed": args.numpy_seed,
                            "numpy_seed_effective": seed_after_import,
                            "numpy_seed_before_detection_import": seed_before_import,
                            "numpy_seed_read_from": ("numpy.random.get_state() inside the "
                                                     "scoring process, after detection.py's "
                                                     "own import-time seeding"),
                            "score_meaning": "the number of masked tokens whose true id is not "
                                             "among the model's top-s predictions; higher means "
                                             "more anomalous"},
        "environment": {"python": sys.version.split()[0],
                        "transformers": transformers.__version__, **provenance},
        "rows": rows,
        "rows_without_score": missing,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"scored {len(rows) - len(missing)}/{len(rows)} trace(s) -> {out_path}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

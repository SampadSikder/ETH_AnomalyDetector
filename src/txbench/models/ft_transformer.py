"""FT-Transformer over this project's own feature views.

The architecture is the paper's (Gorishniy et al., *Revisiting Deep Learning
Models for Tabular Data*, arXiv:2106.11959): a feature tokenizer turns each
numeric feature into its own token, a `[CLS]` token is appended, a stack of
pre-norm Transformer blocks runs over the tokens, and the prediction comes from
the `[CLS]` token alone.

What it is fed comes from `txbench.features`, unchanged: one row per
transaction, one column per selector-candidate or log-topic token, plus the
three out-of-vocabulary mass columns. The non-feature columns the view carries
- `tx_key`, `partition`, `label`, `usable`, `unusable_reason` - are excluded
here by name, and nothing else about a transaction (its incident, protocol,
attack class, label tier, source file or collection path) is in the matrix at
all; `features.NON_FEATURE_COLUMNS` is the list the pipeline enforces.

Everything learned is learned on the training partition only:

  * the vocabulary was fitted on training rows by `bench build-features`;
  * the document-frequency cutoff applied here reads that training-fitted
    frequency table, never the validation or test rows;
  * `log1p` then standardisation uses means and deviations computed on training
    rows, and the fitted values are written into the run artifact;
  * the epoch is chosen on validation average precision, and the test
    partition is not scored until the model and the threshold are frozen.
"""
from __future__ import annotations

from .defiguard_gnn import effective_seed

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..features import OOV_COLUMNS, VIEWS, view_dir
from ..logs import get_logger

log = get_logger("ft_transformer")

MODEL_NAME = "ft_transformer"
MODEL_VERSION = "project-impl-fttransformer"

#: Columns in a feature matrix that are not features.
NON_FEATURE = ("tx_key", "partition", "label", "usable", "unusable_reason")


@dataclass
class ViewData:
    view: str
    columns: list[str]
    partitions: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]   # X, y, keys
    scaler: dict[str, Any]
    selection: dict[str, Any]


def _torch():
    import torch
    return torch


def load_view(features_root: Path, view: str, *, split: str = "temporal",
              min_document_frequency: int = 2) -> ViewData:
    """Read one view's matrices and keep the columns a training-fitted cutoff allows."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    directory = view_dir(Path(features_root), view)
    vocabulary = json.loads((directory / f"vocab_{split}.json").read_text(encoding="utf-8"))
    frequency = vocabulary["document_frequency"]
    kept = sorted(t for t, n in frequency.items() if n >= min_document_frequency)
    columns = [*kept, *OOV_COLUMNS]

    frames: dict[str, pd.DataFrame] = {}
    for partition in ("train", "validation", "test"):
        path = directory / f"matrix_{split}_{partition}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"no feature matrix at {path}; run `bench build-features`")
        frames[partition] = pd.read_parquet(path)
    missing = [c for c in columns if c not in frames["train"].columns]
    if missing:
        raise ValueError(f"{view}: {len(missing)} vocabulary column(s) absent from the matrix")

    train = frames["train"]
    usable_train = train[train["usable"] == True]                       # noqa: E712
    raw = usable_train[columns].to_numpy(dtype=np.float64)
    logged = np.log1p(np.clip(raw, 0, None))
    mean = logged.mean(axis=0)
    std = logged.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    scaler = {"transform": "log1p then standardise", "fitted_on": "training rows only",
              "fitted_on_rows": int(len(usable_train)),
              "mean": mean.tolist(), "std": std.tolist()}

    partitions: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for partition, frame in frames.items():
        usable = frame[frame["usable"] == True]                         # noqa: E712
        values = usable[columns].to_numpy(dtype=np.float64)
        x = (np.log1p(np.clip(values, 0, None)) - mean) / std
        y = usable["label"].to_numpy(dtype=np.float64)
        partitions[partition] = (x.astype(np.float32), y.astype(np.float32),
                                 usable["tx_key"].tolist())
    unusable = {partition: frame.loc[frame["usable"] != True, "tx_key"].tolist()   # noqa: E712
                for partition, frame in frames.items()}
    selection = {
        "view": view, "split": split,
        "vocabulary_tokens": vocabulary["token_count"],
        "min_document_frequency": min_document_frequency,
        "document_frequency_fitted_on": vocabulary["fitted_on"],
        "features_used": len(columns),
        "features_dropped_by_cutoff": vocabulary["token_count"] - len(kept),
        "out_of_vocabulary_columns": list(OOV_COLUMNS),
        "non_feature_columns_excluded": list(NON_FEATURE),
        "rows_not_usable": {k: len(v) for k, v in unusable.items()},
        "rows_not_usable_keys": unusable,
    }
    return ViewData(view=view, columns=columns, partitions=partitions, scaler=scaler,
                    selection=selection)


class FTTransformer:
    """The paper's architecture, in the smallest faithful form."""

    def __init__(self, n_features: int, *, d_token: int = 32, n_blocks: int = 2,
                 n_heads: int = 8, attention_dropout: float = 0.2, ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0, ffn_factor: float = 4 / 3, seed: int = 42):
        torch = _torch()
        torch.manual_seed(seed)
        nn = torch.nn
        self.n_features, self.d_token, self.n_blocks = n_features, d_token, n_blocks
        self.residual_dropout = residual_dropout
        # feature tokenizer: one weight and one bias per feature, plus a [CLS] token
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        bound = 1.0 / max(1, d_token) ** 0.5
        for tensor in (self.weight, self.bias, self.cls):
            nn.init.uniform_(tensor, -bound, bound)
        d_hidden = int(d_token * ffn_factor)
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        if d_token % n_heads:
            raise ValueError(f"d_token {d_token} is not divisible by n_heads {n_heads}")
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "norm_attention": nn.LayerNorm(d_token),
                # the same multi-head self-attention, written out so the forward pass
                # can use torch's fused kernel: with one token per feature the sequence
                # is long, and materialising a [batch, heads, tokens, tokens] matrix is
                # what makes the naive path memory-bound
                "qkv": nn.Linear(d_token, 3 * d_token),
                "attention_out": nn.Linear(d_token, d_token),
                "norm_ffn": nn.LayerNorm(d_token),
                "ffn_in": nn.Linear(d_token, d_hidden * 2),      # ReGLU: two halves
                "ffn_out": nn.Linear(d_hidden, d_token),
                "ffn_dropout": nn.Dropout(ffn_dropout),
            }))
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        self.module = nn.ModuleList([self.blocks, self.head_norm, self.head])
        self.module.register_parameter("tokenizer_weight", self.weight)
        self.module.register_parameter("tokenizer_bias", self.bias)
        self.module.register_parameter("cls_token", self.cls)

    def parameters(self):
        return self.module.parameters()

    def train(self, flag: bool = True):
        self.module.train(flag)
        return self

    def eval(self):
        self.module.eval()
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        torch = _torch()
        tokens = x.unsqueeze(-1) * self.weight + self.bias            # [B, F, d]
        tokens = torch.cat([self.cls.expand(x.shape[0], 1, self.d_token), tokens], dim=1)
        batch, length = tokens.shape[0], tokens.shape[1]
        head_dim = self.d_token // self.n_heads
        for block in self.blocks:
            normed = block["norm_attention"](tokens)
            qkv = block["qkv"](normed).reshape(batch, length, 3, self.n_heads, head_dim)
            query, key, value = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            attended = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.attention_dropout if self.module.training else 0.0)
            attended = block["attention_out"](
                attended.transpose(1, 2).reshape(batch, length, self.d_token))
            if self.residual_dropout:
                attended = torch.nn.functional.dropout(
                    attended, self.residual_dropout, training=self.module.training)
            tokens = tokens + attended
            normed = block["norm_ffn"](tokens)
            projected = block["ffn_in"](normed)
            first, second = projected.chunk(2, dim=-1)
            hidden = block["ffn_dropout"](torch.relu(first) * second)   # ReGLU
            tokens = tokens + block["ffn_out"](hidden)
        return self.head(torch.relu(self.head_norm(tokens[:, 0]))).squeeze(1)


def average_precision(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    from sklearn.metrics import average_precision_score
    if len(set(int(v) for v in labels)) < 2:
        return None
    return float(average_precision_score(list(labels), list(scores)))


def train(data: ViewData, *, d_token: int = 32, n_blocks: int = 2, n_heads: int = 8,
          lr: float = 1e-4, weight_decay: float = 1e-5, batch_size: int = 64,
          epochs: int = 150, patience: int = 20, seed: int = 42) -> dict[str, Any]:
    """Fit on train, choose the epoch on validation average precision."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_readback = effective_seed(torch, np)
    x_train, y_train, _ = data.partitions["train"]
    x_validation, y_validation, validation_keys = data.partitions["validation"]
    model = FTTransformer(x_train.shape[1], d_token=d_token, n_blocks=n_blocks,
                          n_heads=n_heads, seed=seed)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / positives if positives else 1.0])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = np.random.default_rng(seed)

    xt = torch.tensor(x_train)
    yt = torch.tensor(y_train)
    xv = torch.tensor(x_validation)
    history: list[dict[str, Any]] = []
    best = {"epoch": -1, "validation_ap": None, "state": None}
    for epoch in range(epochs):
        model.train(True)
        order = generator.permutation(len(x_train))
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.tensor(order[start:start + batch_size], dtype=torch.long)
            optimizer.zero_grad()
            logits = model.forward(xt[index])
            loss = loss_fn(logits, yt[index])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model.forward(xv)).tolist()
        ap = average_precision(y_validation.tolist(), scores)
        history.append({"epoch": epoch, "train_loss": total / len(x_train),
                        "validation_ap": ap})
        if ap is not None and (best["validation_ap"] is None or ap > best["validation_ap"]):
            best = {"epoch": epoch, "validation_ap": ap,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if best["epoch"] >= 0 and epoch - best["epoch"] >= patience:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    log.info("%s: best validation AP %.4f at epoch %d of %d", data.view,
             best["validation_ap"] or float("nan"), best["epoch"], len(history))
    return {
        "model": model, "history": history, "selected_epoch": best["epoch"],
        "validation_ap": best["validation_ap"], "epochs_run": len(history),
        "config": {"architecture": "FT-Transformer (arXiv:2106.11959)",
                   "d_token": d_token, "n_blocks": n_blocks, "n_heads": n_heads,
                   "ffn_activation": "ReGLU", "lr": lr, "weight_decay": weight_decay,
                   "optimizer": "AdamW", "batch_size": batch_size, "max_epochs": epochs,
                   "patience": patience, "seed": seed, "pos_weight": float(pos_weight),
                   "loss": "BCEWithLogitsLoss(pos_weight=negatives/positives)",
                   "epoch_selection": "highest validation average precision",
                   "features": x_train.shape[1], "parameters": model.parameter_count()},
        "effective_seed": seed_readback,
        "train_rows": int(len(x_train)), "train_positives": positives,
        "validation_rows": int(len(x_validation)),
    }


def predict(model: FTTransformer, x: np.ndarray, *, batch_size: int = 256) -> list[float]:
    torch = _torch()
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.tensor(x[start:start + batch_size])
            out.extend(torch.sigmoid(model.forward(chunk)).tolist())
    return out

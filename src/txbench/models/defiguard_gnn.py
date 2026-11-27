"""DeFiGuard: a paper-based graph neural network over this project's flow graphs.

This is a **reimplementation from the paper**, not the authors' code, and it is
labelled that way everywhere it is reported. What it reuses is this project's
own cash-flow graph (`txbench.flows`), which is where the honest part of the
work already lives: native value moves taken only from frames that committed,
token transfers from receipt logs with the ERC-20 / ERC-721 ambiguity left
unresolved where the shapes do not separate, and a profit score that refuses to
debit a sender whose counterparty could not be resolved.

The paper's Table I lists four node features. Three are computable here, and
the fourth - node type (account kind *and* source-code transparency) - needs an
`eth_getCode` read per address plus an external source-verification index, so it
is **omitted from the input dimension** rather than defaulted to zero. A zero
would be a claim about every address in the set; an absent column is a stated
gap. The three that remain give five scalars, because transfer frequency and
transfer diversity are each directional:

    frequency_in, frequency_out, diversity_in, diversity_out, profit_score

A transaction whose graph is **incomplete** is not scored. Where the native
side cannot be enumerated - a pre-Byzantium receipt carries no status, so
nobody can say which frames committed - a score would be an answer about a
graph this project does not have, so the row carries `unsupported_input` and
the reason instead.

A transaction with **no transfers at all** is a different thing: that is a real
outcome, so it is scored, with an explicit empty-graph indicator among the
graph-level features rather than a silent all-zero vector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..flows import FlowGraph, KIND_AMBIGUOUS, build_flow_graph, defiguard_node_features
from ..logs import get_logger
from ..trace_model import NormalizedTx

log = get_logger("defiguard.gnn")

MODEL_NAME = "defiguard"
MODEL_VERSION = "paper-reimpl-gnn-3feat"

#: The node features the model is given. `node_type` is deliberately absent.
NODE_FEATURES = ("frequency_in", "frequency_out", "diversity_in", "diversity_out",
                 "profit_score")
OMITTED_NODE_FEATURES = ("node_type",)
#: Graph-level indicators, so an empty graph is a stated fact and not a zero vector.
GRAPH_FEATURES = ("log_nodes", "log_edges", "no_transfers_observed")


@dataclass
class TxGraph:
    """One transaction's flow graph, in the form the model consumes."""
    tx_key: str
    x: np.ndarray                     # [n_nodes, len(NODE_FEATURES)]
    edge_index: np.ndarray            # [2, n_edges] (sender, receiver) positions
    edge_weight: np.ndarray           # [n_edges]
    graph_x: np.ndarray               # [len(GRAPH_FEATURES)]
    outcome: str
    nodes: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class Excluded:
    tx_key: str
    partition: str
    reason: str
    outcome: str



def effective_seed(torch_module, np_module) -> dict[str, int | str]:
    """The seed as the libraries actually hold it, read back after seeding.

    A seed copied from the request proves only that it was asked for. These
    values come from inside the process that will use them, so a seeding call
    that silently failed shows up as a mismatch instead of as reproducibility.
    For NumPy's MT19937 the first word of the state key equals the seed it was
    initialised with.
    """
    return {
        "torch_initial_seed": int(torch_module.initial_seed()),
        "numpy_mt19937_key0": int(np_module.random.get_state()[1][0]),
        "read_from": "torch.initial_seed() and numpy.random.get_state(), inside this process",
    }

def edge_weight_of(graph: FlowGraph) -> dict[int, float]:
    """A weight per edge: the amount as a share of that asset's largest move.

    The same normalisation the profit score uses, so one asset's wei cannot
    dwarf another's token units. An edge the standards give no fungible amount
    (an ERC-721 transfer, or an unresolved shape) keeps weight 0 and is still
    present as structure - the transfer happened, its size is unknown.
    """
    largest: dict[str, int] = {}
    for edge in graph.edges:
        if edge.amount is None or edge.asset_kind == KIND_AMBIGUOUS:
            continue
        largest[edge.asset] = max(largest.get(edge.asset, 0), int(edge.amount))
    weights: dict[int, float] = {}
    for position, edge in enumerate(graph.edges):
        if edge.amount is None or edge.asset_kind == KIND_AMBIGUOUS:
            weights[position] = 0.0
            continue
        scale = largest.get(edge.asset) or 1
        weights[position] = float(int(edge.amount) / scale)
    return weights


def to_tx_graph(tx_key: str, graph: FlowGraph) -> TxGraph:
    """Build the model's view of one flow graph. No label is read here."""
    features = defiguard_node_features(graph)
    nodes = graph.nodes()
    position = {address: i for i, address in enumerate(nodes)}
    x = np.zeros((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
    for address, index in position.items():
        row = features["features"][address]
        x[index] = [float(row[name]) for name in NODE_FEATURES]

    weights = edge_weight_of(graph)
    pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for offset, edge in enumerate(graph.edges):
        if edge.sender not in position or edge.receiver not in position:
            # A log whose counterparty could not be read from the topics is not an
            # edge between two known accounts; it is counted in the flow graph's own
            # report and cannot become a message here.
            continue
        pairs.append((position[edge.sender], position[edge.receiver]))
        edge_weights.append(weights[offset])
    edge_index = (np.array(pairs, dtype=np.int64).T if pairs
                  else np.zeros((2, 0), dtype=np.int64))
    graph_x = np.array([math.log1p(len(nodes)), math.log1p(len(pairs)),
                        1.0 if graph.outcome == "no_transfers_observed" else 0.0],
                       dtype=np.float32)
    return TxGraph(tx_key=tx_key, x=x, edge_index=edge_index,
                   edge_weight=np.array(edge_weights, dtype=np.float32), graph_x=graph_x,
                   outcome=graph.outcome, nodes=nodes)


def build_graphs(records: Sequence[tuple[str, NormalizedTx | None]],
                 partitions: dict[str, str]) -> tuple[list[TxGraph], list[Excluded]]:
    """Flow graphs for every transaction, and the ones that cannot be scored."""
    graphs: list[TxGraph] = []
    excluded: list[Excluded] = []
    for tx_key, tx in records:
        partition = partitions.get(tx_key) or ""
        if tx is None:
            excluded.append(Excluded(tx_key, partition, "no normalised record", "absent"))
            continue
        graph = build_flow_graph(tx)
        if not graph.complete:
            reason = "; ".join(r for r in (graph.native_reason, graph.token_reason) if r)
            excluded.append(Excluded(tx_key, partition, reason or "incomplete graph",
                                     graph.outcome))
            continue
        graphs.append(to_tx_graph(tx_key, graph))
    return graphs, excluded


# --- the model ----------------------------------------------------------------

def _torch():
    import torch
    return torch


class DeFiGuardGNN:
    """Two directional message-passing layers, then a graph-level head.

    Each layer gives a node three things: its own state, the mean of what flowed
    **in** to it, and the mean of what flowed **out** of it, each weighted by the
    normalised amount. Direction is kept because the paper's features are
    directional - a drained pool and a draining account differ only in which way
    the value went.
    """

    def __init__(self, *, in_dim: int = len(NODE_FEATURES), graph_dim: int = len(GRAPH_FEATURES),
                 hidden: int = 64, layers: int = 2, dropout: float = 0.2, seed: int = 42):
        torch = _torch()
        torch.manual_seed(seed)
        self.hidden, self.layers, self.dropout_p = hidden, layers, dropout
        nn = torch.nn
        self.self_weights = nn.ModuleList()
        self.in_weights = nn.ModuleList()
        self.out_weights = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for i in range(layers):
            self.self_weights.append(nn.Linear(dims[i], hidden))
            self.in_weights.append(nn.Linear(dims[i], hidden, bias=False))
            self.out_weights.append(nn.Linear(dims[i], hidden, bias=False))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + graph_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.module = nn.ModuleList([self.self_weights, self.in_weights, self.out_weights,
                                     self.norms, self.head])

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

    def forward(self, batch: dict[str, Any]):
        torch = _torch()
        x, graph_x = batch["x"], batch["graph_x"]
        src, dst, weight = batch["src"], batch["dst"], batch["weight"]
        node_graph, n_graphs = batch["node_graph"], batch["n_graphs"]
        for i in range(self.layers):
            messages_in = torch.zeros((x.shape[0], self.hidden), dtype=x.dtype)
            messages_out = torch.zeros((x.shape[0], self.hidden), dtype=x.dtype)
            if src.numel():
                # +1 so an amount of zero still carries structure
                scaled = (1.0 + weight).unsqueeze(1)
                messages_in.index_add_(0, dst, self.in_weights[i](x[src]) * scaled)
                messages_out.index_add_(0, src, self.out_weights[i](x[dst]) * scaled)
                in_degree = torch.zeros((x.shape[0], 1), dtype=x.dtype)
                out_degree = torch.zeros((x.shape[0], 1), dtype=x.dtype)
                in_degree.index_add_(0, dst, torch.ones((src.numel(), 1), dtype=x.dtype))
                out_degree.index_add_(0, src, torch.ones((src.numel(), 1), dtype=x.dtype))
                messages_in = messages_in / in_degree.clamp(min=1.0)
                messages_out = messages_out / out_degree.clamp(min=1.0)
            x = torch.relu(self.norms[i](self.self_weights[i](x) + messages_in + messages_out))
            x = torch.nn.functional.dropout(x, self.dropout_p, training=self.module.training)

        pooled_sum = torch.zeros((n_graphs, self.hidden), dtype=x.dtype)
        counts = torch.zeros((n_graphs, 1), dtype=x.dtype)
        if x.shape[0]:
            pooled_sum.index_add_(0, node_graph, x)
            counts.index_add_(0, node_graph, torch.ones((x.shape[0], 1), dtype=x.dtype))
        mean_pool = pooled_sum / counts.clamp(min=1.0)
        max_pool = torch.zeros((n_graphs, self.hidden), dtype=x.dtype)
        if x.shape[0]:
            # include_self keeps a graph with no nodes at zero instead of -inf
            max_pool = max_pool.scatter_reduce_(
                0, node_graph.unsqueeze(1).expand(-1, self.hidden), x, reduce="amax",
                include_self=True)
        return self.head(torch.cat([mean_pool, max_pool, graph_x], dim=1)).squeeze(1)


def collate(graphs: Sequence[TxGraph]) -> dict[str, Any]:
    """Batch graphs as one block-diagonal graph, with a node->graph index."""
    torch = _torch()
    xs, graph_xs, srcs, dsts, weights, node_graph = [], [], [], [], [], []
    offset = 0
    for position, graph in enumerate(graphs):
        xs.append(graph.x)
        graph_xs.append(graph.graph_x)
        if graph.n_edges:
            srcs.append(graph.edge_index[0] + offset)
            dsts.append(graph.edge_index[1] + offset)
            weights.append(graph.edge_weight)
        node_graph.extend([position] * graph.n_nodes)
        offset += graph.n_nodes
    x = np.concatenate(xs) if xs else np.zeros((0, len(NODE_FEATURES)), dtype=np.float32)
    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "graph_x": torch.tensor(np.stack(graph_xs), dtype=torch.float32),
        "src": torch.tensor(np.concatenate(srcs) if srcs else np.zeros(0), dtype=torch.long),
        "dst": torch.tensor(np.concatenate(dsts) if dsts else np.zeros(0), dtype=torch.long),
        "weight": torch.tensor(np.concatenate(weights) if weights else np.zeros(0),
                               dtype=torch.float32),
        "node_graph": torch.tensor(node_graph, dtype=torch.long),
        "n_graphs": len(graphs),
        "keys": [g.tx_key for g in graphs],
    }


def standardize_fit(graphs: Sequence[TxGraph]) -> dict[str, Any]:
    """Node- and graph-feature means and standard deviations, training rows only."""
    node_rows = [g.x for g in graphs if g.n_nodes]
    node_stack = np.concatenate(node_rows) if node_rows else np.zeros((1, len(NODE_FEATURES)),
                                                                     dtype=np.float32)
    graph_stack = np.stack([g.graph_x for g in graphs]) if graphs else np.zeros(
        (1, len(GRAPH_FEATURES)), dtype=np.float32)
    return {
        "fitted_on": "training rows only",
        "fitted_on_graphs": len(graphs),
        "node_mean": node_stack.mean(axis=0).tolist(),
        "node_std": np.where(node_stack.std(axis=0) > 0, node_stack.std(axis=0), 1.0).tolist(),
        "graph_mean": graph_stack.mean(axis=0).tolist(),
        "graph_std": np.where(graph_stack.std(axis=0) > 0, graph_stack.std(axis=0), 1.0).tolist(),
        "node_features": list(NODE_FEATURES),
        "graph_features": list(GRAPH_FEATURES),
    }


def standardize_apply(graphs: Sequence[TxGraph], stats: dict[str, Any]) -> list[TxGraph]:
    node_mean = np.array(stats["node_mean"], dtype=np.float32)
    node_std = np.array(stats["node_std"], dtype=np.float32)
    graph_mean = np.array(stats["graph_mean"], dtype=np.float32)
    graph_std = np.array(stats["graph_std"], dtype=np.float32)
    out = []
    for graph in graphs:
        x = (graph.x - node_mean) / node_std if graph.n_nodes else graph.x
        out.append(TxGraph(tx_key=graph.tx_key, x=x.astype(np.float32),
                           edge_index=graph.edge_index, edge_weight=graph.edge_weight,
                           graph_x=((graph.graph_x - graph_mean) / graph_std).astype(np.float32),
                           outcome=graph.outcome, nodes=graph.nodes))
    return out


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    from sklearn.metrics import average_precision_score
    if len(set(labels)) < 2:
        return None
    return float(average_precision_score(list(labels), list(scores)))


def train(graphs_by_partition: dict[str, list[TxGraph]], labels: dict[str, int], *,
          hidden: int = 64, layers: int = 2, dropout: float = 0.2, lr: float = 1e-3,
          weight_decay: float = 1e-4, epochs: int = 300, batch_size: int = 32,
          patience: int = 30, seed: int = 42) -> dict[str, Any]:
    """Fit on train, select the epoch on validation average precision, then stop.

    The test partition is never read here: it is not loaded, not scored and not
    used to stop. The returned state is the epoch that was best on validation.
    """
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_readback = effective_seed(torch, np)
    train_graphs = list(graphs_by_partition.get("train", []))
    validation_graphs = list(graphs_by_partition.get("validation", []))
    if not train_graphs:
        raise ValueError("no training graphs")

    stats = standardize_fit(train_graphs)
    train_graphs = standardize_apply(train_graphs, stats)
    validation_graphs = standardize_apply(validation_graphs, stats)

    model = DeFiGuardGNN(hidden=hidden, layers=layers, dropout=dropout, seed=seed)
    positives = sum(1 for g in train_graphs if labels.get(g.tx_key) == 1)
    negatives = len(train_graphs) - positives
    pos_weight = torch.tensor([negatives / positives if positives else 1.0])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = np.random.default_rng(seed)

    history: list[dict[str, Any]] = []
    best = {"epoch": -1, "validation_ap": None, "state": None, "validation_loss": None}
    for epoch in range(epochs):
        model.train(True)
        order = generator.permutation(len(train_graphs))
        total = 0.0
        for start in range(0, len(order), batch_size):
            chunk = [train_graphs[i] for i in order[start:start + batch_size]]
            batch = collate(chunk)
            target = torch.tensor([float(labels.get(k, 0)) for k in batch["keys"]])
            optimizer.zero_grad()
            logits = model.forward(batch)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        train_loss = total / len(train_graphs)
        scores, keys = predict(model, validation_graphs)
        truth = [labels.get(k, 0) for k in keys]
        ap = average_precision(truth, scores) if keys else None
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_ap": ap})
        if ap is not None and (best["validation_ap"] is None or ap > best["validation_ap"]):
            best = {"epoch": epoch, "validation_ap": ap,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "validation_loss": None}
        if best["epoch"] >= 0 and epoch - best["epoch"] >= patience:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    log.info("defiguard: best validation AP %.4f at epoch %d of %d",
             best["validation_ap"] or float("nan"), best["epoch"], len(history))
    return {"model": model, "scaler": stats, "history": history,
            "selected_epoch": best["epoch"], "validation_ap": best["validation_ap"],
            "epochs_run": len(history),
            "config": {"hidden": hidden, "layers": layers, "dropout": dropout, "lr": lr,
                       "weight_decay": weight_decay, "max_epochs": epochs,
                       "batch_size": batch_size, "patience": patience, "seed": seed,
                       "pos_weight": float(pos_weight), "optimizer": "Adam",
                       "loss": "BCEWithLogitsLoss(pos_weight=negatives/positives)",
                       "epoch_selection": "highest validation average precision",
                       "node_features": list(NODE_FEATURES),
                       "omitted_node_features": list(OMITTED_NODE_FEATURES),
                       "graph_features": list(GRAPH_FEATURES),
                       "input_dim": len(NODE_FEATURES)},
            "effective_seed": seed_readback,
            "train_graphs": len(train_graphs), "validation_graphs": len(validation_graphs),
            "train_positives": positives}


def predict(model: DeFiGuardGNN, graphs: Sequence[TxGraph], *, batch_size: int = 64
            ) -> tuple[list[float], list[str]]:
    """Sigmoid probabilities, higher meaning more likely an attack."""
    torch = _torch()
    model.eval()
    scores: list[float] = []
    keys: list[str] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            chunk = list(graphs[start:start + batch_size])
            if not chunk:
                continue
            batch = collate(chunk)
            logits = model.forward(batch)
            scores.extend(torch.sigmoid(logits).tolist())
            keys.extend(batch["keys"])
    return scores, keys

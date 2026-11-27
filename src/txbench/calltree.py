"""Reconstruct a call tree from explicit trace paths, and nothing else.

A trace path is a Parity-style address: the sequence of child indexes from the
root to a frame, so `()` is the root, `(0,)` its first child, `(0, 2)` that
child's third child. The parent of a path is that path with its last index
removed. **That identity is the only nesting evidence used here.** Row order,
repeated addresses, call depth heuristics and gas arithmetic are all ignored:
each of them can agree with the real tree on most transactions and lie on the
ones that matter, and a wrong parent silently changes which frames an
ancestor's revert undid.

What a reconstruction therefore reports, instead of guessing:

  * a **missing parent** - a path is present whose parent path is not, so that
    frame's ancestry is unknown and stays unknown;
  * a **duplicate path** - two rows claim one position in the tree, so neither
    can be trusted to be that position;
  * a **missing child** - the source states a direct-child count that more
    children would be needed to satisfy;
  * **no root** - nothing claims the empty path.

Only a tree with none of these is `complete`. Anything else keeps the frames it
could place and leaves the rest unresolved, because for DeFiGuard an unresolved
ancestry must not become a committed transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import TxBenchError
from .ids import normalize_address, normalize_hex_bytes
from .trace_model import CallFrame, NormalizedTx

#: The reconstruction placed every frame and nothing is missing or doubled.
TREE_COMPLETE = "complete"
#: Frames were placed, but at least one validation issue remains.
TREE_INCOMPLETE = "incomplete"
#: Nothing usable arrived (no rows, or no root).
TREE_UNUSABLE = "unusable"

ISSUE_NO_ROOT = "no_root"
ISSUE_MISSING_PARENT = "missing_parent"
ISSUE_DUPLICATE_PATH = "duplicate_path"
ISSUE_MISSING_CHILD = "missing_child"
ISSUE_BAD_PATH = "unparsable_path"

#: Default provenance label for a warehouse trace row.
TRACE_SOURCE_DEFAULT = "warehouse_trace_row"


class CallTreeError(TxBenchError):
    """A trace path could not be parsed into a sequence of child indexes."""


def parse_trace_path(value: Any) -> tuple[int, ...]:
    """Parse a trace path into a tuple of child indexes. `()` is the root.

    Accepts the two shapes a warehouse table uses: a repeated integer field
    (a list) and a delimited string (`""`, `"0"`, `"0,1"`, `"0_1"`). A value
    that is not a sequence of non-negative integers is an error, never a
    silently-dropped frame.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        parts: Sequence[Any] = value
    else:
        text = str(value).strip()
        if text in ("", "[]", "root"):
            return ()
        if text.startswith("-"):
            raise CallTreeError(f"trace path {value!r} holds a negative child index")
        # "-" is deliberately not a separator: it collides with a minus sign.
        for sep in (",", "_", " "):
            if sep in text:
                parts = [p for p in text.split(sep) if p != ""]
                break
        else:
            parts = [text]
    out: list[int] = []
    for part in parts:
        try:
            number = int(str(part).strip())
        except (TypeError, ValueError):
            raise CallTreeError(f"trace path {value!r} is not a sequence of child indexes") from None
        if number < 0:
            raise CallTreeError(f"trace path {value!r} holds a negative child index")
        out.append(number)
    return tuple(out)


def parent_path(path: Sequence[int]) -> tuple[int, ...] | None:
    """The parent of a trace path: the path without its last index. Root: None."""
    path = tuple(path)
    return None if not path else path[:-1]


@dataclass
class TraceRow:
    """One trace row as the source supplied it, before any tree is built."""
    path: tuple[int, ...]
    call_type: str | None = None
    from_address: str | None = None
    to_address: str | None = None
    value_wei: int | None = None
    input_hex: str | None = None
    output_hex: str | None = None
    error: str | None = None
    gas: int | None = None
    gas_used: int | None = None
    #: The source's own count of this frame's direct children, when it states one.
    subcall_count: int | None = None
    #: Where the row came from, for provenance.
    source: str | None = None

    @property
    def is_root(self) -> bool:
        return not self.path

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass
class ReconstructedTree:
    tx_hash: str
    status: str = TREE_UNUSABLE
    frames: list[CallFrame] = field(default_factory=list)
    #: path -> index into `frames`, for callers that need to look a frame up.
    index_by_path: dict[tuple[int, ...], int] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    #: Frames whose ancestry could not be walked to the root.
    unresolved_ancestry: list[int] = field(default_factory=list)
    row_count: int = 0
    source: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == TREE_COMPLETE

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def max_depth(self) -> int | None:
        depths = [f.depth for f in self.frames if f.depth is not None]
        return max(depths) if depths else None

    @property
    def reverted_frames(self) -> int:
        return sum(1 for f in self.frames if f.reverted)

    @property
    def root(self) -> CallFrame | None:
        position = self.index_by_path.get(())
        return self.frames[position] if position is not None else None

    def note(self, kind: str, **detail: Any) -> None:
        self.issues.append({"issue": kind, **detail})

    def summary(self) -> dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "status": self.status,
            "row_count": self.row_count,
            "frame_count": self.frame_count,
            "max_depth": self.max_depth,
            "reverted_frames": self.reverted_frames,
            "frames_with_unresolved_ancestry": len(self.unresolved_ancestry),
            "issues": self.issues,
            "source": self.source,
        }


def reconstruct(tx_hash: str, rows: Iterable[TraceRow], *, source: str | None = None
                ) -> ReconstructedTree:
    """Place every row at its stated path and derive parents, depth and reverts.

    `reverted` is true for a frame that failed **or whose ancestor failed**: a
    child can return successfully and still be undone when its parent reverts
    later. Where an ancestor could not be found, `reverted` stays `None` -
    unknown - rather than defaulting to false.
    """
    tree = ReconstructedTree(tx_hash=tx_hash, source=source)
    rows = list(rows)
    tree.row_count = len(rows)
    if not rows:
        tree.note(ISSUE_NO_ROOT, detail="the source returned no trace rows")
        return tree

    by_path: dict[tuple[int, ...], TraceRow] = {}
    for row in rows:
        if row.path in by_path:
            tree.note(ISSUE_DUPLICATE_PATH, path=list(row.path))
            continue
        by_path[row.path] = row

    if () not in by_path:
        tree.note(ISSUE_NO_ROOT, detail="no row claims the empty trace path")
        return tree

    # Sort by path so a parent is always placed before its children: shorter
    # paths first, then lexicographically. This orders the output; it never
    # decides who a parent is.
    ordered = sorted(by_path, key=lambda p: (len(p), p))
    position_of: dict[tuple[int, ...], int] = {}
    for position, path in enumerate(ordered):
        position_of[path] = position

    children: dict[tuple[int, ...], int] = {}
    for path in ordered:
        parent = parent_path(path)
        if parent is not None:
            children[parent] = children.get(parent, 0) + 1

    for position, path in enumerate(ordered):
        row = by_path[path]
        parent = parent_path(path)
        parent_position = None
        if parent is not None:
            parent_position = position_of.get(parent)
            if parent_position is None:
                tree.note(ISSUE_MISSING_PARENT, path=list(path), parent=list(parent))
        frame = CallFrame(
            index=position,
            call_type=(row.call_type or "UNKNOWN").upper(),
            from_address=normalize_address(row.from_address),
            to_address=normalize_address(row.to_address),
            value_wei=row.value_wei,
            input_hex=normalize_hex_bytes(row.input_hex),
            output_hex=normalize_hex_bytes(row.output_hex),
            error=row.error or None,
            gas=row.gas,
            gas_used=row.gas_used,
            parent_index=parent_position,
            depth=len(path),
            is_root=row.is_root,
        )
        tree.frames.append(frame)
        tree.index_by_path[path] = position

        if row.subcall_count is not None:
            observed = children.get(path, 0)
            if observed < row.subcall_count:
                tree.note(ISSUE_MISSING_CHILD, path=list(path),
                          stated=row.subcall_count, observed=observed)

    _mark_reverted(tree, ordered, by_path)

    hard = {ISSUE_NO_ROOT, ISSUE_MISSING_PARENT, ISSUE_DUPLICATE_PATH, ISSUE_MISSING_CHILD,
            ISSUE_BAD_PATH}
    tree.status = TREE_INCOMPLETE if any(i["issue"] in hard for i in tree.issues) \
        else TREE_COMPLETE
    return tree


def _mark_reverted(tree: ReconstructedTree, ordered: list[tuple[int, ...]],
                   by_path: dict[tuple[int, ...], TraceRow]) -> None:
    """A frame is reverted when it failed or any ancestor failed; unknown when
    the chain to the root is broken."""
    for path in ordered:
        position = tree.index_by_path[path]
        frame = tree.frames[position]
        failed = by_path[path].failed
        if failed:
            frame.reverted = True
            continue
        # Walk to the root. A gap anywhere leaves the answer unknown.
        cursor = parent_path(path)
        resolved = True
        ancestor_failed = False
        while cursor is not None:
            ancestor = by_path.get(cursor)
            if ancestor is None:
                resolved = False
                break
            if ancestor.failed:
                ancestor_failed = True
                break
            cursor = parent_path(cursor)
        if not resolved:
            frame.reverted = None
            tree.unresolved_ancestry.append(position)
        else:
            frame.reverted = ancestor_failed


def cross_check_root(tree: ReconstructedTree, tx: NormalizedTx) -> dict[str, Any]:
    """Compare the reconstructed root against the local envelope and receipt.

    A mismatch means the rows are not this transaction's, or the source's root
    convention differs. Either way it is reported, not reconciled: a field that
    disagrees is listed with both values.
    """
    root = tree.root
    if root is None:
        return {"checked": False, "reason": "the reconstruction has no root frame"}
    comparisons = {
        "from_address": (root.from_address, tx.from_address),
        "to_address": (root.to_address, tx.to_address),
        "value_wei": (root.value_wei, tx.value_wei),
        "input_hex": (root.input_hex, tx.input_hex),
    }
    mismatches = {}
    compared = []
    for name, (recovered, local) in comparisons.items():
        if recovered is None or local is None:
            continue
        compared.append(name)
        if name == "value_wei":
            equal = int(recovered) == int(local)
        else:
            equal = str(recovered).lower() == str(local).lower()
        if not equal:
            mismatches[name] = {"reconstructed": recovered, "local": local}

    # The receipt says whether the transaction as a whole committed; the root
    # frame's error says whether the outermost call failed. They must agree.
    status_note = None
    if tx.receipt_status is not None:
        root_failed = bool(root.error)
        if (tx.receipt_status == 0) != root_failed:
            status_note = {"receipt_status": tx.receipt_status, "root_error": root.error}
            mismatches["receipt_status"] = status_note
        compared.append("receipt_status")

    return {"checked": True, "fields_compared": compared, "mismatches": mismatches,
            "agrees": not mismatches}


def apply_nesting(tx: NormalizedTx, tree: ReconstructedTree, *, source: str) -> NormalizedTx:
    """Replace a flat frame list with the reconstructed tree.

    Only a complete reconstruction is allowed to claim nesting. An incomplete
    one still replaces the frames - they carry more than the flat list did -
    but `call_tree_has_nesting` stays false, so every consumer that gates on
    nesting keeps treating the transaction as unresolved.
    """
    if not tree.frames:
        return tx
    tx.frames = list(tree.frames)
    tx.call_tree_includes_root = tree.root is not None
    tx.sources["internal_calls"] = source
    # Real trace paths arrived, so every frame's revert state is now either
    # known or explicitly unknown. This stays true even for an incomplete
    # tree, and it is what stops an unresolved frame counting as committed.
    tx.call_tree_ancestry_known = True
    if tree.complete:
        tx.call_tree_has_nesting = True
        tx.call_tree_status = "complete"
        tx.field_gaps.pop("nested_calls", None)
        tx.field_gaps.pop("reverted_ancestry", None)
    else:
        tx.call_tree_has_nesting = False
        reasons = sorted({i["issue"] for i in tree.issues})
        tx.note_gap("nested_calls",
                    f"reconstruction from {source} is incomplete: {', '.join(reasons)}")
    return tx

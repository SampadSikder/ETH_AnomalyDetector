"""Mapping from the upstream release's columns to this project's fields.

Written only after reading the release's data dictionary. The resolved mapping
is persisted next to the manifest so an auditor can see exactly which upstream
column became which field, and so a schema change upstream fails loudly
instead of quietly shifting a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import SchemaMappingError

#: canonical field -> (candidate upstream column names, required)
FIELD_ALIASES: dict[str, tuple[tuple[str, ...], bool]] = {
    "record_id":        (("record_id",), False),
    "chain_slug":       (("chain", "chain_slug"), False),
    "chain_id":         (("chain_id",), True),
    "tx_hash":          (("tx_hash", "hash", "transaction_hash"), True),
    "label":            (("label",), True),
    "label_name":       (("label_name",), False),
    "block_number":     (("block_number",), False),
    "block_timestamp":  (("block_timestamp", "timestamp"), False),
    "source_category":  (("category",), False),
    "subcategory":      (("subcategory",), False),
    "behavior_tags":    (("behavior_tags",), False),
    "label_tier":       (("label_confidence", "label_tier"), False),
    "group_incident":   (("group_incident", "incident_id"), False),
    "group_protocol":   (("group_protocol", "protocol"), False),
    "group_address":    (("group_address",), False),
}

#: Upstream split columns are imported verbatim under their own names.
SPLIT_PREFIX = "split_"

#: Separator used by the upstream list-valued text columns (e.g. behavior_tags).
LIST_SEPARATOR = ";"

#: Values of the upstream `category` column that denote an attack class.
ATTACK_CLASSES = ("exploit", "key_compromise")

#: Label values accepted verbatim. Anything else is `unknown` and is NEVER
#: coerced to benign.
LABEL_VALUES = {"1": 1, "0": 0}


@dataclass
class SchemaMapping:
    source_file: str
    columns: list[str]
    resolved: dict[str, str] = field(default_factory=dict)   # canonical -> upstream column
    absent: list[str] = field(default_factory=list)          # canonical fields with no column
    split_columns: list[str] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)

    def get(self, row: dict[str, str], field_name: str) -> str | None:
        col = self.resolved.get(field_name)
        if col is None:
            return None
        value = row.get(col)
        if value is None:
            return None
        value = value.strip()
        return value or None

    def as_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "upstream_columns": self.columns,
            "field_to_column": dict(sorted(self.resolved.items())),
            "fields_absent_upstream": sorted(self.absent),
            "split_columns": sorted(self.split_columns),
            "columns_not_mapped": sorted(self.unmapped_columns),
            "list_separator": LIST_SEPARATOR,
            "attack_classes": list(ATTACK_CLASSES),
            "label_values": LABEL_VALUES,
        }


def resolve(columns: list[str], source_file: str) -> SchemaMapping:
    """Bind canonical fields to actual columns; fail if a required one is absent."""
    mapping = SchemaMapping(source_file=source_file, columns=list(columns))
    lower = {c.strip().lower(): c for c in columns}
    used: set[str] = set()
    for canonical, (aliases, required) in FIELD_ALIASES.items():
        hit = next((lower[a] for a in aliases if a in lower), None)
        if hit is None:
            if required:
                raise SchemaMappingError(
                    f"{source_file}: no column for required field {canonical!r} "
                    f"(tried {list(aliases)}; found {columns})")
            mapping.absent.append(canonical)
            continue
        mapping.resolved[canonical] = hit
        used.add(hit)
    mapping.split_columns = sorted(c for c in columns if c.strip().lower().startswith(SPLIT_PREFIX))
    used.update(mapping.split_columns)
    mapping.unmapped_columns = [c for c in columns if c not in used]
    return mapping


def parse_label(raw: str | None) -> tuple[int | None, str]:
    """Return (label, status). An unrecognised label is `unknown`, never 0."""
    if raw is None or raw == "":
        return None, "missing"
    key = str(raw).strip()
    if key in LABEL_VALUES:
        return LABEL_VALUES[key], "labeled"
    return None, "unrecognized"


def parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(LIST_SEPARATOR) if part.strip()]

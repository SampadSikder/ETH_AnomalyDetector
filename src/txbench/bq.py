"""BigQuery access for the call-trace backfill: introspect, price, then scan.

Three rules shape this module.

**The schema is discovered, never assumed.** A warehouse table's column names
are not part of this project's contract, so `resolve_columns` matches the roles
the reconstruction needs against the table's actual schema and reports every
role it could not fill. If the call-path role is unfilled the table cannot
support a call tree at all, and that is reported *before* any data is scanned.

**A partition predicate is built from validated literals.** Partition pruning
is what makes this cheap, and pruning on a query parameter is not guaranteed,
so the day predicate is assembled from dates this repo produced and re-validated
here, while the hash list - the long, varying part - stays a parameter.

**Nothing executes without a byte ceiling.** Every job carries
`maximum_bytes_billed`, set from the budget that is left. BigQuery then fails
the job instead of billing past it, which is the only enforcement that does not
depend on this code being correct.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import TxBenchError
from .hashutil import canonical_json
from .logs import get_logger

log = get_logger("bq")

#: The public Ethereum traces table named in the enrichment plan.
DEFAULT_TABLE = "bigquery-public-data.goog_blockchain_ethereum_mainnet_us.traces"

#: Roles the reconstruction needs, each with the column names a warehouse
#: table plausibly uses for it, in order of preference. A role with no match is
#: reported; it is never filled by guessing at a similar-looking column.
#: A candidate may be a dotted path into a RECORD column: the Ethereum public
#: view keeps the caller, callee, value and input inside `action`, so a role
#: that only ever looked at top-level names would report them missing.
COLUMN_ROLES: dict[str, tuple[str, ...]] = {
    "tx_hash": ("transaction_hash", "tx_hash", "hash"),
    "call_path": ("trace_address", "call_path", "trace_path", "address_path", "trace_id"),
    "call_type": ("action.call_type", "call_type", "type", "action_type"),
    "trace_type": ("trace_type", "action_kind"),
    "from_address": ("action.from_address", "from_address", "caller_address",
                     "sender_address", "from"),
    "to_address": ("action.to_address", "to_address", "callee_address",
                   "receiver_address", "to"),
    # The lossless STRING form is preferred: wei does not survive a float, and
    # BIGNUMERIC arrives as a Decimal that still has to be re-parsed.
    "value": ("action.value_lossless", "value_lossless", "action.value", "value",
              "call_value"),
    "input": ("action.input", "input", "call_input", "calldata"),
    "output": ("result.output", "output", "call_output", "return_data"),
    "error": ("error", "revert_reason", "error_message"),
    "status": ("status", "call_status", "success"),
    "subcall_count": ("subtrace_count", "subtraces", "subcall_count", "child_count"),
    "block_number": ("block_number", "block_height"),
    "block_timestamp": ("block_timestamp", "block_time", "timestamp"),
}

#: Columns that can carry an explicit UTC bound when no partition field is
#: advertised - which is the normal case for a view.
TIME_BOUND_CANDIDATES = ("block_timestamp", "block_time", "timestamp")

#: Roles without which no tree can be built.
REQUIRED_ROLES = ("tx_hash", "call_path")
#: Roles needed for the DeFiGuard rebuild specifically.
FLOW_ROLES = ("call_type", "from_address", "to_address", "value")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BigQueryUnavailable(TxBenchError):
    """The client library or credentials are absent."""


class NoCallPath(TxBenchError):
    """The table has no column that can carry a call path."""


class BudgetExhausted(TxBenchError):
    """The configured byte budget cannot cover this job."""


def load_client(project: str | None = None):
    """Build a BigQuery client from Application Default Credentials.

    Raises `BigQueryUnavailable` with an actionable message rather than a
    stack trace when the library or the credentials are missing: offline work
    on this module must fail cleanly, not confusingly.
    """
    try:
        from google.cloud import bigquery  # noqa: PLC0415
    except ImportError as exc:
        raise BigQueryUnavailable(
            "google-cloud-bigquery is not installed in this environment. Install the "
            "declared extra: `conda run -n txbench python -m pip install -e '.[bigquery]'`"
        ) from exc
    try:
        import google.auth  # noqa: PLC0415
        credentials, detected = google.auth.default(
            scopes=["https://www.googleapis.com/auth/bigquery"])
    except Exception as exc:  # google.auth.exceptions.DefaultCredentialsError and friends
        raise BigQueryUnavailable(
            "no Google Application Default Credentials are available. Run "
            "`gcloud auth application-default login`, or point "
            "GOOGLE_APPLICATION_CREDENTIALS at a service-account key file."
        ) from exc
    chosen = project or detected
    if not chosen:
        raise BigQueryUnavailable(
            "no Google Cloud project is configured. Set GOOGLE_CLOUD_PROJECT in .env, or "
            "pass --project: BigQuery bills the querying project, and a public dataset "
            "still needs one.")
    return bigquery.Client(project=chosen, credentials=credentials)


# --- schema introspection -----------------------------------------------------

@dataclass
class TableDescription:
    table: str
    columns: dict[str, str] = field(default_factory=dict)      # name -> type
    modes: dict[str, str] = field(default_factory=dict)        # name -> NULLABLE/REPEATED
    partition_field: str | None = None
    partition_type: str | None = None                          # DAY / MONTH / HOUR / YEAR
    partition_kind: str | None = None        # time / range / none / unadvertised
    table_type: str | None = None            # TABLE / VIEW / MATERIALIZED_VIEW
    view_query: str | None = None
    require_partition_filter: bool = False
    clustering: list[str] = field(default_factory=list)
    num_rows: int | None = None
    num_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "columns": self.columns,
            "repeated_columns": sorted(n for n, m in self.modes.items() if m == "REPEATED"),
            "table_type": self.table_type,
            "partition_field": self.partition_field,
            "partition_granularity": self.partition_type,
            "partition_kind": self.partition_kind,
            "partition_metadata_available": self.partition_kind in ("time", "range"),
            "view_definition_available": bool(self.view_query),
            "view_definition": self.view_query,
            "require_partition_filter": self.require_partition_filter,
            "clustering": self.clustering,
            "num_rows": self.num_rows,
            "num_bytes": self.num_bytes,
        }


def _flatten_schema(fields, prefix: str = ""):
    """Every column as a dotted path, so a RECORD's fields are addressable."""
    for column in fields:
        name = prefix + column.name
        yield name, column
        for nested in _flatten_schema(getattr(column, "fields", ()) or (), name + "."):
            yield nested


def describe_table(client, table: str = DEFAULT_TABLE) -> TableDescription:
    """Read the real schema, object type, partitioning and clustering.

    `get_table` on a view reports no partitioning and no row count, because the
    view has none of its own. That says nothing about the data underneath, so
    the absence is recorded as `unadvertised` rather than as `none`: a view is
    never treated as an unpartitioned table.
    """
    meta = client.get_table(table)
    out = TableDescription(table=table, num_rows=getattr(meta, "num_rows", None),
                           num_bytes=getattr(meta, "num_bytes", None),
                           table_type=getattr(meta, "table_type", None),
                           view_query=(getattr(meta, "view_query", None)
                                       or getattr(meta, "mview_query", None) or None))
    for name, column in _flatten_schema(meta.schema):
        out.columns[name] = column.field_type
        out.modes[name] = column.mode or "NULLABLE"
    time_partitioning = getattr(meta, "time_partitioning", None)
    range_partitioning = getattr(meta, "range_partitioning", None)
    if time_partitioning is not None:
        out.partition_kind = "time"
        out.partition_field = time_partitioning.field or "_PARTITIONTIME"
        out.partition_type = time_partitioning.type_
        out.require_partition_filter = bool(
            getattr(time_partitioning, "require_partition_filter", False)
            or getattr(meta, "require_partition_filter", False))
    elif range_partitioning is not None:
        out.partition_kind = "range"
        out.partition_field = range_partitioning.field
    elif (out.table_type or "TABLE") == "TABLE":
        out.partition_kind = "none"
    else:
        # A view exposes no partition metadata. Unknown, not absent.
        out.partition_kind = "unadvertised"
    out.clustering = list(getattr(meta, "clustering_fields", None) or [])
    return out


def resolve_columns(description: TableDescription) -> dict[str, Any]:
    """Match the roles the reconstruction needs to the table's real columns."""
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for role, candidates in COLUMN_ROLES.items():
        for candidate in candidates:
            if candidate in description.columns:
                mapping[role] = candidate
                break
        else:
            unmatched.append(role)
    missing_required = [r for r in REQUIRED_ROLES if r not in mapping]
    return {
        "mapping": mapping,
        "unmatched_roles": unmatched,
        "missing_required_roles": missing_required,
        "missing_flow_roles": [r for r in FLOW_ROLES if r not in mapping],
        "usable_for_call_tree": not missing_required,
        "path_column_is_repeated": (
            description.modes.get(mapping.get("call_path", ""), "") == "REPEATED"),
        "candidates_considered": {r: list(c) for r, c in COLUMN_ROLES.items()},
    }


# --- the query ----------------------------------------------------------------

def _validate_dates(dates: Iterable[str]) -> list[str]:
    """Accept only `YYYY-MM-DD`, and only a real calendar date.

    These become SQL literals so that partition pruning is certain, which means
    they must be impossible to abuse. Anything else is an error.
    """
    out = []
    for value in dates:
        text = str(value)
        if not _DATE_RE.match(text):
            raise TxBenchError(f"not an ISO date: {value!r}")
        date.fromisoformat(text)                      # raises on 2025-02-30
        out.append(text)
    return sorted(set(out))


def resolve_time_bound(description: TableDescription) -> dict[str, Any]:
    """The column an explicit UTC range is applied to, and how it was chosen.

    A declared partition field is used when there is one. When there is not -
    a view, for instance - a known timestamp column is bounded explicitly
    instead. That is not a guess about the storage underneath; it is a literal
    range that prunes if anything can prune and is correct either way.
    """
    declared = description.partition_field
    if description.partition_kind == "time" and declared:
        # The pseudocolumns of an ingestion-time partitioned table are not in
        # the schema, so they are named here rather than looked up.
        pseudo = {"_PARTITIONTIME": "TIMESTAMP", "_PARTITIONDATE": "DATE"}
        kind = pseudo.get(declared.upper(), description.columns.get(declared, "")).upper()
        # A declared field absent from the schema is not usable: a predicate on
        # a column that does not exist would fail the job, not bound the scan.
        if kind in ("TIMESTAMP", "DATETIME", "DATE"):
            return {"column": declared, "type": kind,
                    "basis": "declared_partition_field",
                    "granularity": (description.partition_type or "DAY").upper()}
    for candidate in TIME_BOUND_CANDIDATES:
        kind = description.columns.get(candidate, "").upper()
        if kind in ("TIMESTAMP", "DATETIME", "DATE"):
            return {"column": candidate, "type": kind,
                    "basis": ("view_without_partition_metadata"
                              if (description.table_type or "TABLE") != "TABLE"
                              else "unpartitioned_table"),
                    # Unknown, so the narrowest useful span is used. A day range
                    # still prunes a coarser partition; it just cannot be sure
                    # how much it saved until the dry run prices it.
                    "granularity": None}
    return {"column": None, "type": None, "basis": "no_time_column", "granularity": None}


def _spans(days: Sequence[str], blocks_by_date: dict[str, Sequence[int]] | None,
           granularity: str | None) -> list[tuple[str, str, list[int]]]:
    """Group the requested days into scan spans: one SQL clause each.

    Days that land in the same span are merged, so a span is never scanned
    twice. When the declared granularity is monthly, each day widens to the
    whole UTC month it touches, because that is what a monthly partition costs
    whatever the predicate says.
    """
    monthly = (granularity or "").upper() in ("MONTH", "YEAR")
    yearly = (granularity or "").upper() == "YEAR"
    buckets: dict[tuple[str, str], set[int]] = {}
    for day in days:
        parsed = date.fromisoformat(day)
        if yearly:
            start = parsed.replace(month=1, day=1)
            end = start.replace(year=start.year + 1)
        elif monthly:
            start = parsed.replace(day=1)
            end = (start + timedelta(days=31)).replace(day=1)
        else:
            start, end = parsed, parsed + timedelta(days=1)
        key = (start.isoformat(), end.isoformat())
        blocks = list((blocks_by_date or {}).get(day) or [])
        buckets.setdefault(key, set()).update(int(b) for b in blocks)
    return [(start, end, sorted(blocks)) for (start, end), blocks in sorted(buckets.items())]


def _range_predicate(column: str, column_type: str,
                     spans: Sequence[tuple[str, str, Sequence[int]]],
                     block_column: str | None = None) -> str:
    """An explicit half-open UTC range per span, narrowed by block number.

    The block numbers come from the frozen manifest, so each is an exact
    integer and a transaction sits in exactly one block. Adding them can only
    narrow the scan; if a manifest block number were ever wrong the hash would
    return no rows and be reported as unmatched, never matched to other data.
    """
    kind = column_type.upper()
    clauses = []
    for start, end, blocks in spans:
        if kind == "DATE":
            bound = f'{column} >= DATE "{start}" AND {column} < DATE "{end}"'
        elif kind in ("TIMESTAMP", "DATETIME"):
            cast = "TIMESTAMP" if kind == "TIMESTAMP" else "DATETIME"
            bound = f'{column} >= {cast}("{start}") AND {column} < {cast}("{end}")'
        else:
            raise TxBenchError(f"time column {column} has type {column_type}, which this "
                               "query builder does not know how to bound")
        if blocks and block_column:
            listed = ", ".join(str(int(b)) for b in blocks)
            bound += f" AND {block_column} IN ({listed})"
        clauses.append(f"({bound})")
    return "(" + "\n       OR ".join(clauses) + ")"


@dataclass
class TraceQuery:
    sql: str
    dates: list[str]
    hashes: list[str]
    mapping: dict[str, str]
    partition_field: str
    partition_type: str
    spans: list[tuple[str, str, list[int]]] = field(default_factory=list)
    time_bound: dict[str, Any] = field(default_factory=dict)
    blocks_narrowed: int = 0

    def parameters(self):
        from google.cloud import bigquery  # noqa: PLC0415
        return [bigquery.ArrayQueryParameter("hashes", "STRING", self.hashes)]

    def as_dict(self) -> dict[str, Any]:
        return {"sql": self.sql, "distinct_partition_days": len(self.dates),
                "scan_spans": len(self.spans), "hashes": len(self.hashes),
                "blocks_narrowed_to": self.blocks_narrowed,
                "time_bound": self.time_bound, "column_mapping": self.mapping}


def build_query(description: TableDescription, resolved: dict[str, Any], *,
                hashes: Sequence[str], dates: Sequence[str],
                blocks_by_date: dict[str, Sequence[int]] | None = None) -> TraceQuery:
    """A parameterized query over only the needed columns and the needed spans.

    What is refused here is an *unbounded* scan, not an object whose
    partitioning is merely undeclared. A view reports no partition metadata, so
    refusing it would refuse the only Ethereum traces source there is; instead
    the range is written out explicitly against a real timestamp column and the
    dry run reports what that actually costs.
    """
    if not resolved["usable_for_call_tree"]:
        raise NoCallPath(
            f"{description.table} has no call-path column (looked for "
            f"{', '.join(COLUMN_ROLES['call_path'])}), so it cannot supply a call tree. "
            "Missing roles: " + ", ".join(resolved["missing_required_roles"]))
    mapping = resolved["mapping"]
    bound = resolve_time_bound(description)
    if not bound["column"]:
        raise TxBenchError(
            f"{description.table} exposes no timestamp column to bound "
            f"(looked for {', '.join(TIME_BOUND_CANDIDATES)}), so every query against it "
            "would be an unbounded scan. Refused before any bytes move.")
    partition_field = bound["column"]
    partition_type = bound["type"] or "TIMESTAMP"
    days = _validate_dates(dates)
    if blocks_by_date:
        unknown = sorted(set(blocks_by_date) - set(days))
        if unknown:
            raise TxBenchError(
                f"block numbers were supplied for day(s) not in the request: {unknown}")
    spans = _spans(days, blocks_by_date, bound["granularity"])
    predicate = _range_predicate(partition_field, partition_type, spans,
                                 mapping.get("block_number"))

    selected = []
    for role, column in mapping.items():
        selected.append(column if column == role else f"{column} AS {role}")
    if partition_field not in mapping.values():
        selected.append(partition_field)
    columns = ",\n       ".join(sorted(set(selected)))
    hash_column = mapping["tx_hash"]

    sql = f"""\
-- txbench call-trace backfill. Only the columns the reconstruction needs, and
-- only the UTC spans and blocks the pilot hashes fall in.
SELECT {columns}
  FROM `{description.table}`
 WHERE {predicate}
   AND LOWER({hash_column}) IN UNNEST(@hashes)
"""
    return TraceQuery(sql=sql, dates=days, hashes=[h.lower() for h in hashes],
                      mapping=mapping, partition_field=partition_field,
                      partition_type=partition_type, spans=spans, time_bound=bound,
                      blocks_narrowed=sum(len(b) for _, _, b in spans))


def build_init_query(description: TableDescription, resolved: dict[str, Any], *,
                     hashes: Sequence[str], dates: Sequence[str],
                     blocks_by_date: dict[str, Sequence[int]] | None = None) -> TraceQuery:
    """A minimal query for creation init bytes: frame identity plus `action.init`.

    `result.code` is deliberately absent. Varying it through the BlockScan
    adapter produced a byte-identical token stream, so fetching it would buy
    nothing this preprocessing step can read.
    """
    mapping = resolved["mapping"]
    if "action.init" not in description.columns:
        raise TxBenchError(
            f"{description.table} has no `action.init` column, so creation init bytes "
            "cannot be recovered from it.")
    bound = resolve_time_bound(description)
    if not bound["column"]:
        raise TxBenchError(f"{description.table} exposes no timestamp column to bound; "
                           "refused before any bytes move.")
    days = _validate_dates(dates)
    spans = _spans(days, blocks_by_date, bound["granularity"])
    predicate = _range_predicate(bound["column"], bound["type"] or "TIMESTAMP", spans,
                                 mapping.get("block_number"))
    hash_column = mapping["tx_hash"]
    path_column = mapping["call_path"]
    sql = f"""\
-- txbench targeted backfill: creation init bytes only.
SELECT {hash_column} AS tx_hash,
       {path_column} AS call_path,
       action.init AS init
  FROM `{description.table}`
 WHERE {predicate}
   AND LOWER({hash_column}) IN UNNEST(@hashes)
   AND trace_type = 'create'
"""
    return TraceQuery(sql=sql, dates=days, hashes=[h.lower() for h in hashes],
                      mapping={"tx_hash": hash_column, "call_path": path_column,
                               "init": "action.init"},
                      partition_field=bound["column"],
                      partition_type=bound["type"] or "TIMESTAMP", spans=spans,
                      time_bound=bound, blocks_narrowed=sum(len(b) for _, _, b in spans))


# --- pricing and budget -------------------------------------------------------

@dataclass
class Budget:
    """A byte budget that survives across runs, in a small JSON ledger."""
    path: Path
    total_bytes: int | None = None
    spent_bytes: int = 0
    jobs: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def load(path: Path, total_bytes: int | None) -> "Budget":
        if path.is_file():
            raw = json.loads(path.read_text())
            budget = Budget(path=path, total_bytes=raw.get("total_bytes"),
                            spent_bytes=int(raw.get("spent_bytes", 0)),
                            jobs=list(raw.get("jobs", [])))
            if total_bytes is not None:
                budget.total_bytes = total_bytes
            return budget
        return Budget(path=path, total_bytes=total_bytes)

    @property
    def configured(self) -> bool:
        return self.total_bytes is not None

    @property
    def remaining(self) -> int | None:
        if self.total_bytes is None:
            return None
        return max(0, self.total_bytes - self.spent_bytes)

    def ceiling_for(self, estimate: int | None) -> int:
        """The `maximum_bytes_billed` for one job.

        The estimate is a floor, not a promise - a dry run can under-report, so
        the ceiling is the remaining budget. With no budget configured, the
        ceiling is the estimate with a small margin, so an unpriced job still
        cannot run away.
        """
        if self.remaining is None:
            if estimate is None:
                raise BudgetExhausted(
                    "the dry run returned no byte estimate and no byte budget is "
                    "configured, so there is no ceiling to run under. Pass "
                    "--budget-bytes to set one explicitly.")
            return max(int(estimate * 1.25), 10 * 1024 * 1024)
        if self.remaining <= 0:
            raise BudgetExhausted(
                f"the byte budget is spent: {self.spent_bytes} of {self.total_bytes} bytes "
                f"used. Raise bigquery.total_byte_budget or pass --budget-bytes.")
        if estimate is not None and estimate > self.remaining:
            raise BudgetExhausted(
                f"the job is estimated at {estimate} bytes but only {self.remaining} bytes "
                f"of budget remain ({self.spent_bytes} of {self.total_bytes} spent).")
        return self.remaining

    def charge(self, *, billed: int, estimated: int, label: str, job_id: str | None) -> None:
        self.spent_bytes += int(billed)
        self.jobs.append({"label": label, "job_id": job_id, "estimated_bytes": estimated,
                          "billed_bytes": int(billed)})

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"total_bytes": self.total_bytes, "spent_bytes": self.spent_bytes,
                   "remaining_bytes": self.remaining, "jobs": self.jobs}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(canonical_json(payload))   # canonical_json returns bytes
        tmp.replace(self.path)
        return self.path

    def snapshot(self) -> dict[str, Any]:
        return {"configured": self.configured, "total_bytes": self.total_bytes,
                "spent_bytes": self.spent_bytes, "remaining_bytes": self.remaining,
                "jobs": len(self.jobs)}


def dry_run(client, query: TraceQuery) -> dict[str, Any]:
    """Price the query without scanning it."""
    from google.cloud import bigquery  # noqa: PLC0415
    config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False,
                                     query_parameters=query.parameters())
    job = client.query(query.sql, job_config=config)
    raw = getattr(job, "total_bytes_processed", None)
    # A missing estimate is reported as unknown. Folding it to zero would read
    # as "this query is free", which is the one thing it cannot be known to be.
    estimate = int(raw) if raw is not None else None
    return {"estimated_bytes_processed": estimate,
            "estimate_available": estimate is not None,
            "estimated_gib": (round(estimate / 1024 ** 3, 4)
                              if estimate is not None else None),
            "estimate_note": (None if estimate is not None else
                              "BigQuery returned no byte estimate for this dry run, so the "
                              "cost is unknown. A byte budget is then the only ceiling."),
            "partition_days": len(query.dates), "scan_spans": len(query.spans),
            "time_bound": query.time_bound, "hashes": len(query.hashes),
            "cache_hit": bool(getattr(job, "cache_hit", False))}


def execute(client, query: TraceQuery, *, budget: Budget, estimate: int,
            label: str = "pilot") -> dict[str, Any]:
    """Run the query under a hard byte ceiling and charge the budget."""
    from google.cloud import bigquery  # noqa: PLC0415
    ceiling = budget.ceiling_for(estimate)
    config = bigquery.QueryJobConfig(
        query_parameters=query.parameters(),
        maximum_bytes_billed=ceiling,
        use_query_cache=True,
        labels={"project": "txbench", "job": "call-trace-backfill"})
    job = client.query(query.sql, job_config=config)
    rows = [dict(row) for row in job.result()]
    billed = int(getattr(job, "total_bytes_billed", 0) or 0)
    budget.charge(billed=billed, estimated=estimate, label=label,
                  job_id=getattr(job, "job_id", None))
    log.info("bigquery job returned %d rows; %d bytes billed (ceiling %d)",
             len(rows), billed, ceiling)
    return {"rows": rows, "bytes_billed": billed, "bytes_processed":
            int(getattr(job, "total_bytes_processed", 0) or 0),
            "maximum_bytes_billed": ceiling, "job_id": getattr(job, "job_id", None),
            "cache_hit": bool(getattr(job, "cache_hit", False))}

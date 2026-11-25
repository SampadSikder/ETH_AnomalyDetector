"""txbench - historical Ethereum transaction benchmark.

Milestone 1: a reproducible, labeled, auditable dataset plus the evaluation
interface for a five-model comparison (BlockGPT, BlockScan, BlockLens,
DeFiGuard, supervised FT-Transformer). Model training is a later milestone.
"""

__version__ = "0.1.0"

# Bumped whenever the on-disk layout of the raw cache changes in a way that
# makes previously cached payloads non-reusable.
CACHE_SCHEMA_VERSION = 1

# Bumped whenever the manifest column set or normalisation rules change.
MANIFEST_SCHEMA_VERSION = 1

# Bumped whenever a feature view's token definition changes.
FEATURE_SCHEMA_VERSION = 1
